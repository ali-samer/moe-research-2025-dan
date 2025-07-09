# %% [markdown]
# Block-MoE — exact 30 % positives per batch & 25 km hit radius

# %% imports & setup
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np, random, math, matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Running on:", device)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

# -------------------------------------------------------------------
# 1)  SRBM launch generator  (25 km hit radius to get ~4 % positives)
# -------------------------------------------------------------------
EARTH_R = 6_371_000.0; g = 9.81
ASSETS  = np.array([[35.0,46.0],[35.1,47.5],[34.7,44.8],[33.9,46.7],[34.3,45.2]])

def great_circle_end(lat0, lon0, az, R):
    az  = np.radians(az)
    φ1  = np.radians(lat0); λ1 = np.radians(lon0)
    σ   = R / EARTH_R
    φ2  = np.arcsin(np.sin(φ1)*np.cos(σ) + np.cos(φ1)*np.sin(σ)*np.cos(az))
    λ2  = λ1 + np.arctan2(np.sin(az)*np.sin(σ)*np.cos(φ1),
                          np.cos(σ) - np.sin(φ1)*np.sin(φ2))
    return np.degrees(φ2), (np.degrees(λ2)+540)%360-180

def simulate_srbm(n=80_000, hit_radius_km=25.0):
    lat0 = np.random.uniform(33, 37, n); lon0 = np.random.uniform(44, 48, n)
    alt0 = np.random.normal (150, 40, n)
    utc  = np.random.uniform(0, 86_400, n)
    azim = np.random.uniform(70, 110, n) + np.random.normal(0, 0.15, n)
    pitch= np.random.uniform(35, 55, n) + np.random.normal(0, 0.15, n)

    T   = np.random.uniform(4.5e5, 6.5e5, n)
    m0  = np.random.uniform(8e3, 12e3, n)
    mdot= np.random.uniform(1.2e3, 1.6e3, n)
    t_b = m0/mdot * np.random.uniform(0.9, 1.0, n)
    m_b = m0 - mdot*t_b

    v_bo   = (T/mdot)*np.log(m0/m_b) - g*t_b + np.random.normal(0, 10, n)
    peak_g = (T/m_b)/g + np.random.normal(0, 0.15, n)

    range_m = v_bo**2 * np.sin(np.radians(2*pitch)) / g
    lat_imp, lon_imp = great_circle_end(lat0, lon0, azim, range_m)

    def hits(lat, lon):
        dlat = np.radians(ASSETS[:,0]-lat); dlon = np.radians(ASSETS[:,1]-lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(np.radians(ASSETS[:,0]))*np.sin(dlon/2)**2
        d = 2*EARTH_R*np.arcsin(np.sqrt(a))
        return (d < hit_radius_km*1000).any()
    y = np.fromiter((hits(a,b) for a,b in zip(lat_imp, lon_imp)), np.float32)

    ir_T  = (T/5e5)**0.25 * 2600 + np.random.normal(0,25,n)
    rcs   = np.digitize(m0, [9e3, 11e3])
    ltype = np.random.choice([0,1], n, p=[.7,.3])

    X = np.column_stack([
        lat0, lon0, alt0, utc,
        azim, pitch, t_b, peak_g, v_bo,
        ir_T, rcs, ltype
    ]).astype(np.float32)

    return torch.tensor(X), torch.tensor(y).unsqueeze(1)

# -------------------------------------------------------------------
# 2)  Block-MoE (same architecture)
# -------------------------------------------------------------------
BLOCKS = {"AB": slice(0,4), "CD": slice(4,9), "EF": slice(9,12)}

class BlockExpert(nn.Module):
    def __init__(self, d_in, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(),
            nn.Linear(h, h//2), nn.ReLU(),
            nn.Linear(h//2, 1)
        )
    def forward(self, x): return self.net(x)

class BlockMoE(nn.Module):
    def __init__(self, full_dim):
        super().__init__()
        self.slices = list(BLOCKS.values())
        self.gate   = nn.Sequential(
            nn.Linear(full_dim, 32), nn.ReLU(),
            nn.Linear(32, len(self.slices))
        )
        self.experts = nn.ModuleList([
            BlockExpert(slc.stop-slc.start) for slc in self.slices
        ])
    def forward(self, x, return_w=False):
        w = torch.softmax(self.gate(x), 1)
        outs = [exp(x[:, slc]) for slc, exp in zip(self.slices, self.experts)]
        logits = torch.cat(outs, 1)
        yhat = (w * logits).sum(1, keepdim=True)
        return (yhat, w) if return_w else yhat

# -------------------------------------------------------------------
# 3)  Exact-ratio sampler + training loop
# -------------------------------------------------------------------
def sampler_for_ratio(labels, target_pos=0.3):
    pos = (labels==1).sum().item(); neg = len(labels)-pos
    w_pos = 1.0
    # solve for w_neg so that expected pos rate = target_pos
    w_neg = (pos/neg) * (1-target_pos)/target_pos
    weights = torch.where(labels==1, w_pos, w_neg)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def run_demo(epochs=50, batch=2048, lr=1e-3, alpha_bal=1e-3):
    X, y = simulate_srbm()
    print(f"Raw hit-rate: {y.mean()*100:.2f}%")

    ds = TensorDataset(X, y)
    tr_ds, va_ds = random_split(ds, [int(.8*len(ds)), int(.2*len(ds))])

    ys_tr = torch.cat([t[1] for t in tr_ds]).squeeze()
    sampler = sampler_for_ratio(ys_tr, target_pos=0.3)

    train_dl = DataLoader(tr_ds, batch_size=batch, sampler=sampler, pin_memory=True)
    val_dl   = DataLoader(va_ds, batch_size=batch, pin_memory=True)

    # sanity-check batch composition
    xb_sample, yb_sample = next(iter(train_dl))
    print(f"Positives in a sampled batch: {yb_sample.sum().item()} / {len(yb_sample)}")

    model = BlockMoE(full_dim=X.size(1)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    tr_loss, va_loss, va_acc, va_auc = [], [], [], []

    for ep in trange(epochs, desc="Epoch", dynamic_ncols=True):
        model.train(); run_loss=seen=0
        for xb, yb in tqdm(train_dl, leave=False, dynamic_ncols=True):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit, w = model(xb, return_w=True)
            loss = criterion(logit, yb)
            entropy = (w.mean(0)*torch.log(w.mean(0)+1e-9)).sum()
            (loss + alpha_bal*entropy).backward()
            opt.step()
            run_loss += loss.item()*yb.size(0); seen += yb.size(0)
        tr_loss.append(run_loss/seen)

        # validation
        model.eval(); tot=ls=correct=0; probs=[]; targ=[]
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logit = model(xb); p = torch.sigmoid(logit)
                ls += criterion(logit, yb).item()*yb.size(0); tot += yb.size(0)
                correct += ((p>.5)==yb).sum().item()
                probs.append(p); targ.append(yb)
        va_loss.append(ls/tot)
        acc = correct/tot; va_acc.append(acc)
        pr = torch.cat(targ).cpu().numpy().ravel()
        pp = torch.cat(probs).cpu().numpy().ravel()
        va_auc.append(roc_auc_score(pr, pp)); sched.step()

    # plot
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(tr_loss, label="train BCE")
    ax1.plot(va_loss, label="val BCE")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(va_acc, label="val acc", color="tab:green")
    ax2.plot(va_auc, label="val AUC", color="tab:red", ls="--")
    ax2.set_ylim(0,1); ax2.set_ylabel("acc / AUC"); ax2.legend(loc="upper right")
    plt.title("Balanced-batch Block-MoE"); plt.tight_layout(); plt.show()

    prec, rec, _, _ = precision_recall_fscore_support(
        pr, (pp>.5), average="binary", zero_division=0
    )
    print(f"Positives in val set: {int(pr.sum())} / {len(pr)}")
    print(f"Final val  |  loss {va_loss[-1]:.4f}"
          f"  |  acc {acc*100:.2f}%"
          f"  |  AUC {va_auc[-1]:.3f}"
          f"  |  precision {prec:.2f}"
          f"  |  recall {rec:.2f}")

run_demo()
