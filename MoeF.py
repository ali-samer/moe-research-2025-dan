# %% [markdown]
# Block-MoE on real missile-launch data (11 features → 3 experts)

# %% imports & setup
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pandas as pd, numpy as np, random, matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using:", device)

torch.manual_seed(0); np.random.seed(0); random.seed(0)

# ------------------------------------------------------------------
# 1) LOAD & PREP DATA  (edit path / label column if needed)
# ------------------------------------------------------------------
CSV_PATH   = "missile_launch.csv"   # <-- update
LABEL_COL  = "converged"            # 0 / 1
df = pd.read_csv(CSV_PATH)

y_np = df[LABEL_COL].astype(float).values.reshape(-1,1)
X_np = df.drop(columns=[LABEL_COL]).values.astype(np.float32)    # 11 feats

# standardise features (fit on full set for simplicity)
scaler = StandardScaler().fit(X_np)
X_np   = scaler.transform(X_np)

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

print("Dataset shape:", X.shape)

# ------------------------------------------------------------------
# 2) DEFINE FEATURE BLOCKS  (3 experts)
#    Adjust slices/indices to match domain knowledge
# ------------------------------------------------------------------
BLOCKS = {
    "AB": slice(0, 4),     # first 4 features
    "CD": slice(4, 8),     # next 4
    "EF": slice(8, 11),    # last 3
}

# ------------------------------------------------------------------
# 3) MODEL
# ------------------------------------------------------------------
class BlockExpert(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid//2), nn.ReLU(),
            nn.Linear(hid//2, 1)
        )
    def forward(self, x): return self.net(x)

class BlockMoE(nn.Module):
    def __init__(self, full_dim, blocks):
        super().__init__()
        self.blocks = list(blocks.values())
        self.gate   = nn.Sequential(
            nn.Linear(full_dim, 32), nn.ReLU(),
            nn.Linear(32, len(self.blocks))
        )
        self.experts = nn.ModuleList([
            BlockExpert(in_dim=blk.stop-blk.start) for blk in self.blocks
        ])
    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=1)               # [B,E]
        outs = [exp(x[:, blk]) for blk, exp in zip(self.blocks, self.experts)]
        logits = torch.cat(outs, dim=1)                      # [B,E]
        return (w * logits).sum(1, keepdim=True)             # [B,1]

# ------------------------------------------------------------------
# 4) TRAIN / VALIDATE
# ------------------------------------------------------------------
@torch.no_grad()
def val_step(model, loader):
    model.eval(); tot, loss_sum, correct, p_all, y_all = 0,0,0,[],[]
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logit  = model(xb)
        loss   = F.binary_cross_entropy_with_logits(logit, yb, reduction="sum")
        p      = torch.sigmoid(logit)
        loss_sum += loss.item();  tot += yb.numel()
        correct  += ((p>.5)==yb).sum().item()
        p_all.append(p); y_all.append(yb)
    p = torch.cat(p_all).cpu().numpy().ravel()
    y = torch.cat(y_all).cpu().numpy().ravel()
    auc = roc_auc_score(y, p)
    return loss_sum/tot, correct/tot, auc

def run_train(epochs=40, batch=1024, lr=1e-3):
    ds = TensorDataset(X, y)
    tr_ds, va_ds = random_split(ds, [int(.8*len(ds)), int(.2*len(ds))])
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=batch, pin_memory=True)

    model = BlockMoE(full_dim=X.shape[1], blocks=BLOCKS).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_loss, va_loss, va_acc, va_auc = [], [], [], []

    for ep in trange(epochs, desc="Epoch", dynamic_ncols=True):
        model.train(); run, n = 0.0, 0
        bar = tqdm(tr_dl, leave=False, dynamic_ncols=True)
        for xb, yb in bar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward(); opt.step()
            run += loss.item()*yb.size(0); n += yb.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        tr_loss.append(run/n)

        vl, acc, auc = val_step(model, va_dl)
        va_loss.append(vl); va_acc.append(acc); va_auc.append(auc)
        sched.step()

    # ---------- plot ----------
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(tr_loss, label="train loss", color="tab:blue")
    ax1.plot(va_loss, label="val loss",   color="tab:orange")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(va_acc, label="val acc", color="tab:green")
    ax2.plot(va_auc, label="val AUC", color="tab:red", linestyle="--")
    ax2.set_ylabel("accuracy / AUC"); ax2.set_ylim(0,1); ax2.legend(loc="upper right")
    plt.title("Block-MoE on real launch data"); plt.tight_layout(); plt.show()

    print(f"\nFinal val  |  loss {va_loss[-1]:.4f}  |  acc {va_acc[-1]*100:.2f}%  |  AUC {va_auc[-1]:.3f}")

run_train()
