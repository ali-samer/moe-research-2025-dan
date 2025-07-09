# %% [markdown]
# MH-MoE — Apple-Silicon / GPU-aware, tidy tqdm bars, full metrics

# %% imports & setup
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score
import numpy as np, random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

# ---- choose device -----------------------------------------------------------
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using {device}")

torch.manual_seed(0); np.random.seed(0); random.seed(0)

# ---- synthetic, balanced labels ---------------------------------------------
def make_data(n=100_000, d=20, n_classes=4):
    X = torch.randn(n, d)
    mtype = torch.randint(0, n_classes, (n, 1)).float()
    logit = 2.0 * (
        1.2*X[:,0] - .8*X[:,5] + .6*X[:,11] - 1.4*X[:,17] + .9*mtype.squeeze()
    )
    y = (logit > 0).float().unsqueeze(1)          # 50 / 50 split
    X = torch.cat([X, mtype], dim=1)
    return X, y

# ---- MoE ---------------------------------------------------------------------
class MHMoE(nn.Module):
    def __init__(self, in_dim, n_experts=6, hidden=256, p_drop=0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_experts)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(hidden//2, 1)
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=1)               # [B,E]
        logits = torch.cat([exp(x) for exp in self.experts], dim=1)
        return (w * logits).sum(1, keepdim=True)

# ---- helpers -----------------------------------------------------------------
@torch.no_grad()
def val_step(model, loader):
    model.eval()
    tot_loss, tot, correct, probs, y_true = 0, 0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logit = model(xb)
        loss  = F.binary_cross_entropy_with_logits(logit, yb, reduction="sum")
        p     = torch.sigmoid(logit)
        tot_loss += loss.item(); tot += yb.numel()
        correct  += ((p > .5) == yb).sum().item()
        probs.append(p); y_true.append(yb)
    probs = torch.cat(probs).cpu().numpy().ravel()
    y_true = torch.cat(y_true).cpu().numpy().ravel()
    auc = roc_auc_score(y_true, probs)
    return tot_loss / tot, correct / tot, auc

# ---- training loop -----------------------------------------------------------
def run_demo(epochs=60, batch=4096, lr=1e-3):
    # data
    X, y = make_data()
    ds = TensorDataset(X, y)
    train_ds, val_ds = random_split(ds, [int(.8*len(ds)), int(.2*len(ds))])
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, pin_memory=True)

    # model
    model = MHMoE(in_dim=X.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_loss, va_loss, va_acc, va_auc = [], [], [], []

    for epoch in trange(epochs, desc="Epoch", dynamic_ncols=True):
        model.train()
        run_loss, seen = 0.0, 0
        batch_bar = tqdm(train_dl, leave=False, dynamic_ncols=True)
        for xb, yb in batch_bar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward(); opt.step()
            run_loss += loss.item() * yb.size(0); seen += yb.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        tr_loss.append(run_loss / seen)

        vl, acc, auc = val_step(model, val_dl)
        va_loss.append(vl); va_acc.append(acc); va_auc.append(auc)
        sched.step()

    # ---- plot ----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(tr_loss, label="train loss", color="tab:blue")
    ax1.plot(va_loss, label="val loss",   color="tab:orange")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(va_acc, label="val acc",  color="tab:green")
    ax2.plot(va_auc, label="val AUC",  color="tab:red", linestyle="--")
    ax2.set_ylabel("accuracy / AUC"); ax2.set_ylim(0,1)
    ax2.legend(loc="upper right")

    plt.title("MH-MoE | loss, accuracy & AUC")
    plt.tight_layout(); plt.show()

    # ---- final metrics -------------------------------------------------------
    print(f"\nFinal validation metrics "
          f"|  loss: {va_loss[-1]:.4f}  "
          f"|  acc: {va_acc[-1]*100:.2f}%  "
          f"|  AUC: {va_auc[-1]:.3f}")

run_demo()
