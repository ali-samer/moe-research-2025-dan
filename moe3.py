# %% [markdown]
# Feature-Block MoE (single missile class, 3 experts on AB | CD | EF)

# %% imports & setup
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score
import numpy as np, random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Running on:", device)

torch.manual_seed(0); np.random.seed(0); random.seed(0)

# ---------------------------------------------------------------------
# 1) Synthetic data : 12 numeric launch-time features  (no class column)
#    Blocks: AB = 0..3,  CD = 4..7,  EF = 8..11
# ---------------------------------------------------------------------
def make_data(n=80_000):                    # 80 k samples to keep things quick
    X = torch.randn(n, 12)
    # ground-truth logit uses a different recipe for each block
    logit = (
        (1.5*X[:,0] - 0.8*X[:,1] + 0.7*X[:,2] - 0.4*X[:,3]) +     # AB part
        (0.9*X[:,4] - 1.1*X[:,5] + 1.2*X[:,6] - 0.7*X[:,7]) +     # CD part
        (1.3*X[:,8] + 0.6*X[:,9] - 1.0*X[:,10] + 0.5*X[:,11])     # EF part
    )
    y = (logit > 0).float().unsqueeze(1)          # perfectly separable, 50/50
    return X, y

# ---------------------------------------------------------------------
# 2) MoE with feature-block experts
# ---------------------------------------------------------------------
BLOCKS = {
    "AB": slice(0, 4),     # features 0-3
    "CD": slice(4, 8),     # features 4-7
    "EF": slice(8, 12),    # features 8-11
}

class BlockExpert(nn.Module):
    """Expert that only sees its own slice of x."""
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x_slice):                 # x_slice already sliced
        return self.net(x_slice)                # [B,1]

class BlockMoE(nn.Module):
    def __init__(self, full_dim, hidden_gate=32):
        super().__init__()
        self.blocks = list(BLOCKS.values())     # slice objects
        self.gate   = nn.Sequential(
            nn.Linear(full_dim, hidden_gate), nn.ReLU(),
            nn.Linear(hidden_gate, len(self.blocks))
        )
        self.experts = nn.ModuleList([
            BlockExpert(in_dim=slc.stop - slc.start) for slc in self.blocks
        ])

    def forward(self, x):
        # gate weights (softmax over experts)
        w = torch.softmax(self.gate(x), dim=1)              # [B, 3]
        # expert outputs
        outs = []
        for slc, expert in zip(self.blocks, self.experts):
            outs.append(expert(x[:, slc]))
        expert_logits = torch.cat(outs, dim=1)              # [B, 3]
        # weighted sum → single logit
        return (w * expert_logits).sum(1, keepdim=True)     # [B,1]

# ---------------------------------------------------------------------
# 3) validation helper
# ---------------------------------------------------------------------
@torch.no_grad()
def val_step(model, loader):
    model.eval()
    tot, loss_sum, correct, all_p, all_y = 0, 0.0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logit  = model(xb)
        loss   = F.binary_cross_entropy_with_logits(logit, yb, reduction="sum")
        p      = torch.sigmoid(logit)
        loss_sum += loss.item();  tot += yb.numel()
        correct  += ((p > .5) == yb).sum().item()
        all_p.append(p);  all_y.append(yb)
    p = torch.cat(all_p).cpu().numpy().ravel()
    y = torch.cat(all_y).cpu().numpy().ravel()
    auc = roc_auc_score(y, p)
    return loss_sum / tot, correct / tot, auc

# ---------------------------------------------------------------------
# 4) training loop with tidy bars
# ---------------------------------------------------------------------
def run_demo(epochs=40, batch=2048, lr=5e-4):
    X, y = make_data()
    ds = TensorDataset(X, y)
    train_ds, val_ds = random_split(ds, [int(.8*len(ds)), int(.2*len(ds))])
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, pin_memory=True)

    model = BlockMoE(full_dim=X.size(1)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_loss, va_loss, va_acc, va_auc = [], [], [], []

    for ep in trange(epochs, desc="Epoch", dynamic_ncols=True):
        model.train()
        running, seen = 0.0, 0
        bar = tqdm(train_dl, leave=False, dynamic_ncols=True)
        for xb, yb in bar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward(); opt.step()
            running += loss.item()*yb.size(0); seen += yb.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        tr_loss.append(running / seen)

        vl, acc, auc = val_step(model, val_dl)
        va_loss.append(vl); va_acc.append(acc); va_auc.append(auc)
        sched.step()

    # -------- plot --------
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(tr_loss, label="train loss", color="tab:blue")
    ax1.plot(va_loss, label="val loss",   color="tab:orange")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(va_acc, label="val acc", color="tab:green")
    ax2.plot(va_auc, label="val AUC", color="tab:red", linestyle="--")
    ax2.set_ylabel("accuracy / AUC"); ax2.set_ylim(0,1); ax2.legend(loc="upper right")
    plt.title("Block-MoE on a single missile class"); plt.tight_layout(); plt.show()

    print(f"\nFinal val  |  loss {va_loss[-1]:.4f}  |  acc {va_acc[-1]*100:.2f}%  |  AUC {va_auc[-1]:.3f}")

run_demo()
