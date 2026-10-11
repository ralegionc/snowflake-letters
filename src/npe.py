"""Neural posterior estimation of growth temperatures from snowflake triplets, trained only on simulations.

Network: the three-view encoder from cnn.py plus log Dmax -> mixture density over (T1, T2), where T1 is the
temperature while the core grew and T2 while the outer crystal grew. Auxiliary heads (columnar vs planar,
capped column, branching) give a transfer check against MASCDB's own habit labels on real images.

Optional domain adaptation (DANN, Ganin et al. 2016): a gradient-reversal domain classifier on the trunk
features, trained against unlabeled real single crystals, so the posterior cannot lean on cues that exist
only in simulations. No sounding information is ever used.

usage: python src/npe.py train <tag> <sims prefix> <dann 0|1> [epochs] [threads]
       python src/npe.py heldout <tag> <sims prefix>    |    python src/npe.py apply <tag>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from cnn import Encoder, RES  # noqa: E402
from sim import THETA_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
K = 10                                  # mixture components
T_MU, T_SD = -16.5, 9.0                 # standardisation of temperatures (prior is U(-32, -1))
T_GRID = np.arange(-32, -0.99, 0.5)
SINGLE = ["planar_crystal", "columnar_crystal", "columnar_planar_combination"]


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class NPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(w=16)
        d = 2 * self.enc.dim + 1
        self.trunk = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mdn = nn.Linear(256, K * 5)    # logit, mu1, mu2, logsd1, logsd2
        self.aux = nn.Linear(256, 3)        # columnar logit, capped logit, branching
        self.dom = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def features(self, img, logD):
        B = img.shape[0]
        e = self.enc(img.reshape(B * 3, 1, RES, RES)).reshape(B, 3, -1)
        return self.trunk(torch.cat([e.mean(1), e.amax(1), logD], 1))

    def heads(self, h):
        m = self.mdn(h).reshape(h.shape[0], K, 5)
        return m[..., 0], m[..., 1:3], m[..., 3:5].clamp(-5, 3), self.aux(h)

    def forward(self, img, logD):
        return self.heads(self.features(img, logD))


def mdn_nll(logit, mu, logsd, y):
    z = (y[:, None, :] - mu) / logsd.exp()
    comp = -0.5 * (z ** 2).sum(-1) - logsd.sum(-1) - np.log(2 * np.pi)
    return -(torch.logsumexp(F.log_softmax(logit, -1) + comp, -1)).mean()


def batch_tensor(images, idx, train, rng):
    x = torch.from_numpy(images[idx].astype(np.float32) / 255.0)
    x = F.interpolate(x, size=RES, mode="bilinear", align_corners=False)
    if train:
        x = torch.rot90(x, int(rng.integers(4)), (2, 3))
        if rng.random() < 0.5:
            x = x.flip(3)
        x = x[:, torch.randperm(3)]
        # photometric jitter: gain and gamma per sample, as a guard against sim-to-real intensity mismatch
        g = torch.from_numpy(rng.uniform(0.6, 1.5, (len(idx), 1, 1, 1)).astype(np.float32))
        gam = torch.from_numpy(rng.uniform(0.7, 1.4, (len(idx), 1, 1, 1)).astype(np.float32))
        x = (x.clamp(min=0) ** gam * g).clamp(0, 1)
    return x


def marginals(logit, mu, logsd):
    """Marginal densities of T1 and T2 on T_GRID (normalised), in degrees C."""
    w = F.softmax(logit, -1).numpy()[:, :, None]
    g = (T_GRID - T_MU) / T_SD
    out = []
    for d in range(2):
        m, s = mu[..., d].numpy()[:, :, None], logsd[..., d].exp().numpy()[:, :, None]
        p = (w * np.exp(-0.5 * ((g - m) / s) ** 2) / s).sum(1)
        out.append(p / p.sum(1, keepdims=True))
    return out


def real_pool():
    t = pd.read_parquet(D / "table.parquet").reset_index(drop=True)
    sel = np.flatnonzero(t.snowflake_class_name.isin(SINGLE).values)
    X = np.ascontiguousarray(np.load(D / "images.npy", mmap_mode="r")[sel])
    logD = torch.from_numpy((np.log(t.flake_Dmax.values[sel, None]) + 7.0).astype(np.float32))
    return X, logD


def train(tag, sims, dann, epochs, threads, lam_max=0.5):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = np.load(D / f"{sims}_images.npy")
    th = np.load(D / f"{sims}_theta.npy")
    N = THETA_NAMES
    Y = torch.from_numpy(((th[:, [N.index("T1"), N.index("T2")]] - T_MU) / T_SD).astype(np.float32))
    logD = torch.from_numpy((th[:, [N.index("logD")]] + 7.0).astype(np.float32))
    aux = torch.from_numpy(np.stack([th[:, N.index("aspect")] > 0, th[:, N.index("capped")] > 0,
                                     th[:, N.index("B")]], 1).astype(np.float32))
    if dann:
        XR, logDR = real_pool()
    n = len(X)
    perm = rng.permutation(n)
    tr = perm[n // 10:]
    model = NPE()
    bs, bs_real = 256, 128
    steps_per_ep = len(tr) // bs + 1
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * steps_per_ep)
    step = 0
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(tr)
        tot, dacc, t0 = 0.0, [], time.time()
        for b in range(0, len(idx), bs):
            i = np.sort(idx[b:b + bs])
            h = model.features(batch_tensor(X, i, True, rng), logD[i])
            logit, mu, logsd, a = model.heads(h)
            loss = mdn_nll(logit, mu, logsd, Y[i])
            loss = loss + 0.3 * (F.binary_cross_entropy_with_logits(a[:, 0], aux[i, 0])
                                 + F.binary_cross_entropy_with_logits(a[:, 1], aux[i, 1])
                                 + F.mse_loss(torch.sigmoid(a[:, 2]), aux[i, 2]))
            if dann:
                lam = lam_max * (2 / (1 + np.exp(-10 * step / (epochs * steps_per_ep))) - 1)
                j = np.sort(rng.choice(len(XR), bs_real, replace=False))
                hr = model.features(batch_tensor(XR, j, True, rng), logDR[j])
                d_logit = model.dom(GradReverse.apply(torch.cat([h, hr]), lam)).squeeze(1)
                d_y = torch.cat([torch.zeros(len(i)), torch.ones(len(j))])
                loss = loss + F.binary_cross_entropy_with_logits(d_logit, d_y)
                dacc.append(((d_logit > 0).float() == d_y).float().mean().item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot += loss.item() * len(i)
        da = f" domain-acc {np.mean(dacc):.3f}" if dann else ""
        print(f"ep{ep} loss {tot / len(idx):.4f}{da} {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), D / f"npe_{tag}.pt")


def report_calibration(th, P1, P2):
    N = THETA_NAMES
    for name, P in [("T1", P1), ("T2", P2)]:
        y = th[:, N.index(name)]
        mean = (P * T_GRID).sum(1)
        cdf = np.cumsum(P, 1)
        rank = np.array([np.interp(v, T_GRID, c) for v, c in zip(y, cdf)])
        prior_mae = np.abs(y - y.mean()).mean()
        print(f"{name}: posterior-mean MAE {np.abs(mean - y).mean():.2f} C (prior {prior_mae:.2f});"
              f" 50% interval coverage {np.mean(np.abs(rank - 0.5) < 0.25):.2f},"
              f" 90% coverage {np.mean(np.abs(rank - 0.5) < 0.45):.2f}")


def run(model, X, idx, logD):
    P1, P2, A = [], [], []
    with torch.no_grad():
        for b in range(0, len(idx), 512):
            i = idx[b:b + 512]
            logit, mu, logsd, a = model(batch_tensor(X, i, False, None), logD[i])
            p1, p2 = marginals(logit, mu, logsd)
            P1.append(p1)
            P2.append(p2)
            A.append(torch.sigmoid(a).numpy())
    return np.concatenate(P1), np.concatenate(P2), np.concatenate(A)


def load_model(tag):
    model = NPE()
    model.load_state_dict(torch.load(D / f"npe_{tag}.pt"), strict=False)  # v1 had no domain head
    return model.eval()


def heldout(tag, sims, threads=12):
    """Score the saved model on the held-out simulations (same split as train())."""
    torch.set_num_threads(threads)
    X = np.load(D / f"{sims}_images.npy", mmap_mode="r")
    th = np.load(D / f"{sims}_theta.npy")
    te = np.sort(np.random.default_rng(0).permutation(len(X))[: len(X) // 10])
    logD = torch.from_numpy((th[:, [THETA_NAMES.index("logD")]] + 7.0).astype(np.float32))
    P1, P2, A = run(load_model(tag), X, te, logD)
    np.savez(D / f"npe_heldout_{tag}.npz", idx=te, p1=P1, p2=P2, aux=A)
    report_calibration(th[te], P1, P2)
    col = th[te, THETA_NAMES.index("aspect")] > 0
    print(f"columnar head accuracy on sims: {np.mean((A[:, 0] > 0.5) == col):.3f}")


def apply(tag, threads=12):
    torch.set_num_threads(threads)
    t = pd.read_parquet(D / "table.parquet").reset_index(drop=True)
    sel = np.flatnonzero(t.snowflake_class_name.isin(SINGLE).values)
    X = np.load(D / "images.npy", mmap_mode="r")
    logD = torch.from_numpy((np.log(t.flake_Dmax.values[:, None]) + 7.0).astype(np.float32))
    P1, P2, A = run(load_model(tag), X, sel, logD)
    np.savez(D / f"npe_real_{tag}.npz", idx=sel, p1=P1, p2=P2, aux=A)
    lab = t.snowflake_class_name.values[sel]
    print("transfer check: mean predicted P(columnar), P(capped) by MASCDB label")
    for c in SINGLE:
        m = lab == c
        print(f"  {c:28s} n={m.sum():5d}  P(col)={A[m, 0].mean():.2f}  P(capped)={A[m, 1].mean():.2f}"
              f"  branching={A[m, 2].mean():.2f}")


if __name__ == "__main__":
    cmd, tag = sys.argv[1], sys.argv[2]
    if cmd == "train":
        train(tag, sys.argv[3], bool(int(sys.argv[4])), int(sys.argv[5]) if len(sys.argv) > 5 else 6,
              int(sys.argv[6]) if len(sys.argv) > 6 else 12)
    elif cmd == "heldout":
        heldout(tag, sys.argv[3])
    else:
        apply(tag)
