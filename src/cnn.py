"""Multi-view CNN: does the snowflake image add information about the air aloft beyond surface weather?

Control: a surface-weather GBM. The network learns the GBM's residual, once with images and once
with the image branch disabled. Folds are grouped by snow episode so no storm spans train and test.

usage: python src/cnn.py [image|scalar] [epochs] [threads]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
TARGETS = ["y_T500", "y_T1500", "y_T3000", "y_RH1500", "c_topT", "c_satfrac", "c_dgz"]
SCALARS = ["env_T", "env_RH", "env_P", "env_FF", "doy_s", "doy_c"]
RES = 64
CAP_PER_HOUR = 40  # particles per hour sampled each epoch, so long storms do not dominate


def prepare():
    t = pd.read_parquet(D / "table.parquet").reset_index(drop=True)
    doy = t.datetime.dt.dayofyear
    t["doy_s"], t["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    s = t[SCALARS].astype(float)
    s = s.fillna(s.median())
    camps = pd.get_dummies(t.campaign).astype(float)
    S = np.concatenate([s.to_numpy(), camps.to_numpy()], 1).astype(np.float32)
    Y = t[TARGETS].to_numpy(np.float32)
    logD = np.log(t.flake_Dmax.to_numpy(np.float32))[:, None]
    return t, S, Y, logD


class Encoder(nn.Module):
    """Small CPU-friendly encoder: strided stem, then three conv/pool stages (~15M MACs per view)."""

    def __init__(self, w=16):
        super().__init__()
        def conv(i, o, stride=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(conv(1, w, 2),                                   # 32x32
                                 conv(w, 2 * w), nn.MaxPool2d(2),                 # 16x16
                                 conv(2 * w, 4 * w), nn.MaxPool2d(2),             # 8x8
                                 conv(4 * w, 8 * w), conv(8 * w, 8 * w))          # 8x8
        self.dim = 8 * w

    def forward(self, x):  # (B*3, 1, H, W)
        return self.net(x).mean((2, 3))


class Net(nn.Module):
    def __init__(self, n_scalar, n_out, use_image):
        super().__init__()
        self.use_image = use_image
        self.enc = Encoder()
        img_dim = 2 * self.enc.dim + 1 if use_image else 0
        self.head = nn.Sequential(nn.Linear(n_scalar + img_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_out))

    def embed(self, img, logD):
        B = img.shape[0]
        e = self.enc(img.reshape(B * 3, 1, RES, RES)).reshape(B, 3, -1)
        return torch.cat([e.mean(1), e.amax(1), logD], 1)

    def forward(self, img, scal, logD):
        z = [scal]
        if self.use_image:
            z.append(self.embed(img, logD))
        return self.head(torch.cat(z, 1))


def to_tensor(images, idx, train):
    x = torch.from_numpy(images[np.sort(idx)].astype(np.float32) / 255.0)
    order = np.argsort(np.argsort(idx))
    x = x[order]
    x = F.interpolate(x, size=RES, mode="bilinear", align_corners=False) if x.shape[-1] != RES else x
    if train:  # snowflakes have no preferred orientation: random rot90/flip per view, shuffled views
        k = np.random.randint(4)
        x = torch.rot90(x, k, (2, 3))
        if np.random.rand() < 0.5:
            x = x.flip(3)
        x = x[:, torch.randperm(3)]
    return x


def sample_epoch(hours, rng):
    idx = []
    for _, rows in hours.items():
        idx.extend(rows if len(rows) <= CAP_PER_HOUR else rng.choice(rows, CAP_PER_HOUR, replace=False))
    return rng.permutation(np.array(idx))


def folds(t, n_splits=5, seed=0):
    return list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(t, groups=t.episode_id))


def gbm_bases(t):
    """Surface-weather GBM (the control). Test rows get predictions from a model fit on the outer
    training fold; training rows get inner out-of-fold predictions, so the CNN never sees a base
    prediction that was fit on its own label. Cached in data/gbm_bases.npz."""
    path = D / "gbm_bases.npz"
    if path.exists():
        z = np.load(path)
        return z["test"], z["train"]
    import lightgbm as lgb
    X = t[["campaign", "doy_s", "doy_c", "env_T", "env_RH", "env_P", "env_FF"]].copy()
    X["campaign"] = X.campaign.astype("category")
    Y = t[TARGETS].to_numpy(float)
    fit = lambda Xa, ya: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.06, num_leaves=31,
                                           min_child_samples=50, subsample=0.8, subsample_freq=1,
                                           n_jobs=12, verbose=-1).fit(Xa, ya)
    test = np.full(Y.shape, np.nan, np.float32)
    train = np.full((5,) + Y.shape, np.nan, np.float32)
    for f, (tr, te) in enumerate(folds(t)):
        for k in range(len(TARGETS)):
            ok = tr[~np.isnan(Y[tr, k])]
            test[te, k] = fit(X.iloc[ok], Y[ok, k]).predict(X.iloc[te])
            for itr, ite in folds(t.iloc[tr], seed=1):
                iok = tr[itr][~np.isnan(Y[tr[itr], k])]
                train[f, tr[ite], k] = fit(X.iloc[iok], Y[iok, k]).predict(X.iloc[tr[ite]])
        print(f"  gbm bases fold {f} done", flush=True)
    np.savez(path, test=test, train=train)
    return test, train


def run_fold(images, S, R, logD, hours_tr, te, use_image, epochs, bs=256, seed=0):
    """Train on residuals R (target minus GBM base); S already includes the base predictions."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_all = np.concatenate(list(hours_tr.values()))
    s_mu, s_sd = np.nanmean(S[tr_all], 0), np.nanstd(S[tr_all], 0) + 1e-6
    y_sd = np.nanstd(R[tr_all], 0) + 1e-6
    Sn, Yn = np.nan_to_num((S - s_mu) / s_sd).astype(np.float32), (R / y_sd).astype(np.float32)
    model = Net(S.shape[1], R.shape[1], use_image)
    steps = epochs * (len(sample_epoch(hours_tr, np.random.default_rng(1))) // bs + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    for ep in range(epochs):
        model.train()
        idx = sample_epoch(hours_tr, rng)
        tot, t0 = 0.0, time.time()
        for b in range(0, len(idx), bs):
            i = idx[b:b + bs]
            img = to_tensor(images, i, True) if use_image else None
            y = torch.from_numpy(Yn[i])
            m = ~torch.isnan(y)
            p = model(img, torch.from_numpy(Sn[i]), torch.from_numpy(logD[i]))
            loss = F.smooth_l1_loss(p[m], y[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(i)
        print(f"  ep{ep} loss {tot / len(idx):.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    preds, embs = [], []
    with torch.no_grad():
        for b in range(0, len(te), 512):
            i = te[b:b + 512]
            img = to_tensor(images, i, False) if use_image else None
            sc, ld = torch.from_numpy(Sn[i]), torch.from_numpy(logD[i])
            preds.append(model(img, sc, ld).numpy())
            if use_image:
                embs.append(model.embed(img, ld).numpy().astype(np.float16))
    P = np.concatenate(preds) * y_sd
    return P, (np.concatenate(embs) if embs else None)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "image"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    t, S0, Y, logD = prepare()
    base_test, base_train = gbm_bases(t)
    images = None
    if mode == "image":  # ~3.6 GB in RAM; random access on a memmap is slow
        images = np.ascontiguousarray(np.load(D / "images.npy", mmap_mode="r"))
    oof = np.full(Y.shape, np.nan, np.float32)
    emb = None
    for f, (tr, te) in enumerate(folds(t)):
        print(f"fold {f}: train {len(tr)} test {len(te)}", flush=True)
        base = base_train[f].copy()
        base[te] = base_test[te]
        S = np.concatenate([S0, base], 1)
        R = Y - base
        hours_tr = {h: g.to_numpy() for h, g in pd.Series(tr).groupby(t.hour_id.values[tr])}
        P, E = run_fold(images, S, R, logD, hours_tr, te, mode == "image", epochs)
        oof[te] = base_test[te] + P
        if E is not None:
            if emb is None:
                emb = np.zeros((len(t), E.shape[1]), np.float16)
            emb[te] = E
        for k, tg in enumerate(TARGETS[:3]):
            g, n = np.nanmean(np.abs(base_test[te, k] - Y[te, k])), np.nanmean(np.abs(oof[te, k] - Y[te, k]))
            print(f"  {tg} fold MAE gbm {g:.3f} -> {mode} {n:.3f}", flush=True)
    pd.DataFrame(oof, columns=TARGETS, index=t.zarr_idx).to_parquet(D / f"oof_cnn_{mode}.parquet")
    pd.DataFrame(base_test, columns=TARGETS, index=t.zarr_idx).to_parquet(D / "oof_gbm_base.parquet")
    if emb is not None:
        np.save(D / "emb_cnn.npy", emb)
    print("saved")
