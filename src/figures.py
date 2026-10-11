"""Figures: hourly skill vs the scalar control, and the snowflakes that move the network's estimate most."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D, FIG = ROOT / "data", ROOT / "figures"
LABEL = {"y_T1500": "T 1.5 km above site (°C)", "y_T3000": "T 3 km above site (°C)",
         "c_topT": "Cloud-top T (°C)", "y_RH1500": "RH 1.5 km above site (%)"}


def hourly(t, p, tg):
    h = pd.DataFrame({"h": t.hour_id.values, "y": t[tg].values, "p": p, "c": t.campaign.values})
    return h.dropna().groupby("h").agg(y=("y", "mean"), p=("p", "mean"), c=("c", "first"))


def scatter(t, img, sca):
    tgs = ["y_T1500", "y_T3000", "c_topT"]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex="col", sharey="col")
    for j, tg in enumerate(tgs):
        for i, (name, o) in enumerate([("surface weather only", sca), ("+ snowflake images", img)]):
            h = hourly(t, o[tg].values, tg)
            a = ax[i, j]
            a.scatter(h.y, h.p, s=4, alpha=0.4, c=pd.factorize(h.c)[0], cmap="tab10")
            lo, hi = h.y.min(), h.y.max()
            a.plot([lo, hi], [lo, hi], "k-", lw=0.8)
            a.set_title(f"{name}\nhourly MAE {np.abs(h.p - h.y).mean():.2f}", fontsize=10)
            if i == 1:
                a.set_xlabel("radiosonde: " + LABEL[tg])
            if j == 0:
                a.set_ylabel("predicted")
    fig.tight_layout()
    fig.savefig(FIG / "hourly_skill.png", dpi=130)


def gallery(t, img, sca, tg="c_topT", n=10):
    """Snowflakes whose image pushes the estimate furthest warm vs cold relative to the control."""
    x = np.load(D / "images.npy", mmap_mode="r")
    shift = img[tg].values - sca[tg].values
    order = np.argsort(shift)
    picks = {"pushed colder": order[:n * 20:20], "pushed warmer": order[::-1][:n * 20:20]}
    fig, ax = plt.subplots(2, n, figsize=(n * 1.3, 3.2))
    for r, (name, idx) in enumerate(picks.items()):
        for c, k in enumerate(idx):
            ax[r, c].imshow(255 - x[k, 0], cmap="gray", vmin=0, vmax=255)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            ax[r, c].set_title(f"{shift[k]:+.1f}°", fontsize=8)
        ax[r, 0].set_ylabel(name, fontsize=9)
    fig.suptitle(f"Image effect on {LABEL[tg]} relative to the surface-weather control", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"gallery_{tg}.png", dpi=130)


if __name__ == "__main__":
    t = pd.read_parquet(D / "table.parquet").reset_index(drop=True)
    img = pd.read_parquet(D / "oof_cnn_image.parquet").reset_index(drop=True)
    sca = pd.read_parquet(D / "oof_cnn_scalar.parquet").reset_index(drop=True)
    scatter(t, img, sca)
    gallery(t, img, sca, "c_topT")
    gallery(t, img, sca, "y_T3000")
    print("figures written")
