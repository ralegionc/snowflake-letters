"""Hourly comparison of all models, with an episode-level bootstrap on the image-vs-control gaps."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
TARGETS = ["y_T500", "y_T1500", "y_T3000", "y_RH1500", "c_topT", "c_satfrac", "c_dgz"]


def hourly(t, pred, target):
    h = pd.DataFrame({"h": t.hour_id.values, "day": t.episode_id.values, "y": t[target].values, "p": pred})
    h = h.dropna().groupby("h").agg(y=("y", "mean"), p=("p", "mean"), day=("day", "first"), n=("y", "size"))
    return h


def boot_gap(ha, hb, target, n=2000, seed=0):
    """Bootstrap over snow episodes: MAE(b) - MAE(a); negative means b is better."""
    rng = np.random.default_rng(seed)
    days = ha.day.unique()
    err_a, err_b = (ha.p - ha.y).abs(), (hb.p - hb.y).abs()
    by_day = {d: np.flatnonzero(ha.day.values == d) for d in days}
    gaps = []
    for _ in range(n):
        idx = np.concatenate([by_day[d] for d in rng.choice(days, len(days))])
        gaps.append(err_b.values[idx].mean() - err_a.values[idx].mean())
    return np.percentile(gaps, [2.5, 50, 97.5])


if __name__ == "__main__":
    t = pd.read_parquet(D / "table.parquet").reset_index(drop=True)
    models = {}
    b = pd.read_parquet(D / "oof_baselines.parquet")
    for name in ["B1 +surface wx", "B2 +descriptors"]:
        models[f"GBM {name}"] = {tg: b[(tg, name)].reindex(t.zarr_idx).values for tg in TARGETS if (tg, name) in b}
    g = D / "oof_gbm_base.parquet"
    if g.exists():
        o = pd.read_parquet(g).reset_index(drop=True)
        models["GBM control (cnn folds)"] = {tg: o[tg].values for tg in TARGETS}
    for mode in ["scalar", "image"]:
        p = D / f"oof_cnn_{mode}.parquet"
        if p.exists():
            o = pd.read_parquet(p).reset_index(drop=True)
            models[f"NN {mode}"] = {tg: o[tg].values for tg in TARGETS}

    rows = []
    for tg in TARGETS:
        for m, preds in models.items():
            if tg not in preds:
                continue
            h = hourly(t, preds[tg], tg)
            r = {"target": tg, "model": m, "MAE_hour": (h.p - h.y).abs().mean()}
            if tg == "c_dgz":
                r["AUC_hour"] = roc_auc_score(h.y >= 0.5, h.p)
            rows.append(r)
    res = pd.DataFrame(rows)
    print(res.pivot(index="model", columns="target", values="MAE_hour").round(3).to_string())
    print(res[res.target == "c_dgz"][["model", "AUC_hour"]].round(3).to_string(index=False))

    for ctrl in ["GBM control (cnn folds)", "NN scalar"]:
        if "NN image" not in models or ctrl not in models:
            continue
        print(f"\nNN image minus {ctrl}, hourly MAE, 95% CI over episodes (negative = images help)")
        for tg in TARGETS:
            ha, hb = hourly(t, models[ctrl][tg], tg), hourly(t, models["NN image"][tg], tg)
            lo, mid, hi = boot_gap(ha, hb, tg)
            rel = mid / (ha.p - ha.y).abs().mean() * 100
            print(f"  {tg:10s} {mid:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({rel:+.1f}%)")
    res.to_csv(ROOT / "results_hourly.csv", index=False)
