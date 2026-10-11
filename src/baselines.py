"""Gradient-boosting ladder: what do surface weather and snowflake descriptors say about the air aloft?

Folds are grouped by snow episode (no gap over 6 h) so no storm is split across train and test. Each model is scored per
particle and per hour (predictions averaged over all particles seen in that hour).
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
TARGETS = ["y_T500", "y_T1500", "y_T3000", "y_RH1500", "c_topT", "c_satfrac", "c_dgz"]


def feature_sets(t):
    t = t.copy()
    doy = t.datetime.dt.dayofyear
    t["doy_s"], t["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    t["camp"] = t.campaign.astype("category")
    t["log_Dmax"] = np.log(t.flake_Dmax)
    base = ["camp", "doy_s", "doy_c"]
    env = ["env_T", "env_RH", "env_P", "env_FF"]
    desc = [c for c in t.columns if c.startswith("d_")] + ["log_Dmax", "fallspeed", "riming_deg_level",
                                                         "snowflake_class_id", "snowflake_class_prob"]
    sets = {
        "B0 site+season": base,
        "B1 +surface wx": base + env,
        "B2 +descriptors": base + env + desc,
        "D  descriptors+site only": base + desc,
    }
    return t, sets


def run(t, feats, target, n_splits=5, seed=0):
    oof = np.full(len(t), np.nan)
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X, y = t[feats], t[target].to_numpy()
    for tr, te in gkf.split(X, y, t.episode_id):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=8, verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof


def score(t, target, pred):
    y = t[target].to_numpy()
    per_particle = np.nanmean(np.abs(pred - y))
    h = pd.DataFrame({"h": t.hour_id.values, "y": y, "p": pred}).groupby("h").mean()
    per_hour = np.nanmean(np.abs(h.p - h.y))
    if target == "c_dgz":  # binary: report hourly AUC in place of particle MAE
        from sklearn.metrics import roc_auc_score
        per_particle = roc_auc_score(h.y >= 0.5, h.p)
    return per_particle, per_hour


if __name__ == "__main__":
    t = pd.read_parquet(D / "table.parquet")
    t, sets = feature_sets(t)
    rows, preds = [], {}
    for target in TARGETS:
        tt = t[t[target].notna()]
        for name, feats in sets.items():
            p = run(tt, feats, target)
            preds[(target, name)] = pd.Series(p, index=tt.index)
            pp, ph = score(tt, target, p)
            rows.append({"target": target, "model": name, "MAE_particle": pp, "MAE_hour": ph})
            print(target, name, round(pp, 3), round(ph, 3), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results_baselines.csv", index=False)
    pd.DataFrame(preds).to_parquet(D / "oof_baselines.parquet")
    print(res.pivot(index="model", columns="target", values="MAE_hour").round(2).to_string())
