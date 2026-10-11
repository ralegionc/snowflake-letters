"""Do physics-inferred growth temperatures agree with the clouds the radiosondes saw?

Primary test (fixed before looking at results): for each real single crystal, score = posterior mass of T2
on temperatures where the matched sounding is ice-supersaturated (RHi >= 100%) above the site. Compare the
mean score with matched soundings against soundings shuffled between snow episodes within a campaign.
No sounding information was used to train the inference network.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from match import CAMPAIGN_STATION, rh_ice  # noqa: E402
from npe import T_GRID  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v1"


def saturated_bins(prof, site_alt, rhi_min=100.0, tol=1.0):
    """Per sounding: boolean over T_GRID marking temperatures found in ice-supersaturated air above the site."""
    zgrid = np.array([int(c[2:]) for c in prof.columns if c.startswith("T_")])
    k0 = np.searchsorted(zgrid, site_alt)
    T = prof[[f"T_{z}" for z in zgrid[k0:]]].to_numpy()
    R = rh_ice(prof[[f"RH_{z}" for z in zgrid[k0:]]].to_numpy(), T)
    out = np.zeros((len(prof), len(T_GRID)), bool)
    for i in range(len(prof)):
        ts = T[i][(R[i] >= rhi_min) & ~np.isnan(T[i])]
        if len(ts):
            out[i] = (np.abs(T_GRID[:, None] - ts[None, :]) <= tol).any(1)
    return out


def load(subset):
    t = pd.read_parquet(D / "table.parquet").reset_index(drop=True)
    z = np.load(D / f"npe_real_{TAG}.npz")
    r = t.iloc[z["idx"]].copy()
    r["p_row"] = np.arange(len(r))
    if subset == "clean":
        r = r[(r.snowflake_class_prob > 0.8) & (r.riming_deg_level.fillna(1) < 0.3)]
    r = r[r.snd_time.notna()]
    sat = np.zeros((len(r), len(T_GRID)), bool)
    snd_key = np.empty(len(r), object)
    for camp, g in r.groupby("campaign"):
        prof = pd.read_parquet(D / f"igra/{CAMPAIGN_STATION[camp]}.parquet")
        prof["time"] = prof.time.astype("datetime64[ns]")
        prof = prof.drop_duplicates("time").set_index("time")
        bins = saturated_bins(prof, g.altitude.iloc[0])
        pos = prof.index.get_indexer(g.snd_time.astype("datetime64[ns]"))
        loc = r.index.get_indexer(g.index)
        sat[loc] = bins[pos]
        snd_key[loc] = camp + "|" + g.snd_time.astype(str).values
    return r, z["p2"][r.p_row.values], z["p1"][r.p_row.values], sat


def perm_test(r, P, sat, n_perm=5000, seed=0):
    """Episode-level test. Each episode gets its mean posterior P_e and the mean saturation profile of the
    soundings matched to its crystals, sat_e. Statistic: mean over episodes of sum_T P_e(T) sat_e(T).
    Null: permute sat_e between episodes within a campaign. Any posterior that ignores the image gives
    exactly the null value, so crystal counts or storm wetness cannot bias the test."""
    rng = np.random.default_rng(seed)
    ep = r.episode_id.values
    eps = np.unique(ep)
    Pe = np.stack([P[ep == e].mean(0) for e in eps])
    Se = np.stack([sat[ep == e].mean(0) for e in eps])
    camp = np.array([r.campaign.values[ep == e][0] for e in eps])
    obs = (Pe * Se).sum(1).mean()
    groups = [np.flatnonzero(camp == c) for c in np.unique(camp)]
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = np.arange(len(eps))
        for g in groups:
            perm[g] = rng.permutation(g)
        null[k] = (Pe * Se[perm]).sum(1).mean()
    return obs, null


if __name__ == "__main__":
    for subset in ["clean", "all"]:
        r, P2, P1, sat = load(subset)
        print(f"\n== {subset} single crystals: {len(r)} crystals, {r.hour_id.nunique()} hours,"
              f" {r.episode_id.nunique()} episodes")
        prior = np.full(len(T_GRID), 1 / len(T_GRID))
        print(f"  sounding saturated share of T grid: {sat.mean():.3f}; prior mass on saturated T: "
              f"{(prior * sat).sum(1).mean():.3f}")
        for name, P in [("T2 (outer)", P2), ("T1 (core)", P1)]:
            obs, null = perm_test(r, P, sat)
            p = (np.sum(null >= obs) + 1) / (len(null) + 1)
            z = (obs - null.mean()) / null.std()
            print(f"  {name}: matched {obs:.4f} vs shuffled {null.mean():.4f} (z = {z:+.2f}) "
                  f"[{np.percentile(null, 2.5):.4f}, {np.percentile(null, 97.5):.4f}]  p = {p:.4f}")
