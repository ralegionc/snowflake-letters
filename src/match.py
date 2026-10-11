"""Match MASCDB triplets to radiosonde profiles and build the modelling table.

Targets are temperature and RH at fixed heights above the MASC site, interpolated in time
between the two soundings that bracket each snowflake (max bracket 26 h).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"

CAMPAIGN_STATION = {
    "APRES3-2016": "AYM00089642",
    "APRES3-2017": "AYM00089642",
    "PLATO-2019": "AYM00089571",
    "ICEGENESIS-2021": "SZM00006610",
    "Jura-2019": "SZM00006610",
    "Jura-2023": "SZM00006610",
    "Remoray-2022": "SZM00006610",
    "Valais-2016": "SZM00006610",
}
DZ = [500, 1000, 1500, 2000, 3000]  # m above the MASC site
MAX_GAP_H = 26

# Per-camera descriptors averaged over the three views (shape/texture only, no labels).
CAM_FEATS = ["area", "perim", "Dmax", "eq_radius", "area_porous_r", "ell_fit_a_r", "ell_fit_ecc",
             "compactness", "roundness", "p_circ_out_r", "rectangularity", "rect_aspect_ratio",
             "solidity", "convexity", "hull_n_angles", "frac_dim_boxcounting", "nb_holes",
             "skel_N_ends", "skel_N_junc", "skel_perim_ratio", "skel_area_ratio", "sym_P6_max_ratio",
             "sym_mean", "sym_std_mean_ratio", "intensity_mean", "intensity_std", "contrast",
             "hist_entropy", "local_std", "lap_energy", "wavs", "complexity", "har_energy",
             "har_contrast", "har_corr", "har_hom", "D90_r"]


def profile_at(prof, times, site_alt):
    """Time-interpolate profiles to `times`, sampling heights site_alt + DZ."""
    prof = prof.sort_values("time").reset_index(drop=True)
    pt = prof.time.values.astype("datetime64[s]").astype(np.int64)
    tt = times.values.astype("datetime64[s]").astype(np.int64)
    j = np.searchsorted(pt, tt)  # first sounding after
    j0, j1 = np.clip(j - 1, 0, len(pt) - 1), np.clip(j, 0, len(pt) - 1)
    gap = (pt[j1] - pt[j0]) / 3600
    w = np.where(pt[j1] > pt[j0], (tt - pt[j0]) / np.maximum(pt[j1] - pt[j0], 1), 0.0)
    out = {"snd_gap_h": gap, "snd_dt_h": np.minimum(np.abs(tt - pt[j0]), np.abs(pt[j1] - tt)) / 3600}
    zgrid = np.array([int(c[2:]) for c in prof.columns if c.startswith("T_")])
    for var in ("T", "RH"):
        M = prof[[f"{var}_{z}" for z in zgrid]].to_numpy()
        for dz in DZ:
            col = np.array([np.interp(site_alt + dz, zgrid, row) for row in M])
            v = (1 - w) * col[j0] + w * col[j1]
            v[(gap > MAX_GAP_H) | (tt < pt[0]) | (tt > pt[-1])] = np.nan
            out[f"y_{var}{dz}"] = v
    return pd.DataFrame(out, index=times.index)


def rh_ice(rh_w, t):
    es_w = 6.112 * np.exp(17.67 * t / (t + 243.5))
    es_i = 6.112 * np.exp(22.46 * t / (272.62 + t))
    return rh_w * es_w / es_i


def cloud_targets(prof, site_alt, rhi_min=95.0, min_thick_m=300):
    """Per sounding, over the column above the site: the top of the highest ice-saturated layer at
    least `min_thick_m` thick, its temperature, the saturated fraction of the column, and whether any
    saturated level sits in the dendritic growth zone (-18..-12 C)."""
    zgrid = np.array([int(c[2:]) for c in prof.columns if c.startswith("T_")])
    T = prof[[f"T_{z}" for z in zgrid]].to_numpy()
    R = rh_ice(prof[[f"RH_{z}" for z in zgrid]].to_numpy(), T)
    k0 = np.searchsorted(zgrid, site_alt)
    T, R, z = T[:, k0:], R[:, k0:], zgrid[k0:]
    out = np.full((len(prof), 4), np.nan)
    n_thick = min_thick_m // 100 + 1
    for i in range(len(prof)):
        ok = ~np.isnan(R[i])
        if ok.sum() < 20:
            continue
        moist = np.where(ok, R[i] >= rhi_min, False)
        # top of the highest run of >= n_thick consecutive saturated levels
        top, run = None, 0
        for k, m in enumerate(moist):
            run = run + 1 if m else 0
            if run >= n_thick:
                top = k
        dgz = float(np.any(moist & (T[i] <= -12) & (T[i] >= -18)))
        frac = moist[ok].mean()
        if top is None:
            out[i] = [np.nan, np.nan, frac, dgz]
        else:
            out[i] = [z[top] - site_alt, T[i, top], frac, dgz]
    return pd.DataFrame(out, columns=["c_topZ", "c_topT", "c_satfrac", "c_dgz"], index=prof.index)


def add_episodes(t, max_gap_h=6):
    """Snow episode = run of snowy hours at one campaign with no gap longer than max_gap_h.
    Folds are grouped by episode so a storm never straddles train and test."""
    hr = t.datetime.dt.floor("h")
    h = pd.DataFrame({"c": t.campaign.values, "hr": hr.values}).drop_duplicates().sort_values(["c", "hr"])
    gap = h.groupby("c").hr.diff().dt.total_seconds() / 3600
    h["episode_id"] = (gap.isna() | (gap > max_gap_h)).cumsum().values
    key = dict(zip(zip(h.c, h.hr), h.episode_id))
    t["episode_id"] = [key[k] for k in zip(t.campaign.values, hr.values)]
    return t


def build():
    t = pd.read_parquet(D / "mascdb/MASCdb_triplet.parquet")
    t["zarr_idx"] = np.arange(len(t))
    t = t[t.campaign.isin(CAMPAIGN_STATION)]
    t = t[(t.melting_class_name == "dry") & (t.bs_precip_class_name == "precip")]
    parts = []
    for camp, g in t.groupby("campaign"):
        prof = pd.read_parquet(D / f"igra/{CAMPAIGN_STATION[camp]}.parquet")
        site_alt = g.altitude.iloc[0]
        g = g.join(profile_at(prof, g.datetime, site_alt))
        # Cloud targets are nonlinear in the profile, so take the nearest sounding within 6 h.
        prof = prof.sort_values("time").reset_index(drop=True)
        ct = cloud_targets(prof, site_alt).assign(snd_time=prof.time.astype("datetime64[ns]"))
        g["datetime"] = g.datetime.astype("datetime64[ns]")
        g = pd.merge_asof(g.sort_values("datetime"), ct, left_on="datetime", right_on="snd_time",
                          direction="nearest", tolerance=pd.Timedelta("6h"))
        g.index = g.zarr_idx.values
        parts.append(g)
    t = pd.concat(parts)
    t = t[t.y_T1500.notna()]

    cams = [pd.read_parquet(D / f"mascdb/MASCdb_cam{i}.parquet", columns=CAM_FEATS).iloc[t.zarr_idx.values]
            for i in range(3)]
    desc = sum(c.to_numpy(dtype=float) for c in cams) / 3
    for k, f in enumerate(CAM_FEATS):
        t[f"d_{f}"] = desc[:, k]

    t["hour_id"] = t.campaign + "_" + t.datetime.dt.floor("h").astype(str)
    t["event_id"] = t.campaign + "_" + t.datetime.dt.floor("D").astype(str)
    t = add_episodes(t)
    t.to_parquet(D / "table.parquet")
    return t


if __name__ == "__main__":
    t = build()
    print(len(t), "triplets,", t.hour_id.nunique(), "hours,", t.event_id.nunique(), "days")
    print(t.groupby("campaign").agg(n=("zarr_idx", "size"), hours=("hour_id", "nunique"),
                                    days=("event_id", "nunique"), T1500=("y_T1500", "mean"),
                                    envT=("env_T", "mean")).round(1).to_string())
    print(t.snowflake_class_name.value_counts().to_string())
    print(t[["env_T", "y_T500", "y_T1500", "y_T3000", "snd_dt_h"]].corr().round(2).to_string())
