"""Parse IGRA2 radiosonde files into temperature/humidity profiles on a fixed height grid.

Output: parquet with one row per sounding, columns T_<z> and RH_<z> for z in metres above sea level.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Z_GRID = np.arange(0, 8001, 100)  # m asl


def _num(s, scale=1.0):
    v = int(s)
    return np.nan if v in (-9999, -8888) else v / scale


def rh_from_dpd(t, dpd):
    """RH over water from temperature and dewpoint depression (Magnus)."""
    td = t - dpd
    es = lambda x: 6.112 * np.exp(17.67 * x / (x + 243.5))
    return 100 * es(td) / es(t)


def parse(path, year_min=2014):
    rows = []
    with open(path) as f:
        hdr, levels = None, []
        for line in f:
            if line[0] == "#":
                if hdr is not None:
                    rows.append((hdr, levels))
                year = int(line[13:17])
                hdr = None if year < year_min else (year, int(line[18:20]), int(line[21:23]), int(line[24:26]))
                levels = []
            elif hdr is not None:
                z, t = _num(line[16:21]), _num(line[22:27], 10)
                rh, dpd = _num(line[28:33], 10), _num(line[34:39], 10)
                p = _num(line[9:15], 100)
                if np.isnan(rh) and not np.isnan(dpd) and not np.isnan(t):
                    rh = rh_from_dpd(t, dpd)
                levels.append((p, z, t, rh))
        if hdr is not None:
            rows.append((hdr, levels))

    out = []
    for (y, m, d, h), lev in rows:
        if h == 99:
            continue
        a = np.array(lev, dtype=float)
        if len(a) < 5:
            continue
        ok = ~np.isnan(a[:, 1]) & ~np.isnan(a[:, 2])
        if ok.sum() < 5:
            continue
        z, t, rh = a[ok, 1], a[ok, 2], a[ok, 3]
        o = np.argsort(z)
        z, t, rh = z[o], t[o], rh[o]
        rec = {"time": pd.Timestamp(y, m, d, h), "z_sfc": z[0], "z_top": z[-1]}
        tg = np.interp(Z_GRID, z, t, left=np.nan, right=np.nan)
        okr = ~np.isnan(rh)
        rg = np.interp(Z_GRID, z[okr], rh[okr], left=np.nan, right=np.nan) if okr.sum() >= 3 else np.full(len(Z_GRID), np.nan)
        for zz, tv, rv in zip(Z_GRID, tg, rg):
            rec[f"T_{zz}"] = tv
            rec[f"RH_{zz}"] = rv
        out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    df = parse(src)
    df.to_parquet(dst)
    print(src.name, len(df), df.time.min(), df.time.max())
