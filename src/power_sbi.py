"""Power check for validate_sbi.py: what z would the test give if the simulator were exactly right?

Each real crystal is given a "true" T2 drawn from the ice-supersaturated temperatures of its own matched
sounding, then the posterior of a held-out simulation that grew near that T2 (same network). Running the
same episode-level permutation test on these synthetic posteriors gives the z expected under success.

usage: python src/power_sbi.py <tag> <sims prefix> [subset]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import validate_sbi as v  # noqa: E402
from npe import T_GRID  # noqa: E402
from sim import THETA_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"

if __name__ == "__main__":
    tag, sims = sys.argv[1], sys.argv[2]
    subset = sys.argv[3] if len(sys.argv) > 3 else "all"
    v.TAG = tag
    h = np.load(D / f"npe_heldout_{tag}.npz")
    T2s = np.load(D / f"{sims}_theta.npy")[h["idx"], THETA_NAMES.index("T2")]
    order = np.argsort(T2s)
    T2sorted = T2s[order]
    r, P2, P1, sat = v.load(subset)
    zs = []
    for rep in range(3):
        rng = np.random.default_rng(rep)
        Psyn = np.empty_like(P2)
        for k in range(len(r)):
            temps = T_GRID[sat[k]] if sat[k].any() else T_GRID
            Ttrue = rng.choice(temps) + rng.uniform(-0.25, 0.25)
            j = np.clip(np.searchsorted(T2sorted, Ttrue) + rng.integers(-20, 21), 0, len(order) - 1)
            Psyn[k] = h["p2"][order[j]]
        obs, null = v.perm_test(r, Psyn, sat, n_perm=1000, seed=rep)
        zs.append((obs - null.mean()) / null.std())
    print(f"{tag} {subset}: expected z if the simulator were right: {np.round(zs, 2)}")
