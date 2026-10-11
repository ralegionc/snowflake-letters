"""Snowflake simulator: grow single crystals along a two-stage temperature path, render MASC-like triplets.

Physics (src/habit.py): laboratory s_char(T) per axis, Nelson-Baker deposition coefficients, a common
surface supersaturation set by diffusion supply, facet-based axis growth. Branching grows with
Lambda = s_env / s_surf (how far growth is diffusion-limited). Unknown physics and imaging details are
nuisance parameters drawn from broad priors, so the inference network has to marginalise over them.

Output per crystal: 3 x 96 x 96 uint8 views (same crop pipeline as the real data), log Dmax, and the
parameters theta.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from habit import alpha, s_char, s_water, surface_supersat, x0  # noqa: E402
from extract import crop  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PIX = 33.5e-6          # MASC native pixel size (m)
CAM_AZ = np.radians([0.0, 36.0, 72.0])
THETA_NAMES = ["T1", "T2", "u1", "u2", "frac", "logD", "m", "jit_c", "jit_a", "P", "lam_c", "c0", "gmax",
               "B", "aspect", "capped"]


def sample_prior(rng, n, logD_pool):
    T1 = rng.uniform(-32, -1, n)
    T2 = np.clip(T1 + rng.uniform(-10, 10, n), -32, -1)
    return {
        "T1": T1, "T2": T2,
        "u1": rng.uniform(0.15, 1.05, n),       # s_env as a fraction of water saturation
        "u2": rng.uniform(0.15, 1.05, n),
        "frac": rng.uniform(0.15, 0.85, n),     # share of final size grown in stage 1
        "logD": rng.choice(logD_pool, n),       # final max dimension (m), from the real size distribution
        "m": rng.uniform(4, 12, n),             # ledge-nucleation exponent
        "jit_c": np.exp(rng.normal(0, 0.15, n)),  # lab-fit uncertainty on s_char
        "jit_a": np.exp(rng.normal(0, 0.15, n)),
        "P": rng.uniform(500, 900, n),          # growth-level pressure (hPa)
        "lam_c": np.exp(rng.uniform(np.log(6), np.log(40), n)),  # branching onset
        "c0": rng.uniform(4e-6, 15e-6, n),      # seed half-size
        # cap on growth anisotropy: thin edges and needle tips become diffusion-limited, so real columns
        # stay stubbier than the kinetics alone predicts (real MASC columns: median view elongation ~2.9)
        "gmax": np.exp(rng.uniform(np.log(3), np.log(30), n)),
    }


def grow(p, steps=120):
    """Integrate axis growth in log-size steps. Returns final (a, c), state at stage switch, mean ln Lambda
    over the outer stage, and stage-2 growth ratio."""
    n = len(p["T1"])
    a, c = p["c0"].copy(), p["c0"].copy()
    D = np.exp(p["logD"])
    a1, c1 = np.full(n, np.nan), np.full(n, np.nan)
    lam_sum, lam_n = np.zeros(n), np.zeros(n)
    G2 = np.ones(n)
    for _ in range(steps):
        size = 2 * np.maximum(a, c)
        live = size < D
        if not live.any():
            break
        stage2 = size >= p["frac"] * D
        newly = stage2 & np.isnan(a1)
        a1[newly], c1[newly] = a[newly], c[newly]
        T = np.where(stage2, p["T2"], p["T1"])
        u = np.where(stage2, p["u2"], p["u1"])
        sc, sa = s_char(T)
        sc, sa = sc * p["jit_c"], sa * p["jit_a"]
        s_env = u * s_water(T)
        R = np.cbrt(a * a * c)  # volume-equivalent length scale
        w_c = a / (a + 2 * c)   # basal share of surface area (hexagonal prism, roughly)
        ss = surface_supersat(s_env, sc, sa, w_c, x0(T, p["P"]) / R, p["m"])
        vc, va = alpha(ss, sc, p["m"]) * ss, alpha(ss, sa, p["m"]) * ss
        G = vc / np.maximum(va, 1e-30)
        G = np.where(G >= 1, G / (1 + G / p["gmax"]) + 1 - 1 / (1 + 1 / p["gmax"]),
                     1 / ((1 / G) / (1 + (1 / G) / p["gmax"]) + 1 - 1 / (1 + 1 / p["gmax"])))
        vc = va * G
        # advance the faster axis by 5% per step, the other in proportion
        fast = np.maximum(va / np.maximum(a, 1e-9), vc / np.maximum(c, 1e-9))
        dt = 0.05 / np.maximum(fast, 1e-30)
        a = np.where(live, a + va * dt, a)
        c = np.where(live, c + vc * dt, c)
        lam = np.log(s_env / ss)
        lam_sum += np.where(live & stage2, lam, 0)
        lam_n += live & stage2
        G2 = np.where(live & stage2, vc / np.maximum(va, 1e-30), G2)
    scale = D / (2 * np.maximum(a, c))  # land exactly on the target size
    a, c = a * scale, c * scale
    a1 = np.where(np.isnan(a1), a, a1)
    c1 = np.where(np.isnan(c1), c, c1)
    lam_mean = np.where(lam_n > 0, lam_sum / np.maximum(lam_n, 1), 0)
    return a, c, a1, c1, lam_mean, G2


def plate_mask(B, n=128):
    """Basal-plane outline in units of the circumradius: hexagon -> sectored -> stellar -> fernlike."""
    x, y = np.meshgrid(np.linspace(-1.05, 1.05, n), np.linspace(-1.05, 1.05, n))
    ang = np.radians(np.arange(6) * 60)
    proj_face = np.max([x * np.cos(t + np.pi / 6) + y * np.sin(t + np.pi / 6) for t in ang], 0)
    hexagon = proj_face <= np.cos(np.pi / 6)
    if B < 0.35:
        return hexagon
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    d_vertex = np.abs(((th + np.pi / 6) % (np.pi / 3)) - np.pi / 6)  # angle to nearest vertex direction
    if B < 0.6:  # sectored plate: V notches at facet centres
        depth = 0.15 + 0.5 * (B - 0.35) / 0.25
        return hexagon & ~((d_vertex > np.radians(20)) & (r > 1 - depth) & (proj_face > np.cos(np.pi / 6) - depth))
    core = proj_face <= np.cos(np.pi / 6) * 0.3 * (1.6 - B)
    w = 0.14 * (1.35 - B)
    m = core.copy()
    for t in ang:
        u, v = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
        m |= (u >= 0) & (u <= 1) & (np.abs(v) <= w / 2 * (1 - 0.4 * u))
        if B > 0.75:  # side branches at +-60 degrees
            L = 0.5 * (B - 0.75) / 0.25 + 0.1
            for s in (0.3, 0.45, 0.6, 0.75):
                for sgn in (1, -1):
                    bt = sgn * np.pi / 3
                    uu, vv = u - s, v
                    bu, bv = uu * np.cos(bt) + vv * np.sin(bt), -uu * np.sin(bt) + vv * np.cos(bt)
                    m |= (bu >= 0) & (bu <= L * (1 - s)) & (np.abs(bv) <= w * 0.35)
    return m & (r <= 1.0)


def hex_points(rng, n, radius, z0, z1, mask=None):
    """Uniform points in a hexagonal prism (or a masked plate) of circumradius `radius`, z in [z0, z1]."""
    if mask is None:
        mask = plate_mask(0.0, 64)
    g = mask.shape[0]
    cells = np.flatnonzero(mask)
    k = rng.choice(cells, n)
    iy, ix = np.divmod(k, g)
    span = 2.1 / g
    px = (-1.05 + (ix + rng.random(n)) * span) * radius
    py = (-1.05 + (iy + rng.random(n)) * span) * radius
    pz = rng.uniform(z0, z1, n)
    area = mask.mean() * (2.1 * radius) ** 2
    return np.stack([px, py, pz], 1), area * (z1 - z0) / n


def build_crystal(rng, a, c, a1, c1, B, G2, npts=40000):
    """Point cloud (c-axis = z) and per-point volume."""
    capped = (c1 > 1.5 * a1) & (G2 < 0.3) & (a > 1.3 * a1)
    if capped:  # capped column: stage-1 column with stage-2 plates on both ends
        col, vc = hex_points(rng, npts // 2, a1, -c1, c1)
        th = max(c - c1, 2e-6)
        m = plate_mask(B)
        p1, vp = hex_points(rng, npts // 4, a, c1, c1 + th, m)
        p2, _ = hex_points(rng, npts // 4, a, -c1 - th, -c1, m)
        pts = np.concatenate([col, p1, p2])
        vol = np.concatenate([np.full(len(col), vc), np.full(len(p1) + len(p2), vp)])
        return pts, vol, True
    if c > a:  # column / needle; branching thins it (sheaths, needles)
        r = a * (1 - 0.4 * max(B - 0.6, 0))
        pts, v = hex_points(rng, npts, r, -c, c)
    else:
        pts, v = hex_points(rng, npts, a, -c, c, plate_mask(B))
    return pts, np.full(len(pts), v), False


def random_rotation(rng, is_plate):
    """Plates fall c-axis near vertical, columns c-axis near horizontal; MASC turbulence adds tumbling."""
    if rng.random() < 0.6:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    tilt = abs(rng.normal(0, np.radians(rng.uniform(10, 70))))
    if not is_plate:
        tilt = np.pi / 2 - tilt
    az, spin = rng.uniform(0, 2 * np.pi, 2)
    Rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    return Rz(az) @ Rx @ Rz(spin)


def _smooth_noise(rng, shape, scale):
    return gaussian_filter(rng.normal(size=shape), scale) * scale


def _warp(rng, img, amp):
    """Smooth random displacement: crystal imperfections, light rime, slight polycrystallinity."""
    if amp <= 0:
        return img
    from scipy.ndimage import map_coordinates
    n = img.shape[0]
    dy = _smooth_noise(rng, img.shape, n / 8)
    dx = _smooth_noise(rng, img.shape, n / 8)
    norm = amp * n / (np.abs(dx).max() + np.abs(dy).max() + 1e-9)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return map_coordinates(img, [yy + dy * norm, xx + dx * norm], order=1)


def render(rng, pts, vol, Dmax):
    """Three horizontal MASC views. Brightness rises with projected ice path length, with edge highlights,
    internal texture and specks (MASC sees scattered light, not absorption), then defocus and noise.
    Intensity ranges are set to match real MASCDB single-crystal crops (mean ~45, p95 ~110-200)."""
    n = int(np.ceil(Dmax / PIX)) + 24
    L0 = np.exp(rng.uniform(np.log(30e-6), np.log(600e-6)))
    bright = rng.uniform(0.15, 0.75)
    edge_w = rng.uniform(0, 1.5)
    tex = rng.uniform(0, 0.6)
    speck = rng.uniform(0, 0.03)
    warp = rng.uniform(0, 0.06)
    blur = rng.uniform(0.3, 2.5)
    views = []
    for az in CAM_AZ:
        u = -pts[:, 0] * np.sin(az) + pts[:, 1] * np.cos(az)
        v = pts[:, 2]
        iu = np.clip((u / PIX + n / 2).astype(int), 0, n - 1)
        iv = np.clip((v / PIX + n / 2).astype(int), 0, n - 1)
        path = np.bincount(iv * n + iu, weights=vol, minlength=n * n).reshape(n, n) / PIX ** 2
        path = gaussian_filter(path, 0.7)
        body = 1 - np.exp(-path / L0)
        gy, gx = np.gradient(body)
        img = body + edge_w * np.hypot(gx, gy)
        img = img * (1 + tex * _smooth_noise(rng, img.shape, 2.0) * 3)
        img = img + (rng.random(img.shape) < speck) * (body > 0.05) * rng.uniform(0.5, 1.5)
        img = _warp(rng, np.clip(img, 0, None), warp)
        img = gaussian_filter(img, blur)
        img = bright * img / (np.percentile(img[img > 0.02], 95) + 1e-6 if (img > 0.02).any() else 1)
        img = img + rng.normal(0, rng.uniform(0, 0.02), img.shape) * (img > 0.02)
        img = np.clip(img * 255, 0, 255)
        img[img < 6] = 0
        views.append(crop(img.astype(np.uint8)))
    return np.stack(views)


def simulate(args):
    seed, n, logD_pool = args
    rng = np.random.default_rng(seed)
    p = sample_prior(rng, n, logD_pool)
    a, c, a1, c1, lam, G2 = grow(p)
    B = 1 / (1 + np.exp(-(lam - np.log(p["lam_c"])) / 0.35))
    imgs = np.zeros((n, 3, 96, 96), np.uint8)
    capped = np.zeros(n)
    for i in range(n):
        pts, vol, cap = build_crystal(rng, a[i], c[i], a1[i], c1[i], B[i], G2[i])
        R = random_rotation(rng, is_plate=(c[i] <= a[i]) or cap)
        pts = pts @ R.T
        imgs[i] = render(rng, pts, vol, np.exp(p["logD"][i]))
        capped[i] = cap
    p.update(B=B, aspect=np.log(c / a), capped=capped)
    theta = np.stack([p[k] for k in THETA_NAMES], 1).astype(np.float32)
    return imgs, theta


if __name__ == "__main__":
    import pandas as pd
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    out = sys.argv[3] if len(sys.argv) > 3 else "sims"
    t = pd.read_parquet(ROOT / "data/table.parquet")
    single = t.snowflake_class_name.isin(["planar_crystal", "columnar_crystal", "columnar_planar_combination"])
    logD_pool = np.log(t.flake_Dmax[single].to_numpy())
    batch = 250
    jobs = [(1000 + k, batch, logD_pool) for k in range(n_total // batch)]
    imgs, thetas = [], []
    with ProcessPoolExecutor(workers) as ex:
        for k, (im, th) in enumerate(ex.map(simulate, jobs)):
            imgs.append(im)
            thetas.append(th)
            if k % 20 == 0:
                print(f"{(k + 1) * batch} sims", flush=True)
    np.save(ROOT / f"data/{out}_images.npy", np.concatenate(imgs))
    np.save(ROOT / f"data/{out}_theta.npy", np.concatenate(thetas))
    print("done", sum(len(i) for i in imgs))
