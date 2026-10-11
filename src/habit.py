"""Ice habit physics from laboratory attachment kinetics.

Characteristic supersaturations s_char(T) for the c-axis (basal facet) and a-axis (prism facet) are the
polynomial fits of Harrington et al. (2019, J. Atmos. Sci., Table 1), in percent, with dT = T - 273.15 K.
Deposition coefficient (Nelson and Baker 1996): alpha = (s/s_char)^m * tanh((s_char/s)^m).
Axis growth follows the facet-based hypothesis dc/da = alpha_c / alpha_a.
"""
import numpy as np

C_FITS = [  # (T_lo, T_hi, coefficients a0..a6), T in C, lo < T <= hi
    (-22, -1, [1.1217, 0.0381, -0.08375, -0.01573, -0.001011, -2.915e-5, -3.182e-7]),
    (-30, -22, [753.63, 105.97, 5.553, 0.1281, 0.0011]),
    (-90, -30, [3.7955, 0.10614, 0.00753]),
]
A_FITS = [
    (-15, -1, [0.3457, -0.0093, 0.000308]),
    (-22, -15, [-5.2367, -1.3184, -0.1107, -0.00323]),
    (-30, -22, [-0.7106, -0.1478, 0.00423]),
    (-90, -30, [3.7955, 0.10614, 0.00753]),
]


def _poly(fits, T):
    T = np.clip(np.asarray(T, float), -60, -1)
    out = np.full(T.shape, np.nan)
    for lo, hi, co in fits:
        m = (T > lo) & (T <= hi)
        if lo == -30 and hi == -22:
            m = (T >= lo) & (T <= hi)
        out[m] = sum(c * T[m] ** n for n, c in enumerate(co))
    return out


def s_char(T):
    """(s_char_c, s_char_a) in percent."""
    return _poly(C_FITS, T), _poly(A_FITS, T)


def alpha(s, sc, m=10.0):
    x = (np.maximum(s, 1e-6) / sc) ** m
    return x * np.tanh(1.0 / x)


def s_water(T):
    """Ice supersaturation (%) at water saturation."""
    es_w = 6.112 * np.exp(17.67 * T / (T + 243.5))
    es_i = 6.112 * np.exp(22.46 * T / (272.62 + T))
    return 100 * (es_w / es_i - 1)


def x0(T, P_hpa):
    """Diffusion length X0 = D / sqrt(kT / 2 pi m) in metres (Libbrecht 2005); alpha_diff = X0 / R."""
    Tk = T + 273.15
    D = 2.11e-5 * (Tk / 273.15) ** 1.94 * (1013.25 / P_hpa)
    vbar = np.sqrt(1.380649e-23 * Tk / (2 * np.pi * 2.99e-26))
    return D / vbar


def surface_supersat(s_env, sc_c, sc_a, w_c, alpha_diff, m, iters=40):
    """Common surface supersaturation (%) where area-weighted attachment balances diffusion supply:
    (w_c a_c(s) + (1 - w_c) a_a(s)) s = alpha_diff (s_env - s). Vectorised bisection in log s."""
    lo = np.full(np.shape(s_env), -12.0)
    hi = np.log(np.asarray(s_env, float))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = np.exp(mid)
        f = (w_c * alpha(s, sc_c, m) + (1 - w_c) * alpha(s, sc_a, m)) * s - alpha_diff * (s_env - s)
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    return np.exp(0.5 * (lo + hi))
