"""
Plot lithostatic stress-gradient diagnostics for crustal seismicity.

Produces the following figures (saved as both PNG and PDF):

1. Stress-gradient magnitude vs depth, coloured by crustal thickness
2. Stress-gradient magnitude vs depth, coloured by surface topography
3. Stress-gradient magnitude vs depth, coloured by local density
4. Topography vs crustal thickness — earthquake locations
5. Topography vs crustal thickness — full model (hexbin)
6. Earthquake gradient percentile vs depth

Dependencies
------------
numpy, pandas, scipy, matplotlib, cmcrameri

Input files (expected in the same directory as this script)
-----------------------------------------------------------
model_prep.csv     – columns: x, y, z, rho
utm_sediments.csv  – columns: x, y, z  (whitespace-separated)
utm_lc_new.csv     – columns: x, y, z  (whitespace-separated)
seismicity.csv     – columns: Lon, Lat, Dep, Ml, Type

Usage
-----
    python plot_stress_gradient_figures.py
"""

import os

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_CSV = os.path.join(os.path.dirname(__file__), "model_prep.csv")
SEIS_CSV_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "seismicity.csv"),
    os.path.join(os.path.dirname(__file__), "Seismicity.csv"),
]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")

G = 9.81                           # gravitational acceleration (m/s²)
SLICE_STEP_KM = 1.0                # depth spacing for interpolation (km)
HORIZONTAL_SMOOTH_SIGMA_KM = 40.0  # Gaussian smoothing sigma (km)
EQ_MAX_DEPTH_KM = 80.0             # maximum plot depth for earthquakes (km)
EQ_DEPTH_TOL_KM = 0.5              # ± window for slice-matching events

# Spatial bounds (metres, UTM)
X_MIN, X_MAX = -480_000, 790_000
Y_MIN, Y_MAX = 5_300_000, 7_900_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_existing_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No file found among: {paths}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_density_grid(model_csv):
    """Load the 3-D density model from *model_csv* and return regular-grid
    arrays ``(x_vals, y_vals, z_vals, rho_3d)``."""
    df = pd.read_csv(model_csv).dropna(subset=["rho", "x", "y", "z"])
    df = df[
        (df["x"] >= X_MIN) & (df["x"] <= X_MAX)
        & (df["y"] >= Y_MIN) & (df["y"] <= Y_MAX)
    ]

    x_vals = np.sort(df["x"].unique())
    y_vals = np.sort(df["y"].unique())
    z_vals = np.sort(df["z"].unique())
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

    x_to_ix = {v: i for i, v in enumerate(x_vals)}
    y_to_iy = {v: i for i, v in enumerate(y_vals)}
    z_to_iz = {v: i for i, v in enumerate(z_vals)}

    rho_3d = np.full((nz, ny, nx), np.nan, dtype=float)
    rho_3d[
        df["z"].map(z_to_iz).to_numpy(dtype=int),
        df["y"].map(y_to_iy).to_numpy(dtype=int),
        df["x"].map(x_to_ix).to_numpy(dtype=int),
    ] = df["rho"].to_numpy(dtype=float)

    # Fill NaN columns
    global_mean = np.nanmean(rho_3d)
    for j in range(ny):
        for i in range(nx):
            col = rho_3d[:, j, i]
            if np.isnan(col).all():
                rho_3d[:, j, i] = global_mean
            elif np.isnan(col).any():
                idx = np.arange(nz)
                good = np.isfinite(col)
                rho_3d[:, j, i] = np.interp(idx, idx[good], col[good])

    return x_vals, y_vals, z_vals, rho_3d


def compute_crustal_thickness_and_topography(x_vals, y_vals, z_vals, rho_3d):
    """Return ``(thickness_km, mean_crust_rho, topography_km)`` grids of
    shape ``(ny, nx)``."""
    _dir = os.path.dirname(__file__)
    sed = pd.read_csv(os.path.join(_dir, "utm_sediments.csv"), sep=r"\s+")
    interp_sed = NearestNDInterpolator(sed[["x", "y"]].values, sed["z"].values)

    lc = pd.read_csv(os.path.join(_dir, "utm_lc_new.csv"), sep=r"\s+")
    interp_lc = NearestNDInterpolator(lc[["x", "y"]].values, lc["z"].values)

    # Identify oceanic columns (no crustal densities above -80 km)
    model_df = pd.read_csv(MODEL_CSV, usecols=["x", "y", "z", "rho"])
    top_rho = model_df[model_df["z"] > -80000].groupby(["x", "y"])["rho"].max()
    oceanic_idx = set(top_rho[top_rho == 0].index)

    ny, nx = len(y_vals), len(x_vals)
    thickness = np.full((ny, nx), np.nan)
    mean_crust_rho = np.full((ny, nx), np.nan)
    topography_km = np.full((ny, nx), np.nan)

    xx, yy = np.meshgrid(x_vals, y_vals)
    sed_z = interp_sed(xx, yy)
    lc_z = interp_lc(xx, yy)

    for j in range(ny):
        for i in range(nx):
            x, y = x_vals[i], y_vals[j]
            s_z, l_z = sed_z[j, i], lc_z[j, i]
            m_z = (l_z - 7000.0) if (x, y) in oceanic_idx else l_z

            thickness[j, i] = (s_z - m_z) / 1000.0
            topography_km[j, i] = s_z / 1000.0

            in_crust = (z_vals >= m_z) & (z_vals <= s_z)
            if np.any(in_crust):
                mean_crust_rho[j, i] = np.nanmean(rho_3d[in_crust, j, i])

    return thickness, mean_crust_rho, topography_km


def load_crustal_seismicity(seis_csv, x_vals, y_vals):
    """Load crustal earthquakes (Ml > 2) and return a DataFrame with
    columns ``x, y, dep_km, magnitude``."""
    eq = pd.read_csv(seis_csv)

    # Filter to crustal events
    if "Type" not in eq.columns:
        raise ValueError("Seismicity CSV must contain 'Type'")
    eq = eq.loc[eq["Type"].astype(str).str.strip().str.lower() == "crustal"].copy()

    # Coordinate handling: try native x/y first, then reproject Lon/Lat
    if {"x", "y"}.issubset(eq.columns):
        eq["x"] = pd.to_numeric(eq["x"], errors="coerce")
        eq["y"] = pd.to_numeric(eq["y"], errors="coerce")
    elif {"Lon", "Lat"}.issubset(eq.columns):
        eq["x"] = pd.to_numeric(eq["Lon"], errors="coerce")
        eq["y"] = pd.to_numeric(eq["Lat"], errors="coerce")
    else:
        raise ValueError("Seismicity CSV must contain [x,y] or [Lon,Lat]")

    eq["dep_km"] = pd.to_numeric(eq["Dep"], errors="coerce").abs()
    eq["magnitude"] = pd.to_numeric(eq.get("Ml", np.nan), errors="coerce")
    eq = eq.dropna(subset=["x", "y", "dep_km", "magnitude"])
    eq = eq[eq["magnitude"] > 2.0].copy()

    # Clip to model domain
    eq = eq[
        (eq["x"] >= np.min(x_vals)) & (eq["x"] <= np.max(x_vals))
        & (eq["y"] >= np.min(y_vals)) & (eq["y"] <= np.max(y_vals))
    ].reset_index(drop=True)

    # Fallback: reproject Lon/Lat if no events landed inside the domain
    if eq.empty and {"Lon", "Lat"}.issubset(pd.read_csv(seis_csv).columns):
        from pyproj import Transformer
        raw = pd.read_csv(seis_csv)
        raw = raw.loc[raw["Type"].astype(str).str.strip().str.lower() == "crustal"]
        raw = raw.dropna(subset=["Lon", "Lat", "Dep"]).copy()
        lon = pd.to_numeric(raw["Lon"], errors="coerce").to_numpy(float)
        lat = pd.to_numeric(raw["Lat"], errors="coerce").to_numpy(float)
        good = np.isfinite(lon) & np.isfinite(lat)

        # Find best-fitting UTM zone
        best_epsg, best_n = None, -1
        for zone in range(17, 22):
            epsg = 32700 + zone
            tf = Transformer.from_crs("epsg:4326", f"epsg:{epsg}", always_xy=True)
            ex, ey = tf.transform(lon[good], lat[good])
            n = int(np.sum(
                (ex >= np.min(x_vals) - 25000) & (ex <= np.max(x_vals) + 25000)
                & (ey >= np.min(y_vals) - 25000) & (ey <= np.max(y_vals) + 25000)
            ))
            if n > best_n:
                best_n, best_epsg = n, epsg

        tf = Transformer.from_crs("epsg:4326", f"epsg:{best_epsg}", always_xy=True)
        ex, ey = tf.transform(lon[good], lat[good])
        raw = raw.loc[good].copy()
        raw["x"], raw["y"] = ex, ey
        raw["dep_km"] = pd.to_numeric(raw["Dep"], errors="coerce").abs()
        raw["magnitude"] = pd.to_numeric(raw.get("Ml", np.nan), errors="coerce")
        raw = raw.dropna(subset=["x", "y", "dep_km", "magnitude"])
        raw = raw[raw["magnitude"] > 2.0].copy()
        eq = raw[
            (raw["x"] >= np.min(x_vals)) & (raw["x"] <= np.max(x_vals))
            & (raw["y"] >= np.min(y_vals)) & (raw["y"] <= np.max(y_vals))
        ].reset_index(drop=True)

    return eq


# ---------------------------------------------------------------------------
# Stress and gradient computation
# ---------------------------------------------------------------------------

def _build_depth_axis(z_vals):
    return -z_vals if np.nanmedian(z_vals) < 0 else z_vals


def _depth_km_to_z(depth_km, z_vals):
    depth_m = np.asarray(depth_km, dtype=float) * 1000.0
    z = -depth_m if np.nanmedian(z_vals) < 0 else depth_m
    return np.clip(z, np.min(z_vals), np.max(z_vals))


def compute_lithostatic_stress(rho_3d, z_vals):
    """Integrate rho·g downward → sigma_zz (Pa)."""
    depth = _build_depth_axis(z_vals)
    order = np.argsort(depth)
    sigma_sorted = G * cumulative_trapezoid(
        rho_3d[order, :, :], depth[order], axis=0, initial=0.0,
    )
    sigma = np.empty_like(sigma_sorted)
    sigma[order, :, :] = sigma_sorted
    return sigma


def interpolate_to_depths(field_3d, z_vals, target_depths_km):
    """Interpolate a (nz, ny, nx) field to specified depth levels."""
    z_targets = _depth_km_to_z(target_depths_km, z_vals)
    nz, ny, nx = field_3d.shape
    out = np.empty((len(z_targets), ny, nx), dtype=float)
    for j in range(ny):
        for i in range(nx):
            out[:, j, i] = np.interp(z_targets, z_vals, field_3d[:, j, i])
    return out


def compute_horizontal_gradients(stress_interp, x_vals, y_vals,
                                 smooth_sigma_km=HORIZONTAL_SMOOTH_SIGMA_KM):
    """Gaussian-smooth each depth slice, then compute |∇_h σ_zz| (Pa/m)."""
    ndepth = stress_interp.shape[0]
    grad_mag = np.empty_like(stress_interp)
    dx = float(np.abs(np.nanmedian(np.diff(x_vals))))
    dy = float(np.abs(np.nanmedian(np.diff(y_vals))))
    sx = (smooth_sigma_km * 1000.0) / dx
    sy = (smooth_sigma_km * 1000.0) / dy

    for k in range(ndepth):
        smoothed = gaussian_filter(stress_interp[k], sigma=(sy, sx), mode="nearest")
        gy, gx = np.gradient(smoothed, y_vals, x_vals)
        grad_mag[k] = np.sqrt(gx**2 + gy**2)
    return grad_mag


def assign_gradient_at_events(eq_df, grad_interp3d, depth_min_km, depth_max_km):
    """Interpolate the gradient field at each earthquake location and
    add ``stress_grad_at_eq_pam`` and ``dep_km_for_interp`` columns."""
    eq_df = eq_df.copy()
    dep = np.clip(eq_df["dep_km"].to_numpy(float), depth_min_km, depth_max_km)
    pts = np.column_stack((dep, eq_df["y"], eq_df["x"]))
    eq_df["stress_grad_at_eq_pam"] = grad_interp3d(pts)
    eq_df["dep_km_for_interp"] = dep
    return eq_df


def compute_eq_gradient_percentiles(eq_df, grad_3d, target_depths_km):
    """For each earthquake, compute the percentile of its gradient value
    within the depth-matched background distribution."""
    dep = eq_df["dep_km_for_interp"].to_numpy(float)
    eq_grad = eq_df["stress_grad_at_eq_pam"].to_numpy(float)

    percentiles = np.full(len(eq_df), np.nan)
    for i in range(len(eq_df)):
        if not (np.isfinite(dep[i]) and np.isfinite(eq_grad[i])):
            continue
        iz = int(np.argmin(np.abs(target_depths_km - dep[i])))
        bg = grad_3d[iz].ravel()
        bg = bg[np.isfinite(bg)]
        if bg.size > 0:
            percentiles[i] = 100.0 * np.mean(bg <= eq_grad[i])

    eq_df = eq_df.copy()
    eq_df["eq_grad_percentile"] = percentiles
    return eq_df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _magnitude_sizes(mag_vals, s_min=10.0, s_max=50.0):
    """Map magnitudes to scatter point sizes."""
    finite = np.isfinite(mag_vals)
    if not np.any(finite):
        return np.full_like(mag_vals, 18.0)
    lo, hi = np.nanmin(mag_vals[finite]), np.nanmax(mag_vals[finite])
    if hi <= lo:
        return np.full_like(mag_vals, (s_min + s_max) / 2)
    norm = np.clip((mag_vals - lo) / (hi - lo), 0.0, 1.0)
    return s_min + (s_max - s_min) * norm


def _save(fig, path):
    """Save as PNG and PDF side by side."""
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_gradient_vs_depth(eq_df, output_path,
                           color_col, color_label, cmap,
                           color_vmin=None, color_vmax=None):
    """Scatter: horizontal stress-gradient vs depth, coloured by
    *color_col*."""
    df = eq_df[eq_df["dep_km"] <= EQ_MAX_DEPTH_KM].copy()
    sizes = _magnitude_sizes(df["magnitude"].to_numpy(float))
    order = np.argsort(np.where(np.isfinite(df["magnitude"]), df["magnitude"], -np.inf))

    c_vals = df[color_col].to_numpy(float)
    if color_vmin is None:
        color_vmin = float(np.nanpercentile(c_vals[np.isfinite(c_vals)], 2))
    if color_vmax is None:
        color_vmax = float(np.nanpercentile(c_vals[np.isfinite(c_vals)], 98))

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["stress_grad_at_eq_pam"].to_numpy()[order],
        df["dep_km"].to_numpy()[order],
        c=c_vals[order],
        cmap=cmap,
        vmin=color_vmin, vmax=color_vmax,
        s=sizes[order],
        alpha=0.55,
        edgecolors="black", linewidths=0.15,
    )
    fig.colorbar(sc, ax=ax, label=color_label)
    ax.set_xlabel("Horizontal stress-gradient magnitude (Pa/m)")
    ax.set_xlim(left=0)
    ax.set_ylabel("Depth (km)")
    ax.set_ylim(0, EQ_MAX_DEPTH_KM)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, output_path)


def plot_topography_vs_thickness_eq(eq_df, output_path):
    """Scatter: topography vs crustal thickness at earthquake locations,
    coloured by depth."""
    df = eq_df.dropna(subset=["topography_km", "crustal_thickness_km"]).copy()
    if len(df) == 0:
        return
    sizes = _magnitude_sizes(df["magnitude"].to_numpy(float))

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["topography_km"], df["crustal_thickness_km"],
        c=df["dep_km"], cmap=cmc.batlow_r,
        s=sizes, alpha=0.7,
        edgecolors="black", linewidths=0.15,
    )
    fig.colorbar(sc, ax=ax, label="Earthquake depth (km)")
    ax.set_xlabel("Topography / bathymetry (km)")
    ax.set_ylabel("Crustal thickness (km)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


def plot_topography_vs_thickness_model(topo_grid, thickness_grid, output_path):
    """Hexbin: topography vs crustal thickness for the full model."""
    topo = topo_grid.ravel()
    thick = thickness_grid.ravel()
    valid = np.isfinite(topo) & np.isfinite(thick)
    if not np.any(valid):
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(
        topo[valid], thick[valid],
        gridsize=70, mincnt=1, bins="log",
        cmap=cmc.lajolla_r, linewidths=0.0,
    )
    fig.colorbar(hb, ax=ax, label="log₁₀(count)")
    ax.set_xlabel("Topography / bathymetry (km)")
    ax.set_ylabel("Crustal thickness (km)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save(fig, output_path)


def plot_percentile_vs_depth(eq_df, output_path):
    """Scatter: earthquake gradient-percentile vs depth, coloured by
    gradient magnitude."""
    df = eq_df[eq_df["dep_km"] <= EQ_MAX_DEPTH_KM].copy()
    valid = np.isfinite(df["dep_km"]) & np.isfinite(df["eq_grad_percentile"])
    d = df.loc[valid, "dep_km"].to_numpy(float)
    p = df.loc[valid, "eq_grad_percentile"].to_numpy(float)
    grad = df.loc[valid, "stress_grad_at_eq_pam"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        p, d, c=grad, cmap=cmc.acton,
        s=16, alpha=0.85,
        edgecolors="black", linewidths=0.2,
    )
    fig.colorbar(sc, ax=ax, label="Horizontal stress-gradient (Pa/m)")

    median_p = np.nanmedian(p) if p.size > 0 else np.nan
    ax.axvline(50, color="gray", ls="--", lw=1.2, label="Random expectation")
    if np.isfinite(median_p):
        ax.axvline(median_p, color="crimson", lw=1.5,
                   label=f"Median = {median_p:.1f}")

    ax.set_xlabel("Gradient percentile within depth-slice background")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, EQ_MAX_DEPTH_KM)
    ax.invert_yaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dir(OUTPUT_DIR)

    # Load data ------------------------------------------------------------
    print("Loading density model …")
    x_vals, y_vals, z_vals, rho_3d = load_density_grid(MODEL_CSV)

    print("Computing crustal thickness and topography …")
    thickness_grid, mean_rho_grid, topo_grid = \
        compute_crustal_thickness_and_topography(x_vals, y_vals, z_vals, rho_3d)

    print("Loading seismicity …")
    seis_csv = find_existing_file(SEIS_CSV_CANDIDATES)
    eq_df = load_crustal_seismicity(seis_csv, x_vals, y_vals)
    if eq_df.empty:
        raise ValueError("No crustal earthquakes found inside the model domain.")

    # Assign crustal properties at earthquake locations --------------------
    interp_thick = RegularGridInterpolator(
        (y_vals, x_vals), thickness_grid, bounds_error=False, fill_value=np.nan)
    interp_topo = RegularGridInterpolator(
        (y_vals, x_vals), topo_grid, bounds_error=False, fill_value=np.nan)
    rho_interp3d = RegularGridInterpolator(
        (z_vals, y_vals, x_vals), rho_3d, bounds_error=False, fill_value=np.nan)

    eq_df["crustal_thickness_km"] = interp_thick((eq_df["y"], eq_df["x"]))
    eq_df["topography_km"] = interp_topo((eq_df["y"], eq_df["x"]))
    z_model = _depth_km_to_z(eq_df["dep_km"].to_numpy(float), z_vals)
    eq_df["exact_density"] = rho_interp3d(
        np.column_stack((z_model, eq_df["y"], eq_df["x"])))

    # Compute stress and horizontal gradient -------------------------------
    print("Computing lithostatic stress …")
    sigma_3d = compute_lithostatic_stress(rho_3d, z_vals)

    max_depth_km = float(np.max(_build_depth_axis(z_vals)) / 1000.0)
    target_depths_km = np.arange(0.0, max_depth_km + 0.5 * SLICE_STEP_KM,
                                 SLICE_STEP_KM)

    print("Interpolating to target depths …")
    sigma_interp = interpolate_to_depths(sigma_3d, z_vals, target_depths_km)

    print("Computing horizontal gradients …")
    grad_interp = compute_horizontal_gradients(sigma_interp, x_vals, y_vals)

    grad_interp3d = RegularGridInterpolator(
        (target_depths_km, y_vals, x_vals), grad_interp,
        bounds_error=False, fill_value=np.nan)

    # Gradient and percentile at each earthquake ---------------------------
    eq_df = assign_gradient_at_events(
        eq_df, grad_interp3d,
        depth_min_km=float(np.min(target_depths_km)),
        depth_max_km=float(np.max(target_depths_km)),
    )
    eq_df = compute_eq_gradient_percentiles(
        eq_df, grad_interp, target_depths_km)

    # Figures --------------------------------------------------------------
    print("Plotting …")

    # 1–3. Stress-gradient vs depth, coloured by three properties
    plot_gradient_vs_depth(
        eq_df, os.path.join(OUTPUT_DIR, "gradient_vs_depth_thickness.png"),
        color_col="crustal_thickness_km",
        color_label="Crustal thickness (km)",
        cmap=cmc.devon,
    )
    plot_gradient_vs_depth(
        eq_df, os.path.join(OUTPUT_DIR, "gradient_vs_depth_topography.png"),
        color_col="topography_km",
        color_label="Topography / bathymetry (km)",
        cmap=cmc.devon,
    )
    plot_gradient_vs_depth(
        eq_df, os.path.join(OUTPUT_DIR, "gradient_vs_depth_density.png"),
        color_col="exact_density",
        color_label="Local density (kg/m³)",
        cmap=cmc.lipari_r,
        color_vmin=2500, color_vmax=3100,
    )

    # 4–5. Topography vs crustal thickness
    plot_topography_vs_thickness_eq(
        eq_df, os.path.join(OUTPUT_DIR, "topo_vs_thickness_seismicity.png"))
    plot_topography_vs_thickness_model(
        topo_grid, thickness_grid,
        os.path.join(OUTPUT_DIR, "topo_vs_thickness_model.png"))

    # 6. Gradient percentile vs depth
    plot_percentile_vs_depth(
        eq_df, os.path.join(OUTPUT_DIR, "gradient_percentile_vs_depth.png"))

    print(f"All figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
