"""
Compute lithostatic (vertical normal) stress and its horizontal gradient
from a 3-D density model.

Reads a regularly-gridded density model (CSV with columns x, y, z, rho),
integrates rho·g downward to obtain the vertical normal stress sigma_zz,
smooths each depth slice horizontally, and computes the magnitude of the
horizontal gradient of sigma_zz.  Results are exported as one CSV per
depth slice.

Dependencies
------------
numpy, pandas, scipy

Usage
-----
    python compute_lithostatic_stress_gradient.py

Input files (expected in the same directory as this script)
-----------------------------------------------------------
model_prep.csv   – columns: x (m), y (m), z (m, negative-down), rho (kg/m³)

Output
------
<OUTPUT_DIR>/csv_depth_slices/stress_gradient_depth_NNNkm.csv
    Columns: x_m, y_m, depth_km, stress_mpa, grad_pam
"""

import os

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_CSV = os.path.join(os.path.dirname(__file__), "../data_inmodel_prep.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "stress_gradient_output")

G = 9.81                        # gravitational acceleration (m/s²)
SLICE_STEP_KM = 1.0             # depth spacing for output slices (km)
HORIZONTAL_SMOOTH_SIGMA_KM = 40.0  # Gaussian smoothing sigma (km)

# Spatial bounds applied when loading the model (metres, UTM)
X_MIN, X_MAX = -480_000, 790_000
Y_MIN, Y_MAX = 5_300_000, 7_900_000


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create *path* and any parents if they do not exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_density_grid(model_csv: str):
    """
    Load a 3-D density model from CSV and return regular grid arrays.

    Parameters
    ----------
    model_csv : str
        Path to CSV with columns ``x``, ``y``, ``z``, ``rho``.

    Returns
    -------
    x_vals : 1-D array
        Sorted unique x coordinates (m).
    y_vals : 1-D array
        Sorted unique y coordinates (m).
    z_vals : 1-D array
        Sorted unique z coordinates (m, negative-down convention).
    rho_3d : 3-D array, shape (nz, ny, nx)
        Density field (kg/m³).  NaN columns are filled with the global
        mean; partially-NaN columns are linearly interpolated.
    """
    df = pd.read_csv(model_csv).dropna(subset=["rho", "x", "y", "z"])

    df = df[
        (df["x"] >= X_MIN)
        & (df["x"] <= X_MAX)
        & (df["y"] >= Y_MIN)
        & (df["y"] <= Y_MAX)
    ]

    required = {"rho", "x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Model CSV must contain {required}, found {list(df.columns)}"
        )

    x_vals = np.sort(df["x"].unique())
    y_vals = np.sort(df["y"].unique())
    z_vals = np.sort(df["z"].unique())

    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
    if min(nx, ny, nz) < 2:
        raise ValueError("Model grid must have at least 2 samples in x, y, z")

    x_to_ix = {v: i for i, v in enumerate(x_vals)}
    y_to_iy = {v: i for i, v in enumerate(y_vals)}
    z_to_iz = {v: i for i, v in enumerate(z_vals)}

    rho_3d = np.full((nz, ny, nx), np.nan, dtype=float)
    ix = df["x"].map(x_to_ix).to_numpy(dtype=int)
    iy = df["y"].map(y_to_iy).to_numpy(dtype=int)
    iz = df["z"].map(z_to_iz).to_numpy(dtype=int)
    rho_3d[iz, iy, ix] = df["rho"].to_numpy(dtype=float)

    # Fill missing values -------------------------------------------------
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


# ---------------------------------------------------------------------------
# Depth-axis helpers
# ---------------------------------------------------------------------------

def build_depth_axis(z_vals):
    """Convert z (negative-down) to positive-downward depth (m)."""
    if np.nanmedian(z_vals) < 0:
        return -z_vals
    return z_vals


def model_depth_km_to_z(depth_km, z_vals):
    """Map depth in km to model z coordinate (m)."""
    depth_m = np.asarray(depth_km, dtype=float) * 1000.0
    if np.nanmedian(z_vals) < 0:
        z = -depth_m
    else:
        z = depth_m
    return np.clip(z, np.min(z_vals), np.max(z_vals))


# ---------------------------------------------------------------------------
# Stress computation
# ---------------------------------------------------------------------------

def compute_lithostatic_stress(rho_3d, z_vals):
    """
    Integrate rho·g from the surface downward to obtain sigma_zz.

    Parameters
    ----------
    rho_3d : 3-D array (nz, ny, nx)
        Density field in kg/m³.
    z_vals : 1-D array (nz,)
        Model z coordinates (m).

    Returns
    -------
    sigma : 3-D array (nz, ny, nx)
        Lithostatic (vertical normal) stress in Pa.
    """
    depth = build_depth_axis(z_vals)
    order = np.argsort(depth)
    depth_sorted = depth[order]
    rho_sorted = rho_3d[order, :, :]

    sigma_sorted = G * cumulative_trapezoid(
        rho_sorted, depth_sorted, axis=0, initial=0.0
    )
    sigma = np.empty_like(sigma_sorted)
    sigma[order, :, :] = sigma_sorted
    return sigma


# ---------------------------------------------------------------------------
# Interpolation to target depth slices
# ---------------------------------------------------------------------------

def interpolate_3d_to_depths(field_3d, z_vals, target_depths_km):
    """
    Interpolate a 3-D field at specified depth levels.

    Parameters
    ----------
    field_3d : 3-D array (nz, ny, nx)
    z_vals : 1-D array (nz,)
    target_depths_km : 1-D array
        Desired depths in km.

    Returns
    -------
    out : 3-D array (n_depths, ny, nx)
    """
    z_targets = model_depth_km_to_z(target_depths_km, z_vals)
    nz, ny, nx = field_3d.shape
    out = np.empty((len(z_targets), ny, nx), dtype=float)
    for j in range(ny):
        for i in range(nx):
            out[:, j, i] = np.interp(z_targets, z_vals, field_3d[:, j, i])
    return out


# ---------------------------------------------------------------------------
# Horizontal gradient
# ---------------------------------------------------------------------------

def compute_horizontal_gradients(
    stress_3d_interp, x_vals, y_vals,
    smooth_sigma_km=HORIZONTAL_SMOOTH_SIGMA_KM,
):
    """
    Compute the magnitude of the horizontal gradient of sigma_zz.

    Each depth slice is first smoothed with a 2-D Gaussian kernel
    (sigma = *smooth_sigma_km*) before finite-difference gradients
    are taken in x and y.

    Parameters
    ----------
    stress_3d_interp : 3-D array (n_depths, ny, nx)
        Stress field at the target depth levels (Pa).
    x_vals, y_vals : 1-D arrays
        Horizontal coordinates (m).
    smooth_sigma_km : float
        Gaussian smoothing length scale in km.

    Returns
    -------
    grad_mag : 3-D array (n_depths, ny, nx)
        |∇_h σ_zz| in Pa/m.
    """
    ndepth = stress_3d_interp.shape[0]
    grad_mag = np.empty_like(stress_3d_interp)
    dx = float(np.abs(np.nanmedian(np.diff(x_vals))))
    dy = float(np.abs(np.nanmedian(np.diff(y_vals))))
    sigma_x_px = (smooth_sigma_km * 1000.0) / dx
    sigma_y_px = (smooth_sigma_km * 1000.0) / dy

    for k in range(ndepth):
        smoothed = gaussian_filter(
            stress_3d_interp[k, :, :],
            sigma=(sigma_y_px, sigma_x_px),
            mode="nearest",
        )
        gy, gx = np.gradient(smoothed, y_vals, x_vals)
        grad_mag[k, :, :] = np.sqrt(gx**2 + gy**2)
    return grad_mag


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_stress_gradient_csv(x_vals, y_vals, target_depths_km,
                               stress_3d, grad_3d, output_dir):
    """
    Write a single CSV containing stress and gradient values for all depths.

    Parameters
    ----------
    x_vals, y_vals : 1-D arrays
        Horizontal coordinates (m).
    target_depths_km : 1-D array
    stress_3d : 3-D array (n_depths, ny, nx) – Pa
    grad_3d : 3-D array (n_depths, ny, nx) – Pa/m
    output_dir : str

    Returns
    -------
    csv_path : str
        Path to the output CSV.
    """
    ensure_dir(output_dir)

    xx, yy = np.meshgrid(x_vals, y_vals)
    n_xy = xx.size
    frames = []
    for i, dkm in enumerate(target_depths_km):
        frames.append(pd.DataFrame({
            "x_m": xx.ravel(),
            "y_m": yy.ravel(),
            "depth_km": np.full(n_xy, dkm, dtype=float),
            "stress_mpa": (stress_3d[i, :, :] / 1e6).ravel(),
            "grad_pam": grad_3d[i, :, :].ravel(),
        }))

    out_df = pd.concat(frames, ignore_index=True)
    csv_path = os.path.join(output_dir, "stress_gradient.csv")
    out_df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dir(OUTPUT_DIR)

    # 1. Load density model ------------------------------------------------
    print("Loading density model …")
    x_vals, y_vals, z_vals, rho_3d = load_density_grid(MODEL_CSV)
    print(f"  Grid dimensions: nx={len(x_vals)}, ny={len(y_vals)}, nz={len(z_vals)}")

    # 2. Compute lithostatic stress ----------------------------------------
    print("Computing lithostatic (vertical normal) stress …")
    sigma_3d = compute_lithostatic_stress(rho_3d, z_vals)

    # 3. Build target depth axis -------------------------------------------
    max_depth_km = float(np.max(build_depth_axis(z_vals)) / 1000.0)
    target_depths_km = np.arange(
        0.0, max_depth_km + 0.5 * SLICE_STEP_KM, SLICE_STEP_KM
    )
    print(f"  Depth range: 0 – {max_depth_km:.0f} km  "
          f"({len(target_depths_km)} slices at {SLICE_STEP_KM} km spacing)")

    # 4. Interpolate stress to target depths -------------------------------
    print("Interpolating stress to target depths …")
    sigma_interp = interpolate_3d_to_depths(sigma_3d, z_vals, target_depths_km)

    # 5. Compute horizontal gradient magnitude -----------------------------
    print(f"Computing horizontal gradients "
          f"(smoothing σ = {HORIZONTAL_SMOOTH_SIGMA_KM} km) …")
    grad_interp = compute_horizontal_gradients(
        sigma_interp, x_vals, y_vals,
        smooth_sigma_km=HORIZONTAL_SMOOTH_SIGMA_KM,
    )

    # 6. Export single CSV -------------------------------------------------
    print("Exporting stress & gradient CSV …")
    csv_path = export_stress_gradient_csv(
        x_vals, y_vals, target_depths_km,
        sigma_interp, grad_interp, OUTPUT_DIR,
    )
    print(f"  Written to {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
