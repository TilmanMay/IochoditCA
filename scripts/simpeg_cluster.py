#!/usr/bin/env python
"""
SimPEG Gravity Inversion Script for Cluster Execution
Two-stage inversion:
1. Forward model initial tomography-based density model
2. Invert for residual density corrections
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import json

# SimPEG functionality
from simpeg.potential_fields import gravity
from simpeg.utils import plot2Ddata
from simpeg import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
)

# discretize functionality
from discretize import TreeMesh
from discretize.utils import active_from_xyz


def next_pow2(x):
    """Compute next power of 2"""
    return int(2 ** np.ceil(np.log2(x)))


def depth_dependent_density(cell_centers, background=1e-3):
    """Create density that varies with depth"""
    depths = -cell_centers[:, 2]  # depth (negative z)
    density_gradient = 2e-5  # density increase per meter depth
    densities = background + density_gradient * depths
    return densities


def moho_proximity_density(cell_centers, moho_surface, background=1e-3):
    """Create density based on distance to Moho surface"""
    from scipy.interpolate import LinearNDInterpolator

    # Interpolate Moho depth at cell locations
    moho_interp = LinearNDInterpolator(moho_surface[:, :2], moho_surface[:, 2])
    moho_depths_at_cells = moho_interp(cell_centers[:, :2])

    # Distance from cell to Moho (positive = above Moho, negative = below)
    distance_to_moho = cell_centers[:, 2] - moho_depths_at_cells

    # Create density contrast based on distance to Moho
    max_distance = 10000  # maximum distance to consider (m)
    normalized_distance = np.clip(np.abs(distance_to_moho) / max_distance, 0, 1)

    # Density decreases with distance from Moho
    density_variation = 0.3 * (1 - normalized_distance)  # max 0.3 g/cm³ variation
    densities = background + density_variation

    return densities


def load_tomography_density(
    cell_centers,
    tomo_file="data/tomography/saci_crust_utm20s.csv",
    utm_zone=None,
    hemisphere="s",
):
    """
    Load density from processed tomography data using efficient nearest neighbor search.
    For each mesh cell, find the closest tomography point (in 3D space) and use its density value.

    Parameters:
    -----------
    cell_centers : array-like
        Mesh cell center coordinates in UTM (X, Y, Z in meters)
    tomo_file : str
        Path to tomography CSV file
    utm_zone : int, optional
        UTM zone number for coordinate conversion. If None, assumes data is already in UTM.
        If provided, will convert lon/lat coordinates to UTM.
    hemisphere : str
        'n' for northern hemisphere, 's' for southern hemisphere (default: 's')

    Returns:
    --------
    densities : array
        Density values for each mesh cell
    vp : array
        Vp values for each mesh cell
    """
    from pyproj import CRS, Transformer

    try:
        # Load the processed tomography data
        print("Loading tomography data...")
        tomo_data = pd.read_csv(tomo_file)

        print(f"Loaded {len(tomo_data)} tomography points")
        print(
            f"Density range: {tomo_data['rho'].min():.3f} to {tomo_data['rho'].max():.3f} g/cm³"
        )

        # Handle coordinate transformation if UTM zone is specified
        if utm_zone is not None:
            print(
                f"Converting coordinates from lon/lat to UTM Zone {utm_zone}{hemisphere.upper()}"
            )

            # Setup coordinate transformation
            epsg_code = utm_zone + (32600 if hemisphere.lower() == "n" else 32700)
            crs_wgs84 = CRS.from_epsg(4326)
            crs_utm = CRS.from_epsg(epsg_code)
            transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)

            # Convert coordinates (assuming x=longitude, y=latitude in the CSV)
            utm_x, utm_y = transformer.transform(
                tomo_data["x"].values, tomo_data["y"].values
            )

            # Prepare tomography coordinates (X, Y, Z in meters)
            tomo_coords = np.column_stack(
                [
                    utm_x,
                    utm_y,
                    -tomo_data["z"].values
                    * 1000,  # Convert depth from km to meters, negative for SimPEG convention
                ]
            )

            print(f"Coordinate transformation complete:")
            print(f"  X range: {utm_x.min():.0f} to {utm_x.max():.0f} m")
            print(f"  Y range: {utm_y.min():.0f} to {utm_y.max():.0f} m")

        else:
            print("Using coordinates as-is (assuming already in UTM)")
            # Prepare tomography coordinates (X, Y, Z in meters)
            tomo_coords = np.column_stack(
                [
                    tomo_data["x"].values,
                    tomo_data["y"].values,
                    -tomo_data["z"].values
                    * 1000,  # Convert depth from km to meters, negative for SimPEG convention
                ]
            )

        # Prepare cell coordinates
        cell_coords = cell_centers  # Already in the right format (X, Y, Z in meters)

        # Convert tomography densities to contrasts (if needed)
        tomo_densities = tomo_data["rho"].values
        tomo_vp = tomo_data["Vp_kms"].values

        # Filter densities between 2.4 and 3.1 g/cm³
        mask = (tomo_densities >= 2.65) & (tomo_densities <= 3.15)
        tomo_coords = tomo_coords[mask]
        tomo_densities = tomo_densities[mask]
        tomo_vp = tomo_vp[mask]
        print(
            f"Density range after filtering: {tomo_densities.min():.3f} to {tomo_densities.max():.3f} g/cm³"
        )

        if np.mean(tomo_densities) > 1.0:
            print("Converting absolute densities to contrasts (subtracting 2.8 g/cm³)")
            tomo_densities = tomo_densities - 2.8
        else:
            print("Using tomography values as density contrasts")

        # Build KDTree for efficient nearest neighbor search
        print("Building spatial index for efficient interpolation...")
        tree = cKDTree(tomo_coords)

        # Find nearest neighbor for each mesh cell
        print(f"Finding nearest tomography point for {len(cell_coords)} mesh cells...")
        distances, indices = tree.query(cell_coords, k=1)

        # Assign the density value from the closest tomography point to each mesh cell
        densities = tomo_densities[indices]
        vp = tomo_vp[indices]

        print(
            f"✓ Successfully assigned tomography densities to all {len(cell_coords)} cells"
        )
        print(f"Distance statistics:")
        print(f"  Mean distance to nearest point: {np.mean(distances):.0f} m")
        print(f"  Median distance: {np.median(distances):.0f} m")
        print(f"  Max distance: {np.max(distances):.0f} m")
        print(f"  Min distance: {np.min(distances):.0f} m")
        print(
            f"Final density range: {densities.min():.4f} to {densities.max():.4f} g/cm³"
        )

        return densities, vp

    except FileNotFoundError:
        print(f"ERROR: Tomography file {tomo_file} not found!")
        print("Falling back to depth-dependent density model.")
        fallback = depth_dependent_density(cell_centers, 1e-4)
        return fallback, np.zeros_like(fallback)
    except Exception as e:
        print(f"ERROR loading tomography data: {e}")
        print("Falling back to depth-dependent density model.")
        fallback = depth_dependent_density(cell_centers, 1e-4)
        return fallback, np.zeros_like(fallback)


def main():
    """Main inversion workflow"""

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    USE_TOMOGRAPHY = False  # Set to True to use tomography-based starting model

    print("=" * 80)
    if USE_TOMOGRAPHY:
        print("SimPEG Gravity Inversion - Two-Stage Process (with Tomography)")
    else:
        print("SimPEG Gravity Inversion - Single Stage (Background Density)")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load Input Data
    # =========================================================================
    print("\n[1/8] Loading input data...")

    # File paths
    topo_filename = "../data_input/utm_crust.csv"
    lower_filename = "../data_input/utm_lc.csv"
    data_filename = "../data_input/gravity_disturbance.csv"

    # Load topography (xyz file)
    topo_xyz = np.loadtxt(str(topo_filename), skiprows=1)
    print(f"  Loaded topography: {topo_xyz.shape[0]} points")

    # Load lower boundary (xyz file)
    lower_xyz = np.loadtxt(str(lower_filename), skiprows=1)
    print(f"  Loaded lower boundary: {lower_xyz.shape[0]} points")

    # Load field data (xyz file)
    data_array = np.loadtxt(str(data_filename), skiprows=1)[:, [0, 1, 2, -1]]
    receiver_locations = data_array[:, 0:3]
    dobs = data_array[:, -1]
    print(f"  Loaded gravity data: {len(dobs)} observations")

    # Keep original observations for reference
    dobs_original = dobs.copy()

    # =========================================================================
    # STEP 2: Create Mesh
    # =========================================================================
    print("\n[2/8] Creating octree mesh...")

    # Define base cell widths (in meters)
    dx = 2500  # base cell width in x
    dy = 2500  # base cell width in y
    dz = 500  # base cell width in z

    # Define domain extents from topo_xyz
    x_min, x_max = topo_xyz[:, 0].min(), topo_xyz[:, 0].max()
    y_min, y_max = topo_xyz[:, 1].min(), topo_xyz[:, 1].max()
    z_min, z_max = lower_xyz[:, 2].min(), topo_xyz[:, 2].max()

    x_length = x_max - x_min
    y_length = y_max - y_min
    z_length = z_max - z_min

    print(
        f"  Domain: X={x_length/1000:.1f} km, Y={y_length/1000:.1f} km, Z={z_length/1000:.1f} km"
    )

    # Compute number of base mesh cells required in each direction
    nbcx = next_pow2(x_length / dx)
    nbcy = next_pow2(y_length / dy)
    nbcz = next_pow2(z_length / dz)

    # Define the base mesh
    hx = [(dx, nbcx)]
    hy = [(dy, nbcy)]
    hz = [(dz, nbcz)]
    mesh = TreeMesh([hx, hy, hz], x0=[x_min, y_min, z_min])

    # Refine surface topography
    mesh.refine_surface(topo_xyz, level=-1, finalize=False)

    # Refine Moho/lower boundary surface
    xyz_moho = lower_xyz
    mesh.refine_surface(xyz_moho, level=-1, finalize=False)
    mesh.refine_bounding_box(
        points=[[x_min, y_min, z_min], [x_max, y_max, z_max]], level=-5
    )

    mesh.finalize()
    print(f"  Mesh created: {mesh.nC} cells")

    # =========================================================================
    # STEP 3: Define Active Cells
    # =========================================================================
    print("\n[3/8] Defining active cells...")

    active_tensor_cells_1 = active_from_xyz(mesh, topo_xyz)
    active_tensor_cells_2 = active_from_xyz(mesh, lower_xyz)
    active_cells = np.logical_and(active_tensor_cells_1, ~active_tensor_cells_2)

    # Filter out cells where topography and lower boundary thickness is negligible
    # Use horizontal nearest-neighbor search to estimate thickness at candidate cell positions
    topo_xy = topo_xyz[:, :2]
    lower_xy = lower_xyz[:, :2]
    topo_tree = cKDTree(topo_xy)
    lower_tree = cKDTree(lower_xy)

    # Query positions of all candidate active cell centers (before finalizing mask)
    candidate_centers = mesh.cell_centers[active_cells]
    _, idx_t = topo_tree.query(candidate_centers[:, :2], k=1)
    _, idx_l = lower_tree.query(candidate_centers[:, :2], k=1)
    topo_at_cells = topo_xyz[idx_t, 2]
    lower_at_cells = lower_xyz[idx_l, 2]
    thickness = topo_at_cells - lower_at_cells  # positive if topo above lower

    # Minimum thickness threshold: small absolute value or small fraction of dz
    min_thickness = max(1e-3, dz * 1e-3)  # 1 mm or 0.1% of dz, tune as needed

    # Build keep mask and update active_cells
    keep_mask = thickness > min_thickness
    candidate_indices = np.where(active_cells)[0]
    removed_indices = candidate_indices[~keep_mask]
    if removed_indices.size > 0:
        active_cells[removed_indices] = False

    n_active = int(active_cells.sum())
    print(f"  Active cells: {n_active} / {mesh.nC} ({100*n_active/mesh.nC:.1f}%)")
    if removed_indices.size > 0:
        print(f"  Filtered tiny-thickness cells: removed {removed_indices.size} cells (min_thickness={min_thickness:.4e} m)")

    # Define model mapping
    model_map = maps.IdentityMap(nP=n_active)
    background_density = 1e-4

    # Get active cell centers
    active_cell_centers = mesh.cell_centers[active_cells]

    # =========================================================================
    # STEP 4: Create Initial Model (Conditional on USE_TOMOGRAPHY flag)
    # =========================================================================
    if USE_TOMOGRAPHY:
        print("\n[4/8] Creating initial tomography-based density model...")

        try:
            continuous_model, model_vp = load_tomography_density(
                active_cell_centers, "data/tomography/saci_crust_utm20s.csv"
            )
            print("✓ Successfully loaded tomography-based density model")
        except Exception as e:
            print(f"⚠ Failed to load tomography: {e}")
            print("⚠ Falling back to background density model")
            continuous_model = background_density * np.ones(n_active)
            model_vp = np.zeros_like(continuous_model)
            USE_TOMOGRAPHY = False  # Override flag due to loading failure

        initial_model = continuous_model.copy()

        print(f"Initial model statistics:")
        print(f"  Mean: {initial_model.mean():.4f} g/cm³")
        print(f"  Std:  {initial_model.std():.4f} g/cm³")
        print(f"  Min:  {initial_model.min():.4f} g/cm³")
        print(f"  Max:  {initial_model.max():.4f} g/cm³")
    else:
        print("\n[4/8] Using background density model (skipping tomography)...")
        continuous_model = background_density * np.ones(n_active)
        model_vp = np.zeros_like(continuous_model)
        initial_model = continuous_model.copy()
        print(f"  Background density: {background_density:.4f} g/cm³")

    # =========================================================================
    # STEP 5: Forward Model Initial Model (Stage 1) - Only if using tomography
    # =========================================================================
    if USE_TOMOGRAPHY:
        print("\n[5/8] Forward modeling initial tomography model...")

        # Setup initial survey and data object
        maximum_anomaly = np.max(np.abs(dobs))
        floor_uncertainty = 0.02 * maximum_anomaly
        uncertainties = floor_uncertainty * np.ones(np.shape(dobs))
        print(f"  Floor uncertainty: {floor_uncertainty:.4f} mGal")

        # Define receivers and survey
        receiver_list = gravity.receivers.Point(receiver_locations, components="gz")
        receiver_list = [receiver_list]
        source_field = gravity.sources.SourceField(receiver_list=receiver_list)
        survey = gravity.survey.Survey(source_field)
        data_object_stage1 = data.Data(
            survey, dobs=dobs, standard_deviation=uncertainties
        )

        # Create simulation for initial model
        simulation_stage1 = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=model_map,
            ind_active=active_cells,
            engine="choclo",
        )

        # Compute predicted data from initial model
        dpred_stage1 = simulation_stage1.dpred(initial_model)
        print(f"  Forward model complete")
        print(
            f"  Predicted anomaly range: [{dpred_stage1.min():.2f}, {dpred_stage1.max():.2f}] mGal"
        )

        # Add predicted data to observations (for stage 2 inversion)
        dobs_inversion = dobs + dpred_stage1
        print(f"  Modified observations for stage 2")
        print(
            f"  New anomaly range: [{dobs_inversion.min():.2f}, {dobs_inversion.max():.2f}] mGal"
        )
    else:
        print("\n[5/8] Skipping forward modeling (using original observations)...")
        dobs_inversion = dobs.copy()
        print(f"  Using original gravity observations")
        print(
            f"  Anomaly range: [{dobs_inversion.min():.2f}, {dobs_inversion.max():.2f}] mGal"
        )

    # =========================================================================
    # STEP 6: Setup Inversion
    # =========================================================================
    if USE_TOMOGRAPHY:
        print("\n[6/8] Setting up stage 2 inversion (residual corrections)...")
    else:
        print("\n[6/8] Setting up inversion...")

    # Define receivers and survey (if not already defined)
    if not USE_TOMOGRAPHY:
        receiver_list = gravity.receivers.Point(receiver_locations, components="gz")
        receiver_list = [receiver_list]
        source_field = gravity.sources.SourceField(receiver_list=receiver_list)
        survey = gravity.survey.Survey(source_field)

    # Update data object with inversion observations
    maximum_anomaly_inv = np.max(np.abs(dobs_inversion))
    floor_uncertainty_inv = 0.02 * maximum_anomaly_inv
    uncertainties_inv = floor_uncertainty_inv * np.ones(np.shape(dobs_inversion))
    print(f"  Floor uncertainty: {floor_uncertainty_inv:.4f} mGal")

    data_object_inversion = data.Data(
        survey, dobs=dobs_inversion, standard_deviation=uncertainties_inv
    )

    # Starting model for inversion (small background)
    starting_model = background_density * np.ones(n_active)

    # Map for visualization
    plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

    # Create simulation
    simulation_inversion = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        ind_active=active_cells,
        engine="choclo",
    )

    # Setup data misfit
    dmis = data_misfit.L2DataMisfit(
        data=data_object_inversion, simulation=simulation_inversion
    )

    # Compute initial predicted data
    dpred_initial = simulation_inversion.dpred(starting_model)

    # =========================================================================
    # STEP 7: Run Inversion
    # =========================================================================
    if USE_TOMOGRAPHY:
        print("\n[7/8] Running stage 2 inversion...")
    else:
        print("\n[7/8] Running inversion...")

    # Inversion directives
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    # target_misfit = directives.TargetMisfit(chifact=1)

    save_iteration = directives.SaveModelEveryIteration("/tmp")
    update_jacobi = directives.UpdatePreconditioner()

    update_IRLS = directives.Update_IRLS(
        f_min_change=1e-4,
        max_irls_iterations=30,
        coolingFactor=1.5,  # Note: 'coolFact' not 'irls_cooling_factor'
    )

    sensitivity_weights = directives.UpdateSensitivityWeights(
        every_iteration=False,
        threshold_method="percentile",
        threshold_value=0.01,
        normalization_method="maximum",
    )

    directives_list = [
        sensitivity_weights,
        starting_beta,
        update_IRLS,
        save_iteration,
        update_jacobi,
        # target_misfit,
    ]

    # Define regularization
    reg_sensw = regularization.Sparse(
        mesh,
        active_cells=active_cells,
        mapping=model_map,
        irls_scaled=True,
        irls_threshold=1e-8,
    )
    reg_sensw.norms = [2, 2, 2, 2]

    # Define optimization
    opt = optimization.ProjectedGNCG(
        maxIter=30,
        lower=background_density - 0.2,
        upper=background_density + 0.2,
        maxIterLS=30,
        maxIterCG=10,
        tolCG=1e-8,
    )

    # Define inverse problem
    inv_prob_sensw = inverse_problem.BaseInvProblem(dmis, reg_sensw, opt)

    # Combine inverse problem and directives
    inv = inversion.BaseInversion(inv_prob_sensw, directives_list)

    # Run inversion
    print("  Starting inversion iterations...")
    recovered_model_sensw = inv.run(starting_model)
    print("✓ Inversion complete")

    # =========================================================================
    # STEP 7.5: Convert iteration models (.npy) to VTU for ParaView
    # =========================================================================
    # print("\n[7.5/8] Converting iteration models to VTU format...")

    # # Create vtus directory and sequence subdirectory
    # vtus_dir = "vtus_2"
    # sequence_dir = os.path.join(vtus_dir, "sequence")
    # os.makedirs(vtus_dir, exist_ok=True)
    # os.makedirs(sequence_dir, exist_ok=True)

    # print(f"  🔍 Searching for .npy files in: /tmp")

    # # Find all .npy files in the /tmp directory
    # npy_search_path = "/tmp/*.npy"
    # npy_files = sorted(glob.glob(npy_search_path))  # Sort for consistent ordering

    # print(f"  ✅ Found {len(npy_files)} .npy files\n")

    # # Create models list for sequence
    # models_list = []

    # # Add initial model (timestep 0) - this is the continuous_model
    # print(f"  000. Initial Model (continuous_model)")
    # if USE_TOMOGRAPHY:
    #     initial_model_full = plotting_map * continuous_model
    # else:
    #     initial_model_full = plotting_map * (background_density * np.ones(n_active))
    # models_list.append((0, "Initial_Model", initial_model_full))

    # # Load each .npy file and add to models list (starting from idx 1)
    # for file_idx, npy_file in enumerate(npy_files):
    #     try:
    #         # Load the numpy array (this is the recovered model from iteration)
    #         iteration_model = np.load(npy_file)

    #         # Get filename without extension for reference
    #         model_name = os.path.splitext(os.path.basename(npy_file))[0]

    #         # Index starts at 1 since 0 is the initial model
    #         idx = file_idx + 1

    #         # Check if the loaded data has the same shape as active cells
    #         if iteration_model.shape == (n_active,):
    #             # Compute: continuous_model - iteration_model
    #             if USE_TOMOGRAPHY:
    #                 model_to_save = continuous_model - iteration_model
    #             else:
    #                 model_to_save = iteration_model

    #             # Data matches active cells - use plotting_map
    #             models_list.append((idx, model_name, plotting_map * model_to_save))
    #             print(f"  {idx:03d}. {model_name}")
    #         elif iteration_model.shape == mesh.shape_cells:
    #             # Data is full mesh format - use directly
    #             models_list.append((idx, model_name, iteration_model))
    #             print(f"  {idx:03d}. {model_name}")
    #         else:
    #             print(
    #                 f"  ⚠ Skipping {model_name}: shape {iteration_model.shape} doesn't match"
    #             )

    #     except Exception as e:
    #         print(f"  ⚠ Error loading {os.path.basename(npy_file)}: {e}")

    # # Save ONLY sequentially numbered files for ParaView time series
    # print(
    #     f"\n  📁 Saving {len(models_list)} sequential VTU files for ParaView time series:"
    # )
    # print(f"     Location: {sequence_dir}\n")

    # for idx, model_name, model_data in models_list:
    #     try:
    #         # Create sequential filename (e.g., model_0000.vtu, model_0001.vtu, etc.)
    #         seq_filename = os.path.join(sequence_dir, f"model_{idx:04d}.vtu")

    #         # Save with consistent field name "density" for ParaView time series
    #         mesh.to_vtk(models={"density": model_data}).save(seq_filename)

    #         if os.path.exists(seq_filename):
    #             file_size = os.path.getsize(seq_filename)
    #             print(f"    ✓ model_{idx:04d}.vtu ({file_size:,} bytes) ← {model_name}")
    #         else:
    #             print(f"    ❌ Failed to create model_{idx:04d}.vtu")

    #     except Exception as e:
    #         print(f"    ❌ Error saving model_{idx:04d}.vtu: {e}")

    # # Save mapping file for reference
    # mapping_file = os.path.join(sequence_dir, "sequence_mapping.json")
    # try:
    #     with open(mapping_file, "w") as f:
    #         mapping = {f"model_{idx:04d}.vtu": name for idx, name, _ in models_list}
    #         json.dump(mapping, f, indent=2)
    #     print(f"\n    ✓ Saved mapping to: sequence_mapping.json")
    # except Exception as e:
    #     print(f"\n    ⚠ Error saving mapping file: {e}")

    # # Summary
    # print(f"\n  {'='*60}")
    # print(f"  🎉 VTU EXPORT COMPLETE!")
    # print(f"  {'='*60}")
    # print(f"    📂 Output directory: {sequence_dir}")
    # print(f"    📊 Files created: {len(models_list)}")
    # print(f"    🏷️  Field name: density")
    # print(f"    📋 Mapping file: sequence_mapping.json")
    # if USE_TOMOGRAPHY:
    #     print(f"    ⏰ Timestep 0: Initial model (continuous_model)")
    #     print(f"    ⏰ Timesteps 1+: continuous_model - iteration_model")
    # else:
    #     print(f"    ⏰ Timestep 0: Initial model (background)")
    #     print(f"    ⏰ Timesteps 1+: iteration_model")
    # print(f"  {'='*60}")

    # =========================================================================
    # STEP 7.6: Create Data Comparison Plot
    # =========================================================================
    print("\n[7.6/8] Creating data comparison plot...")

    # Get predicted data from the inversion
    dpred_sensw = inv_prob_sensw.dpred

    # Create figures directory
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Data arrays: [obs, pred, misfit]
    data_arrays = [np.c_[dobs_inversion, dpred_sensw, dobs_inversion - dpred_sensw]]
    plot_titles = ["Observed", "Predicted", "Data Misfit"]
    plot_units = ["mGal", "mGal", "mGal"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.25)

    for row in range(1):
        v_lim = [
            np.max(np.abs(dobs)),
            np.max(np.abs(data_arrays[row][:, 1])),
            np.max(np.abs(data_arrays[row][:, 2])),
        ]
        for col in range(3):
            ax = axes[col]
            norm = mpl.colors.Normalize(vmin=-v_lim[col], vmax=v_lim[col])
            cplot = plot2Ddata(
                receiver_list[0].locations,
                data_arrays[row][:, col],
                ax=ax,
                ncontour=30,
                contourOpts={"cmap": "bwr", "norm": norm},
            )
            ax.set_title(plot_titles[col], fontsize=14, fontweight="bold")
            ax.set_xlabel("x (m)", fontsize=12)
            if col == 0:
                ax.set_ylabel("y (m)", fontsize=12)
            else:
                ax.set_yticks([])

            # Colorbar
            cbar = plt.colorbar(cplot[0], ax=ax, orientation="horizontal", pad=0.1)
            cbar.set_label(plot_units[col], labelpad=5, fontsize=11)
            cbar.ax.locator_params(nbins=3)

    # Add main title
    fig.suptitle(
        "Gravity Data: Observed vs Predicted",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Save plot
    plot_filename = os.path.join(figures_dir, "data_comparison_obs_pred_misfit.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"  ✓ Plot saved to: {plot_filename}")

    # Also save as PDF
    plot_filename_pdf = os.path.join(figures_dir, "data_comparison_obs_pred_misfit.pdf")
    plt.savefig(plot_filename_pdf, bbox_inches="tight")
    print(f"  ✓ Plot saved to: {plot_filename_pdf}")

    plt.close(fig)

    # Print data statistics
    print(f"\n  Data statistics:")
    print(f"    Observed - mean: {dobs.mean():.2f} mGal, std: {dobs.std():.2f} mGal")
    print(
        f"    Predicted - mean: {dpred_sensw.mean():.2f} mGal, std: {dpred_sensw.std():.2f} mGal"
    )
    print(
        f"    Misfit - mean: {(dobs - dpred_sensw).mean():.2f} mGal, std: {(dobs - dpred_sensw).std():.2f} mGal"
    )
    print(f"    RMS misfit: {np.sqrt(np.mean((dobs - dpred_sensw)**2)):.2f} mGal")

    # =========================================================================
    # STEP 8: Export Results
    # =========================================================================
    print("\n[8/8] Exporting results...")

    # Extract coordinates
    x_coords = active_cell_centers[:, 0]
    y_coords = active_cell_centers[:, 1]
    z_coords = active_cell_centers[:, 2]

    model_combined = continuous_model - recovered_model_sensw

    # Get octree levels for active cells
    active_indices = np.where(active_cells)[0]
    octree_levels = mesh._cell_levels_by_indexes(active_indices)

    # Create DataFrame
    df_model_data = pd.DataFrame(
        {
            "x": x_coords,
            "y": y_coords,
            "z": z_coords,
            "model_sensw": recovered_model_sensw,
            "model_vp": model_vp,
            "model_tomo": continuous_model + 2.8,
            "model_combined": model_combined + 2.8,
            "octree_level": octree_levels,
        }
    )

    # Export to CSV
    if USE_TOMOGRAPHY:
        output_filename = "Suppl_model_tomo.csv"
    else:
        output_filename = "Suppl_model_no_tomo_new.csv"
    df_model_data.to_csv(output_filename, index=False)
    print(f"  ✓ CSV exported to: {output_filename}")
    print(f"  DataFrame shape: {df_model_data.shape}")

    # Export to VTK
    vtk_dir = "vtk_exports"
    os.makedirs(vtk_dir, exist_ok=True)

    models_dict = {
        "recovered_model_sensw": plotting_map * recovered_model_sensw,
        "continuous_model": plotting_map * continuous_model,
        "model_vp": plotting_map * model_vp,
        "combined_model": plotting_map * (continuous_model - recovered_model_sensw),
    }

    if USE_TOMOGRAPHY:
        vtk_filename = os.path.join(vtk_dir, "Suppl_model_tomo")
    else:
        vtk_filename = os.path.join(vtk_dir, "Suppl_model_no_tomo_new")

    # Try to export VTK with multiple fallback methods
    vtk_success = False
    try:
        # Method 1: Modern discretize API
        mesh.write_vtk(vtk_filename + ".vtu", models=models_dict)
        print(f"  ✓ VTK exported to: {vtk_filename}.vtu")
        vtk_success = True
    except (AttributeError, TypeError):
        try:
            # Method 2: Older discretize API
            mesh.writeVTK(vtk_filename, models=models_dict)
            print(f"  ✓ VTK exported to: {vtk_filename}.vtu")
            vtk_success = True
        except (AttributeError, TypeError):
            try:
                # Method 3: Use toVTK method if available
                vtk_obj = mesh.toVTK(models=models_dict)
                vtk_obj.save(vtk_filename + ".vtu")
                print(f"  ✓ VTK exported to: {vtk_filename}.vtu")
                vtk_success = True
            except (AttributeError, TypeError):
                pass

    if not vtk_success:
        print(f"  ⚠ VTK export not available (discretize version may not support it)")
        print(f"  ✓ All models saved in CSV: {output_filename}")
    else:
        print(f"  Models included: {list(models_dict.keys())}")

    # Export gravity data
    gravity_df = pd.DataFrame(
        {
            "x": receiver_locations[:, 0],
            "y": receiver_locations[:, 1],
            "z": receiver_locations[:, 2],
            "observed": dobs,
            "predicted": dpred_sensw,
            "residual": dobs_inversion - dpred_sensw,
        }
    )
    gravity_csv = os.path.join(vtk_dir, "Suppl_model.csv")
    gravity_df.to_csv(gravity_csv, index=False)
    print(f"  ✓ Gravity data exported to: {gravity_csv}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("INVERSION SUMMARY")
    print("=" * 80)
    if USE_TOMOGRAPHY:
        print(f"Initial model (tomography-based):")
        print(f"  Mean: {continuous_model.mean():.4f} g/cm³")
        print(f"  Std:  {continuous_model.std():.4f} g/cm³")
        print(
            f"  Range: [{continuous_model.min():.4f}, {continuous_model.max():.4f}] g/cm³"
        )
        print(f"\nRecovered corrections:")
        print(f"  Mean: {recovered_model_sensw.mean():.4f} g/cm³")
        print(f"  Std:  {recovered_model_sensw.std():.4f} g/cm³")
        print(
            f"  Range: [{recovered_model_sensw.min():.4f}, {recovered_model_sensw.max():.4f}] g/cm³"
        )
        print(f"\nCombined model:")
        print(f"  Mean: {model_combined.mean():.4f} g/cm³")
        print(f"  Std:  {model_combined.std():.4f} g/cm³")
        print(
            f"  Range: [{model_combined.min():.4f}, {model_combined.max():.4f}] g/cm³"
        )
    else:
        print(f"Recovered model (from background density):")
        print(f"  Mean: {recovered_model_sensw.mean():.4f} g/cm³")
        print(f"  Std:  {recovered_model_sensw.std():.4f} g/cm³")
        print(
            f"  Range: [{recovered_model_sensw.min():.4f}, {recovered_model_sensw.max():.4f}] g/cm³"
        )
    print("=" * 80)
    print("✓ All tasks completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
