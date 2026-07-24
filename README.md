# 3D Lithospheric Density Model of the Central Andes

This repository contains the core scripts, input constraints, and output data necessary to reproduce the gravity inversion and stress analysis presented in:

**[Insert Paper Citation Here]**

## Directory Structure

### `/data_input`

- The regional gravity disturbance grid (e.g., extracted from XGM2019e).
- The structural boundaries (Moho depth, sediments, basement, topography).
- Sampled model for stress analysis (model_prep.csv)

### `/scripts`
- `simpeg_cluster.py`  - The SimPEG script used to perform the gravity inversion.
- `calculate_lithostatic_stress.py`  - The script used to calculate the vertical normal stress gradients.
- `plot_stress_gradient_figures.py` - The script used to plot the stress gradient figures.
*(Note: Minor pre-processing and plotting scripts are not included).*

### `/data_output`
- The final 3D inverted density model of the crystalline crust (inversion_model.csv).

## Third-Party Data Requirements
The following datasets were used to build the initial models but are hosted in their own respective repositories. They are not duplicated here:
- **SACI-24:** [(https://doi.org/10.1016/j.gr.2024.10.012)]
- **V2Rho_Gibbs:** [https://doi.org/10.5281/zenodo.6538257]
- **ETOPO1:** [https://www.ncei.noaa.gov/products/etopo-global-relief-model]
- **Earth Crustal Model 1:** [https://doi.org/10.1016/j.earscirev.2023.104493]
- **Earthquake Catalog (Riedel-Hornig et al., 2026):** [https://doi.org/10.5194/essd-18-1-2026]

## Usage
The simpeg_cluste.py script performs the Simpeg inversion given the inputs for gravity and structural boundaryies (utm_crust and utm_lc). Also takes saci_crust_utm20s as constrains.
Calculate_lithostatic_stress.py takes the model_prep.csv and calculates the vertical normal stress gradients. (Should be straight forward)
