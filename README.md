# Grounding-line inference from SAR range offsets

This repository infers Antarctic grounding-line (GL) positions with posterior
uncertainty from Sentinel-1 range offsets. Range offsets are extracted along
flow-bend profiles, a one-dimensional elastic beam model of tidal flexure is fitted
to every (profile, acquisition pair) combination by Bayesian inversion with CEBU
(cross-entropy-based importance sampling for Bayesian updating), and the retained
posteriors are turned into GL points, credible intervals, comparisons with DInSAR
delineations, calibration diagnostics and figures. It was applied to the Larsen C Ice
Shelf for 2019–2021.

## Repository layout

| Path | Content |
| --- | --- |
| `config.yaml` | All workflow settings and scientific parameters |
| `data_preparation/` | Snakefile and modules that build the inversion inputs |
| `Snakefile` | Profile extraction and Bayesian inversion |
| `postprocessing/` | Snakefile and modules for products, distances and calibration |
| `analysis/cebu_utils.py` | Priors, likelihood and the per-profile inversion |
| `CEBU/` | CEBU implementation of Engel et al. (2023), see `CEBU/README.md` |
| `visualizations/` | Figure scripts |
| `rois/` | Region-of-interest polygons (`LarsenC`, and `Crane` for a map inset) |
| `tests/` | Unit tests (`pytest tests`) |

## Installation

```bash
conda env create -f environment.yml
conda activate icesat
```

`environment.yml` installs this repository in editable mode as the package
`gl_from_range_offsets`. The commands below are run from the repository root. The
incidence-angle rules additionally need ESA SNAP with its Python bindings
(`esa_snappy`, `jpy`), which are not available from conda-forge.

## Configuration

Relative entries of `Directories` are resolved from `Directories.work_dir`, which is
itself resolved from the directory Snakemake is launched in. With the default
`work_dir: ..`, inputs live in `../data/<roi>/`, inversion results in
`../inversion_outputs/<roi>/<expt>_<flowline_variant>/` and products in
`../results/<roi>/<expt>_<flowline_variant>/`. The experiment name of the published
run is `bedmap_weighted_hybrid_corrected`.

## Inputs

The ROI data directory (`data/<roi>/`) holds:

- `2019/`, `2020/`, `2021/`: Sentinel-1 range-offset rasters (`*.slfmap.tif`), in range
  pixels per day, with the acquisition dates in the file name
- `slc/` (configured as `Directories.slc`): the Sentinel-1 IW SLCs (`*.SAFE.zip`) used for
  the incidence angles
- `all_correlation_coeffs.tif`: per-pair offset-tracking correlation coefficients, one
  band per pair, band descriptions `YYYYMMDD_YYYYMMDD`
- `tmoc_gl_2020.geojson`: the reference GL with an integer `Id` per part
- `all_dinsar/`: one GeoJSON per DInSAR delineation; `dinsar_uncertainty/`: the ensemble
  standard-deviation polygons of the automatic delineations
- `bedmap3.tif` (thickness and its uncertainty) for the covariates

The configured `data/` directory also holds the ice-velocity rasters, the CATS2008-v2023
tide model (`CATS2008_v2023/`), NCEP/NCAR sea level pressure (`air_pressure/slp.<year>.nc`),
the categorised basemap, the RADARSAT mosaic and the Bedmap3 thickness table
(`Prior.h.arg`, columns `thickness` and `uncertainty`).

## Workflow

### 1. Data preparation

```bash
snakemake --snakefile data_preparation/Snakefile --cores 8
```

Run once per `Offset_params.year`. It builds, on the range-offsets grid:

- `merged_incidence_angles.tif`: incidence angle from the ellipsoid of every SLC
  (orbit correction, deburst and terrain correction in SNAP), reprojected onto the
  offsets grid and mosaicked by averaging overlaps
- `detrended_range_offsets_<year>.tif`: offsets of the configured temporal baseline,
  converted to metres (`range_pixel_spacing`), detrended by the per-pixel mean of the
  year and divided by the cosine of the incidence angle
- `corrected_tides_<year>/`, `differential_tides_<year>.tif`: CATS2008-v2023 tides plus the
  inverse-barometer correction at the acquisition times, which are queried from the
  Copernicus Data Space catalogue (network access, no account needed)
- `correlation_<year>.tif` and `tmoc_gl_<year>.geojson`: Spearman correlation of the
  offsets with the differential tides, and the reference GL obtained by thresholding it
  (`Offset_params.correlation_threshold`)
- `hybrid_grad_field.tif`: the flexure/flow gradient field, using the Gaussian smoothing
  `Flowline_params.sigma` (pixels) and the bending-zone thresholds `lower_stddev` /
  `upper_stddev` (m) of the offsets standard deviation
- `points/`, `hybrid_flowlines/` and `hybrid_flowlines.gpkg`: profile seeds every
  `flowline_spacing` m along the reference GL, and the flow-bend lines traced
  `landward` / `seaward` m from them. Lines with backtracking steps
  (`max_reversal_fraction`) are rejected. The GeoPackage holds `Id`, `ref_line_id` (the
  reference-GL part the seed lies on) and the line geometry.
- `data_error.tif` and the error sample `Data_error.path` (see below)
- `tidal_range.tif`: total CATS2008-v2023 tidal range, the basis of the ΔA prior bounds

The detrended offsets of all three years are needed before the gradient field and the
error raster can be built.

### 2. Inversion

```bash
snakemake --cores 64
```

For every distinct flow-bend line of `hybrid_flowlines.gpkg` (exact duplicate geometries
keep their lowest `Id`), the detrended offsets are sampled along the line at its 2 m
vertices. Only acquisition pairs whose profile has no missing sample are kept. Each
profile is then inverted with CEBU; one `<primary>_<secondary>.zstd` result per pair is
written to `inversion_outputs/<roi>/<expt>_<flowline_variant>/<Id>/`. Each result is a
zstd-compressed pickle with the posterior samples, the joint and marginal KDE maxima,
the normalised effective sample size (`nESS`), the evidence, the KS distances to the
prior, `n_eff` and the seed.

### 3. Postprocessing

```bash
snakemake --snakefile postprocessing/Snakefile --cores 8 postprocess
snakemake --snakefile postprocessing/Snakefile --cores 8 calibration
snakemake --snakefile postprocessing/Snakefile --cores 8 paper
```

`postprocess` builds the point and interval products, `calibration` the credible-interval
calibration, and `paper` the distance table, the posterior-spread summary and the
figures. The targets read existing inversion results; they do not rerun inversions.
Every product has a JSON sidecar recording its inputs, settings, seed and code revision.
Individual modules expose their arguments through `--help`.

## Model and likelihood

The forward model is the elastic beam solution

```text
G(x) = ΔA (1 - exp(-β(x - x_GL)) (cos β(x - x_GL) + sin β(x - x_GL))),  x > x_GL
β = (3 ρ_w g (1 - ν²) / (E h³))^(1/4),   ρ_w = 1030 kg m⁻³, g = 9.81 m s⁻², ν = 0.3
```

and zero landward of `x_GL`. The parameters are Young's modulus `E` (GPa), ice thickness
`h` (m), the differential tide `ΔA` (m) and the GL position `x_GL` (m along the profile).
The inversion uses one shared standard deviation, sigma, for all observations on a
profile and marginalises it over the supplied error sample:

```text
S(theta) = (n_eff / n_used) * sum_j m_j * (y_j - G(theta, x_j))^2
log L_i(theta) = -(n_eff / 2) * log(2*pi*sigma_i^2) - S(theta)/(2*sigma_i^2)
log L(theta) = logsumexp_i(log L_i(theta)) - log(N_sigma)
```

The profile is subsampled from its 2 m vertices to `Likelihood.sample_step` (200 m, one
observation per range-offset pixel), so a complete 40 km profile has `n_eff = 201`
observations. `m_j` are the observation weights stored with each profile; they are all
one, so `n_used = n_eff`. Conditional on sigma the observations are treated as
independent. `N_sigma` is the number of values in the error sample; every value is
used. The implementation is `analysis/cebu_utils.py::log_likelihood_changepoint_avg`.

## Range-offset errors

`data_error.tif` assigns every pixel of every pair an error standard deviation: pixels
with ice speed below 0.01 m/day are treated as stable, and the standard deviation of the
stable-pixel offsets is computed in ten equal bins of the pair's own correlation
coefficient, closed on the right ((0, 0.1], …, (0.9, 1.0]). The error sample passed to
the likelihood (`Data_error.path`) contains the finite errors of all pairs at pixels whose
correlation coefficient is at least `Data_error.correlation_threshold`.

## Priors

`E`, `ΔA` and `x_GL` have uniform priors on `Prior.<name>.bounds`; `x_GL` on
[15, 25] km is ±5 km around the reference GL, which lies 20 km along each complete
profile. The thickness prior is empirical: `analysis/cebu_utils.py::get_prior_pdf` reads
the thickness table, drops rows with missing values and keeps positive thicknesses.
With `weights: true`, the normalised weight of sample i is proportional to
`1 / uncertainty_i` (not inverse variance); nonfinite inverse weights are set to 200 times
the largest finite one. The weighted step eCDF and its inverse define the
transformation to standard-normal space and the sampling; a weighted KDE is used only to
evaluate the PDF, on the grid `Prior.h.bounds`. The configured Bedmap3 table gives
392,965 usable samples.

## Products

`postprocessing/products.py` writes, for the distinct flow-bend lines:

- `paper_points.parquet`: one GL point per retained (line, pair), located at the
  `x_GL` coordinate of the joint posterior mode (maximum of the four-dimensional KDE).
  Retained pairs satisfy `nESS >= Posterior_params.nESS` (0.7) and
  `|ΔA| >= Posterior_params.tide` (0.3 m), with `ΔA` the inferred differential tide
  (column `diff_tides`), not the modelled one. The published run has 128,775 points from
  169 acquisition pairs on 2,986 lines.
- `pair_intervals.parquet`: the 68 % and 95 % central credible intervals of every
  retained pair posterior, in metres along the line, with the inversion diagnostics and
  the `Covariates` sampled along the seaward half of the line.
- `cumulative_intervals.parquet`: intervals of the equally weighted mixture of the
  retained pair posteriors of each line with at least three retained pairs.

## Distances to DInSAR delineations

`postprocessing/table1_distances.py` computes the distance table:

1. Sentinel-1 delineations are compared with the GL points of their two exact constituent
   acquisition pairs. CSK delineations use the pair with the nearest primary acquisition
   date; `--csk-max-dt` can cap the temporal separation.
2. Points inside the delineation's bounding box are assigned to its nearest line part by
   Euclidean distance and must also lie inside the bounding box of that part.
3. Signs come from the categorised basemap: in 5 km single-sided buffers on either side of
   the matched part, the largest category decides (Land negative, Ice shelf or Ocean
   positive). Negative distances are landward of the delineation, positive seaward.
4. Per delineation and constituent pair, signed distances with absolute modified z-score
   (median and MAD, factor 0.6745) of at least 3 are removed.
5. Distances are pooled by year and dataset. Absolute median, absolute mean and the MAD of
   the absolute distances disregard direction; signed median and mean retain it.

The per-point distances are written to `table1_distances_points.parquet` and `.csv`.

## Calibration

`postprocessing/calibration_truth.py` locates each DInSAR delineation along every
flow-bend line it crosses within `Calibration.coverage_threshold` (100 m).
`postprocessing/reference_uncertainty.py` assigns each crossing a DInSAR delineation
uncertainty: `manual_reference_sigma_m` for manual delineations and, for automatic ones,
the local half-width of the matched ensemble standard-deviation polygon perpendicular to
the delineation. Automatic delineations whose median half-width exceeds
`max_reference_half_width_m` are dropped. `postprocessing/calibrate.py` keeps
Sentinel-1 crossings that are uniquely localised (`identifiable_crossing_tolerance_m`),
inside the `x_GL` prior support and covered by both constituent pair posteriors. Each
crossing is evaluated under the equal-weight mixture of Gaussian approximations of the
two pair posteriors (midpoint and half-width of the 68 % interval), without and with the
delineation uncertainty added in quadrature. Coverage confidence intervals come from a
block bootstrap over delineations (2,000 replicates, seed 0).

## Figures

| Script | Figure |
| --- | --- |
| `visualizations/flowbend_map.py` | Flow-bend lines on the gradient field (Fig. 4a) |
| `visualizations/flowbend_profiles.py` | Example profile and bending zone (Fig. 4b–d) |
| `visualizations/gl_points_map.py` | GL points, cumulative intervals, coverage (Fig. 11) |
| `postprocessing/calibrate.py` | Reliability and PIT (Fig. 12, Fig. S4.5) |
| `visualizations/ci_widths_regplots.py` | Interval widths against controls (Fig. 13) |
| `visualizations/grad_field.py` | Gradient-field construction (Fig. A2) |
| `visualizations/detrending_rmse.py` | Monthly vs yearly detrending (Fig. S1.1) |
| `visualizations/correlation_maps.py` | Offsets–tide correlation (Fig. S1.2) |
| `visualizations/posteriors_sum.py` | Cumulative posterior of one line (Fig. S3.1) |
| `visualizations/csx_comp.py` | CSK comparison example (Fig. S4.1) |
| `visualizations/s1_comp.py` | Sentinel-1 comparison examples (Figs. S4.2, S4.3) |
| `visualizations/violin_dinsar_comp.py` | Signed distance distributions (Fig. S4.4) |
| `visualizations/ci_spatial.py` | Spatial structure of the intervals (Fig. S6.1) |

`grad_field.py`, `detrending_rmse.py`, `correlation_maps.py`, `csx_comp.py` and
`s1_comp.py` read precomputed arrays from `data/figure_inputs/`. Figures 1–3, 5–10, A1,
B1, S2.1 and S5.1 are not produced by this repository.

## Tests

```bash
pytest tests
```
