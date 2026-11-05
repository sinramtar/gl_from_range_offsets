import os
# Single-threaded per job; the parallelism is across flow-bend lines, via `--jobs`.
os.environ["NUMBA_THREADING_LAYER"] = "safe"
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_global_cache'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import functools
import pathlib
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import zstandard as zstd
from tqdm import tqdm

from data_preparation import range_offsets as ro
from data_preparation import flowlines as fl
from analysis import cebu_utils as cu
from postprocessing.profiles import canonical_flow_bends

repo_root = pathlib.Path(workflow.basedir)
configfile: str(repo_root / 'config.yaml')

directories = config['Directories']
work_dir = pathlib.Path(directories['work_dir']).expanduser().resolve()
workdir: str(work_dir)

roi_name = config['Offset_params']['roi']
expt_name = f"{directories['expt']}_{directories['flowline_variant']}"
data_dir = work_dir / directories['data_dir'] / roi_name
profiles_dir = work_dir / directories['processing_dir'] / roi_name / f"clean_offsets_{directories['flowline_variant']}"
results_dir = work_dir / directories['processing_dir'] / roi_name / expt_name

roi = gpd.read_file(repo_root / 'rois' / f'{roi_name}.geojson').to_crs('EPSG:3031')
flow_bends_file = data_dir / 'hybrid_flowlines.gpkg'
flowlines, _ = canonical_flow_bends(gpd.read_file(flow_bends_file, bbox = tuple(roi.total_bounds)))
flowlines = flowlines.set_index('Id', drop = False)
flowline_ids = flowlines.Id.values
data_error = pd.read_parquet(work_dir / config['Data_error']['path'], engine = 'pyarrow', use_threads = True)

offsets = []
times = []
for f in sorted(data_dir.glob('detrended_range_offsets*.tif')):
    with rio.open(f) as tif:
        window = rio.windows.from_bounds(left = roi.total_bounds[0], right = roi.total_bounds[2],
                                     bottom =  roi.total_bounds[1], top = roi.total_bounds[3],
                                     transform = tif.transform)
        offsets.append(tif.read(window = window))
        transform = rio.windows.transform(window = window, transform = tif.transform)
        times.append(tif.descriptions)
offsets = np.concatenate(offsets, axis = 0)
times = np.concatenate(times)

N = int(config['CEBU_params']['N'])
# One likelihood observation per range-offset pixel; see config Likelihood.sample_step.
vertex_step = float(config['Flowline_params']['step'])
stride = max(1, int(round(config['Likelihood']['sample_step'] / vertex_step)))
seed_base = int(config['CEBU_params']['seed'])
priors, kdes = cu.setup_priors(config = config['Prior'], kde_array_length = config['Plot_params']['Nplot'])

rule all:
    input:
        expand(str(profiles_dir / '{id}'), id = flowline_ids),
        expand(str(results_dir / '{id}'), id = flowline_ids)

rule extract_profiles:
    input:
        flow_bends = str(flow_bends_file),
    output:
        directory(str(profiles_dir / '{id}'))
    threads: 1
    resources:
        mem = '32GB'
    run:
        flowline = flowlines.loc[[int(wildcards.id)]]
        clean_offsets_dir = pathlib.Path(output[0])
        clean_offsets_dir.mkdir(exist_ok = True, parents = True)

        offsets_along_flowline = np.atleast_2d(fl.sample_raster_along_single_flowline(gdf = flowline, raster = offsets, transform = transform))
        offsets_along_flowline = offsets_along_flowline.astype(np.float32, copy=False)
        for t in tqdm(range(offsets_along_flowline.shape[1])):
            ro.save_complete_profile(offset = offsets_along_flowline[:, t], time = times[t], savedir = clean_offsets_dir)

rule apply_bayesian_inversion:
    input:
        clean_offset = str(profiles_dir / '{id}')
    output:
        bay_inv = directory(str(results_dir / '{id}'))
    threads: 1
    resources:
        runtime = "8h",
        mem = "64GB"
    run:
        clean_offsets_dir = pathlib.Path(input.clean_offset)
        out_dir = pathlib.Path(output.bay_inv)
        out_dir.mkdir(exist_ok = True, parents = True)
        err = np.ascontiguousarray(data_error['error'].values, dtype = np.float64)

        for npz in tqdm(clean_offsets_dir.glob('*.npz'), desc = 'Bayesian inversion...'):
            out_file = out_dir / (npz.name.removesuffix('.npz') + '.zstd')
            if out_file.exists():
                continue
            clean_offset = np.load(npz, allow_pickle = True)['arr_0']
            # true along-line distance: lines that end early are shorter than 40 km
            profile_full = np.arange(clean_offset.shape[1]) * vertex_step

            # one observation per raster pixel
            offset = np.ascontiguousarray(clean_offset[0, ::stride], dtype = np.float64)
            mask = np.ascontiguousarray(clean_offset[1, ::stride], dtype = np.float64)
            profile = np.ascontiguousarray(profile_full[::stride], dtype = np.float64)
            n_eff = float(np.count_nonzero((mask > 0) & np.isfinite(offset)))

            pair_seed = cu.seed_from_key(f'{wildcards.id}/{npz.name.removesuffix(".npz")}', base = seed_base)
            try:
                results_bayinv = cu.bayesian_inversion_grounding_line(desired_quantiles = config['CEBU_params']['quantiles'], N = N,
                                                                Nlast = N * config['CEBU_params']['Nlast'],
                                                                max_steps = config['CEBU_params']['max_steps'], tarCoV = config['CEBU_params']['tarCoV'],
                                                                k_init = config['CEBU_params']['k_init'],
                                                                method = config['CEBU_params']['method'],
                                                                loglikelihood = functools.partial(cu.log_likelihood_changepoint_avg,
                                                                offset = offset, profile = profile,
                                                                interpolation_mask = mask,
                                                                data_error = err,
                                                                n_eff = n_eff),
                                                                prior = priors, kdes = kdes, seed = pair_seed)
                results_bayinv['n_eff'] = n_eff
                results_bayinv['seed'] = pair_seed
                compressed = zstd.ZstdCompressor(level = 5).compress(pickle.dumps(results_bayinv))
                with open(out_file, "wb") as f:
                    f.write(compressed)
            except Exception as e:
                print(f'Failed: {npz.name} with exception: {e}')
