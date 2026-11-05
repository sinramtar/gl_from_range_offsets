import math
import zlib

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats
from scipy.optimize import minimize

from CEBU import CEBU_GM, ERANataf, ERADist

@njit(cache = True, fastmath = True)
def logsumexp(x):
    """log(sum(exp(x))) of a 1-D float64 array; -inf for an empty array."""
    n = x.shape[0]
    if n == 0:
        return -np.inf

    # find max (scalar)
    m = float(x[0])
    for i in range(1, n):
        xi = float(x[i])
        if xi > m:
            m = xi

    # accumulate exp of scalars only, using math.exp
    s = 0.0
    for i in range(n):
        xi = float(x[i])            # force scalar float
        s += math.exp(xi - m)

    if s == 0.0:
        return -np.inf

    return m + math.log(s)

def kolmogorov_smirnov_distance_U(samples_U):
    samples = np.atleast_2d(samples_U)
    N, dims = samples.shape
    ks_distances = np.zeros(dims)
    ks_pvalues = np.zeros(dims)

    for d in range(dims):
        ksresult = stats.ks_1samp(samples[:, d], stats.norm.cdf)
        ks_distances[d] = ksresult.statistic
        ks_pvalues[d] = ksresult.pvalue

    return ks_distances, ks_pvalues

def get_prior_pdf(param_config, kde_array_length):
    if isinstance(param_config['arg'], str):
        data = pd.read_parquet(param_config['arg'], engine = 'pyarrow', use_threads = True).dropna()
        param = [col for col in data.columns if col in ['thickness', 'error', 'diff_tides']][0]
        arg = data[param].values
        weights = None
        if param == 'thickness':
            arg = arg[arg > 0]
            uncertainty = data.loc[data['thickness'] > 0, 'uncertainty'].values
            weights = 1 / uncertainty
            weights[~np.isfinite(weights)] = np.max(weights[np.isfinite(weights)]) * 200
            weights = weights/np.sum(weights)
        if param_config['dist'] == 'empirical':
            if not param_config.get('weights', False):
                weights = None
            arg = [arg, weights, 'kde', None, {"bw_method":None}]

        param_config['arg'] = arg
    pdf = ERADist.ERADist(param_config['dist'], param_config['type'], param_config['arg'])
    kde = np.linspace(float(param_config['bounds'][0]), float(param_config['bounds'][1]), kde_array_length)
    return pdf, kde

def setup_priors(config, kde_array_length):
    prior_pdfs = []
    kdes = []

    for key in config.keys():
        prior, kde = get_prior_pdf(param_config = config[key], kde_array_length = kde_array_length)
        prior_pdfs.append(prior)
        kdes.append(kde)

    return prior_pdfs, kdes

@njit(parallel = False, fastmath = True, cache = True)
def log_likelihood_changepoint_avg(theta, offset, profile, interpolation_mask, data_error, n_eff):
    """Sigma-marginalised Gaussian log-likelihood.

        logsumexp_i[ -(n_eff/2)*log(2*pi*sigma_i^2) - S(theta)/(2*sigma_i^2) ] - log N_sigma

    theta : (n, 4) samples of Young's modulus (GPa), ice thickness (m), differential
            tide (m) and grounding-line position (m along the profile).
    offset, profile, interpolation_mask : observations, their along-profile positions
            (m) and weights; observations with zero weight or NaN offset are skipped.
    data_error : sample of the error standard deviation sigma (m) to marginalise over.
    n_eff : number of independent observations, i.e. one per range-offset pixel.

    The sum of squares is rescaled by n_eff/n_used so the data term and the
    normalisation count the same observations. The elastic beam model is evaluated
    inline; landward of the grounding line it is zero.
    """
    n_beams = theta.shape[0]
    n_sigma = data_error.shape[0]
    n_obs = profile.shape[0]
    llvec = np.empty(n_beams, dtype = np.float64)

    log_norm = np.empty(n_sigma, dtype = np.float64)
    inv_2var = np.empty(n_sigma, dtype = np.float64)
    for i in range(n_sigma):
        s2 = data_error[i] * data_error[i]
        log_norm[i] = -0.5 * n_eff * math.log(2.0 * math.pi * s2)
        inv_2var[i] = 0.5 / s2

    tmp = np.empty(n_sigma, dtype = np.float64)
    log_n_sigma = math.log(n_sigma)

    for k in range(n_beams):
        E = theta[k, 0]; h = theta[k, 1]; tide = theta[k, 2]; x_gl = theta[k, 3]
        beta = (3.0 * 1030.0 * 9.81 * (1.0 - 0.3 * 0.3) / (E * 1e9 * h * h * h)) ** 0.25

        S = 0.0
        n_used = 0.0
        for j in range(n_obs):
            m = interpolation_mask[j]
            if m == 0.0:
                continue
            o = offset[j]
            if np.isnan(o):
                continue
            bp = beta * (profile[j] - x_gl)
            if bp <= 0.0:
                model = 0.0
            else:
                model = tide * (1.0 - math.exp(-bp) * (math.cos(bp) + math.sin(bp)))
            d = model - o
            S += d * d * m
            n_used += m

        # guard: no-op on decimated input (n_used == n_eff), self-corrects otherwise
        if n_used > 0.0:
            S = S * (n_eff / n_used)

        for i in range(n_sigma):
            tmp[i] = log_norm[i] - S * inv_2var[i]
        llvec[k] = logsumexp(tmp) - log_n_sigma
    return llvec

def get_gaussian_KDE_values(data, array):
    kde = stats.gaussian_kde(data)
    return kde(array)

def find_kde_maximum(data):
    # Create the Gaussian KDE
    kde = stats.gaussian_kde(data.T)
    
    # Define the objective function (negative KDE)
    def objective(x):
        return -kde(x)
    
    # Initial guess (mean of the data)
    initial_guess = np.mean(data, axis=0)
    
    # Perform the optimization to find the maximum
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    
    # The maximum point and its KDE value
    max_point = result.x
    
    return max_point

def seed_from_key(key, base = 0):
    """Stable 32-bit CRC32 seed from a string key, so each (flow-bend, pair) is
    reproducible on its own. `base` is a run-level offset."""
    return zlib.crc32(f"{base}:{key}".encode()) & 0xFFFFFFFF


def bayesian_inversion_grounding_line(desired_quantiles, kdes, N, Nlast, max_steps, tarCoV, k_init, method, loglikelihood, prior, seed = None):
    """Run CEBU for one (flow-bend, acquisition pair) and summarise the posterior.

    `seed` seeds numpy's global random stream, which CEBU draws from throughout.
    `method` must be "GM" (Gaussian-mixture importance sampling density).
    """
    if seed is not None:
        np.random.seed(seed)

    d = len(prior)
    R = np.identity(d)          # independent case

    # object with distribution information
    prior = ERANataf.ERANataf(prior, R)
    ### inference
    if method != "GM":
        raise ValueError(f"unsupported CEBU method {method!r}; only 'GM' is available")
    samplesU, samplesX, v_tot, beta_tot, k_final, evidence, Wlast_normed, f_s_iid = CEBU_GM.CEBU_GM(N, loglikelihood, prior, max_steps, tarCoV, k_init, 2, Nlast)

    ### metrics
    nESS = 1/(1+(np.std(Wlast_normed)/np.mean(Wlast_normed))**2)
    means = np.mean(f_s_iid, axis=0)
    stds = np.std(f_s_iid, axis=0)
    quants = []
    for q in desired_quantiles:
        quants.append(np.quantile(f_s_iid, q, axis=0))
        
    max_kdes = find_kde_maximum(f_s_iid)
    
    max_kdes_marginal = np.array([find_kde_maximum(f_s_iid[:,k]) for k in range(d)])
    kde_values = np.array([get_gaussian_KDE_values(f_s_iid[:,k], ar) for k, ar in enumerate(kdes)])
    
    ks_distance, ks_pvalues = kolmogorov_smirnov_distance_U(samplesU[-1])

    # `evidence` underflows to 0 below log Z = -708; the log is stored as well
    with np.errstate(divide = 'ignore'):
        log_evidence = float(np.log(np.asarray(evidence, dtype = np.float64)))

    ### return
    return {
        "evidence": evidence,
        "log_evidence": log_evidence,
        "seed": seed,
        "max_kdes": max_kdes,
        "max_kdes_marginal": max_kdes_marginal.squeeze(),
        "kde_values": kde_values,
        "means": means,
        "stds": stds,
        "quantiles": quants,
        "samples": f_s_iid,
        "v_tot": v_tot[-1],
        "beta_tot": beta_tot,
        "Wlast_normed": Wlast_normed,
        "nESS": nESS,
        "ks_distance": ks_distance,
        "ks_pvalue": ks_pvalues
    }