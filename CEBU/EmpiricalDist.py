import numpy as np
import scipy as scp
import scipy.stats as sps
from scipy.interpolate import interp1d

class EmpDist():
    """
    Returns a distribution object similar to scipy.stats based on a dataset
    given by the user.
    ---------------------------------------------------------------------------
    Developed by: Michael Engel
    ---------------------------------------------------------------------------
    Initial Version: 2025-07
    ---------------------------------------------------------------------------
    """
    ## initialization
    def __init__(
        self,
        data,
        weights = None, # None for equal weights
        pdfMethod = "kde", # kde, linear, slinear, quadratic, cubic, nearest, next...
        pdfPoints = None, # None or integer (only relevant if pdfMethod is not equal to 'KDE')
        **pdfMethodParams
    ):
        '''
        :param data: One dimensional data array.
        :param weights: Weights associated to the data array. The default is None.
        :param pdfMethod: Desired method for the PDF creation. The default is KDE.
        :param pdfPoints: Desired number of points for the PDF creation. The default is None (will resolve in the square root of the number of data points).
        :params pdfMethodParams: Optional scipy.stats.gaussian_kde keyword arguments for the case of KDE based PDF creation.
        '''
        
        self.cleanData = data[~np.isnan(data)]
        self._N = len(self.cleanData)
        self.totalN = len(self.cleanData)
        
        self.normalizedWeights = np.ones_like(self.cleanData)/self._N if weights is None else weights[~np.isnan(data)]/np.sum(weights[~np.isnan(data)])
        self.sumWeights = self._N if weights is None else np.sum(weights[~np.isnan(data)])
        self.totalWeight = self.totalN if weights is None else np.sum(weights[~np.isnan(data)])
        
        self.pdfMethod = pdfMethod
        self.pdfPoints = pdfPoints if pdfPoints is not None else np.max([2,int(np.sqrt(self._N))])
        self.pdfMethodParams = pdfMethodParams
        
        # statistics
        self._mean = np.sum(self.cleanData*self.normalizedWeights)
        self._var = np.cov(self.cleanData, aweights=self.normalizedWeights)
        self._std = np.sqrt(self._var)
        
        # cdf and inverse cdf
        self._cdf = create_weighted_cdf_interp1d(self.cleanData, self.normalizedWeights, kind="previous")
        self._ppf = create_weighted_ppf_interp1d(self.cleanData, self.normalizedWeights, kind="next")
            
        # pdf
        if self.pdfMethod.lower()=="kde":
            # print("EmpDist: Using Gaussian KDE for PDF!")
            dataSorted, weightsSorted = sortDataWeights(self.cleanData, self.normalizedWeights)
            self._pdf = sps.gaussian_kde(dataset=dataSorted, weights=weightsSorted, **pdfMethodParams)
        else:
            # print("EmpDist: Using numerical derivative for PDF!")
            self._pdf = create_normalized_pdf_from_cdf(self._cdf, self.cleanData.min(), self.cleanData.max(),
                num_points = self.pdfPoints,
                kind = self.pdfMethod.lower()
            )
        pass
    
    def __add__(self, otherDist):
        data = np.concatenate([self.cleanData, otherDist.cleanData]).squeeze()
        weights = np.concatenate([self.normalizedWeights*self.sumWeights, otherDist.normalizedWeights*otherDist.sumWeights]).squeeze()
        return EmpDist(data=data, weights=weights, pdfMethod=self.pdfMethod, pdfPoints=self.pdfPoints, **self.pdfMethodParams)
    
    def __radd__(self, other): # support sum()
        if other == 0 or other is None:
            return self
        if isinstance(other, EmpDist) or isinstance(other, EmpDigest):
            return self.__add__(other)
        return NotImplemented
    
    def __iadd__(self, other):
        if other == 0 or other is None:
            return self
        if isinstance(other, EmpDist) or isinstance(other, EmpDigest):
            new = self.__add__(other)
            self.__dict__.update(new.__dict__)
            return self
        return NotImplemented
    
    def __len__(self):
        return self._N
    
    def N(self):
        return self._N
                
    def mean(self):
        return self._mean
    
    def var(self):
        return self._var
    
    def std(self):
        return self._std

    def pdf(self, x):
        return self._pdf(x)
                
    def cdf(self, x):
        return self._cdf(x)
    
    def icdf(self, y):
        return self._ppf(y)
    
    def ppf(self, y):
        return self._ppf(y)
       
    def random(self, size=None): # random samples
        rands = np.random.rand(size)
        return self.icdf(rands)
    
    def rvs(self, size=None):
        rands = np.random.rand(size)
        return self.icdf(rands)
    
class EmpDigest(EmpDist):
    """
    Returns a distribution object similar to scipy.stats based on a dataset
    given by the user using an approximation in standard normal space for a
    chosen number of approximation points.
    ---------------------------------------------------------------------------
    Developed by: Michael Engel
    ---------------------------------------------------------------------------
    Initial Version: 2025-10
    ---------------------------------------------------------------------------
    """
    ## initialization
    def __init__(
        self,
        data,
        weights = None, # None for equal weights
        nApprox = 1000,
        mode = "densityFocused",
        eps = None,
        pdfMethod = "kde", # kde, linear, slinear, quadratic, cubic, nearest, next...
        pdfPoints = None, # None or integer (only relevant if pdfMethod is not equal to 'KDE')
        **pdfMethodParams
    ):
        '''
        :param data: One dimensional data array.
        :param weights: Weights associated to the data array. The default is None.
        :param nApprox: Desired number of points to approximate the empirical CDF in standard normal space. The default is 1000.
        :param mode: Desired mode to construct the approximation points in standard normal space. The default is 'densityFocused'.
        :param eps: Desired infinitesimal to check the maximum distinguishable values in standard normal space. The default is None.
        :param pdfMethod: Desired method for the PDF creation. The default is KDE.
        :param pdfPoints: Desired number of points for the PDF creation. The default is None (will resolve in the square root of the number of data points).
        :params pdfMethodParams: Optional scipy.stats.gaussian_kde keyword arguments for the case of KDE based PDF creation.
        '''
        self.nApprox = nApprox if nApprox is not None else int(np.sqrt(len(data)))
        self.mode = mode
        self.eps = eps
        
        cleanData_pre = data[~np.isnan(data)]
        totalN = len(cleanData_pre)
        totalWeight = totalN if weights is None else np.sum(weights[~np.isnan(data)])
        
        normalizedWeights_pre = np.ones_like(cleanData_pre)/totalWeight if weights is None else weights[~np.isnan(data)]/totalWeight
        ppf_pre = create_weighted_ppf_interp1d(cleanData_pre, normalizedWeights_pre, kind="next")
                
        weights_X, cdfValues = getRepresentativeWeightsAndCdfValues(self.nApprox, mode=self.mode, eps=self.eps)
        points_X = ppf_pre(cdfValues)
        
        super().__init__(
            points_X,
            weights = weights_X, # None for equal weights
            pdfMethod = pdfMethod, # kde, linear, slinear, quadratic, cubic, nearest, next...
            pdfPoints = pdfPoints, # None or integer (only relevant if pdfMethod is not equal to 'KDE')
            **pdfMethodParams
        )
        self.totalWeight = totalWeight
        self.sumWeights = totalWeight
        self.totalN = totalN
        pass
    
    def __add__(self, other):
        points_X = np.concatenate([self.cleanData, other.cleanData]).squeeze()
        weights_X = np.concatenate([self.normalizedWeights*self.sumWeights, other.normalizedWeights*other.sumWeights]).squeeze()
        resultingDigest = EmpDigest(data=points_X, weights=weights_X, nApprox=self.nApprox, mode=self.mode, eps=self.eps, pdfMethod=self.pdfMethod, pdfPoints=self.pdfPoints, **self.pdfMethodParams)
        resultingDigest.totalN = self.totalN + other.totalN
        return resultingDigest
        
def sortDataWeights(data, weights):
    data = np.asarray(data)
    weights = np.asarray(weights)
    
    sort_idx = np.argsort(data)
    sorted_data = data[sort_idx]
    sorted_weights = weights[sort_idx]
    
    return sorted_data, sorted_weights

### CDF utils
# interpolation based cdf
def create_weighted_cdf_interp1d(data, weights, kind="linear"):
    sorted_data, sorted_weights = sortDataWeights(data, weights)
    
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    weighted_cdf_values = cum_weights / total_weight

    cdf_func = interp1d(sorted_data, weighted_cdf_values,
                        kind=kind,
                        bounds_error=False,
                        fill_value=(0.0, 1.0),
                        assume_sorted=True)
    return cdf_func

def create_weighted_ppf_interp1d(data, weights, kind="linear"):
    sorted_data, sorted_weights = sortDataWeights(data, weights)

    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    weighted_cdf_values = cum_weights / total_weight

    cdf_func = interp1d(weighted_cdf_values, sorted_data, 
                        kind=kind,
                        bounds_error=False,
                        fill_value=(sorted_data.min(), sorted_data.max()),
                        assume_sorted=True)
    return cdf_func

### PDF utils
def create_normalized_pdf_from_cdf(cdf_func, x_min, x_max, num_points=1000, kind="linear"):
    x_grid = np.linspace(x_min, x_max, num_points)
    cdf_vals = cdf_func(x_grid)
    
    raw_pdf_vals = np.gradient(cdf_vals, x_grid)
    area = np.trapz(raw_pdf_vals, x_grid)
    
    if area != 0:
        pdf_vals = raw_pdf_vals / area
    else:
        pdf_vals = raw_pdf_vals
    
    pdf_func = interp1d(x_grid, pdf_vals, kind=kind, bounds_error=False, fill_value=0.0)
    return pdf_func

### approximate gaussian utilities
def findStandardNormalCdfValidInterval(eps=1e-2):
    converged = False
    lowerLimit = 0
    while not converged:
        lowerLimit = lowerLimit-eps
        converged = sps.norm.cdf(lowerLimit)==0
        
    lowerLimit = lowerLimit+eps
    
    converged = False
    upperLimit = 0
    while not converged:
        upperLimit = upperLimit+eps
        converged = sps.norm.cdf(upperLimit)==1
    upperLimit = upperLimit-eps
    
    return (lowerLimit, upperLimit)

def findStandardNormalCdfValidInterval_symmetric(eps=1e-3):    
    interval = np.abs(findStandardNormalCdfValidInterval(eps=eps))
    return -np.min(interval), np.min(interval)
EPS_STANDARD_NORMAL_CDF = 1e-3
STANDARD_NORMAL_CDF_VALID_INTERVAL_SYMMETRIC = findStandardNormalCdfValidInterval_symmetric(EPS_STANDARD_NORMAL_CDF)

def standardNormalCdfNthDerivative(x, n=1):
    if not isinstance(n, int) or n < 0:
        raise ValueError("The order of the derivative 'n' must be a non-negative integer.")
    if n==0:
        return sps.norm.cdf(x)
    n = n-1

    phi_x = sps.norm.pdf(x)
    hn_x = scp.special.eval_hermitenorm(n, x)
    return ((-1)**n) * hn_x * phi_x

def artCdf_unnormalized(x):
    x_negative = x[x<=0]
    x_positive = x[x>0]
    return np.concatenate([standardNormalCdfNthDerivative(x_negative, n=1), 2*standardNormalCdfNthDerivative(0, n=1)-standardNormalCdfNthDerivative(x_positive, n=1)])

def artCdf(x):
    y = artCdf_unnormalized(x)
    return y/artCdf_unnormalized(STANDARD_NORMAL_CDF_VALID_INTERVAL_SYMMETRIC[1]+EPS_STANDARD_NORMAL_CDF)

def getRepresentativeStandardNormalWeightsAndCdfValues(N, mode="tailFocused", eps=1e-2):
    raise NotImplementedError(f"{mode} mode not implemented!")
    
    if eps is None:
        interval = STANDARD_NORMAL_CDF_VALID_INTERVAL_SYMMETRIC
        eps = EPS_STANDARD_NORMAL_CDF
    else:
        interval = findStandardNormalCdfValidInterval_symmetric(eps=eps)
    
    if mode=="tailFocused":
        points_U = np.concatenate([[-np.inf, -np.inf], np.linspace(interval[0],interval[1],N-2), [np.inf, np.inf]])
    
    elif mode=="densityFocused":
        minQ, maxQ = sps.norm.cdf(interval)
        quantiles = np.concatenate([[0,0], np.linspace(minQ, maxQ, N-2), [1,1]])
        points_U = sps.norm.ppf(quantiles)
    elif mode=="quantileFocused":
        points_pre = np.linspace(interval[0]-eps,interval[1]+eps,N+2)
        quantiles = artCdf(points_pre)
        points_U = sps.norm.ppf(quantiles)
    else:
        points_U = np.asarray(mode)
        
    weights = sps.norm.cdf(points_U)
    weights = weights[1:]-weights[:-1]
    weights = weights[1:]
    
    cdfValues = sps.norm.cdf(points_U[1:-1])
    
    return weights, cdfValues

def getRepresentativeWeightsAndCdfValues(N, mode="densityFocused", eps=1e-2):
    if mode=="tailFocused":
        return getRepresentativeStandardNormalWeightsAndCdfValues(N, mode=mode, eps=eps)
    
    elif mode=="densityFocused":
        weights = 1/N*np.ones((N))
        cdfValues = np.linspace(0,1,N)
        return weights, cdfValues
        
    elif mode=="quantileFocused":
        return getRepresentativeStandardNormalWeightsAndCdfValues(N, mode=mode, eps=eps)
    
    else:
        return getRepresentativeStandardNormalWeightsAndCdfValues(N, mode=mode, eps=eps)
    
if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def sample_bimodal_gaussian(n_samples=1000, mix_weights=(0.4, 0.6),
                                means=(-2, 3), stds=(0.7, 1.2)):
        comps = np.random.choice([0, 1], size=n_samples, p=mix_weights)
        data = np.where(
            comps == 0,
            np.random.normal(loc=means[0], scale=stds[0], size=n_samples),
            np.random.normal(loc=means[1], scale=stds[1], size=n_samples),
        )
        return data
    
    np.random.seed(2025)
    # 1. Generate a bimodal Gaussian mixture dataset
    N = 10000
    data = sample_bimodal_gaussian(n_samples=N,
                                   mix_weights=(0.2, 0.8),
                                   means=(-2, 3),
                                   stds=(0.5, 1.0))
    weights = np.ones_like(data)  # uniform empirical weights
    
    data2 = sample_bimodal_gaussian(n_samples=N//2,
                                   mix_weights=(0.8, 0.2),
                                   means=(-3, 2),
                                   stds=(0.5, 1.0))
    weights2 = np.ones_like(data2)  # uniform empirical weights
    
    dataSum = np.concatenate([data, data2])
    weightsSum = np.concatenate([weights, weights2])

    # 2. Fit the empirical distribution
    nApprox = 100
    mode = "densityFocused"
    
    empDist = EmpDist(data, weights=weights, pdfMethod="kde", pdfPoints=None, bw_method=0.1)
    empDigest = EmpDigest(data, weights=weights, nApprox=nApprox, mode=mode, eps=1e-2, pdfMethod="kde", pdfPoints=None, bw_method=0.1)
    
    empDist2 = EmpDist(data2, weights=weights2, pdfMethod="kde", pdfPoints=None, bw_method=0.1)
    empDigest2 = EmpDigest(data2, weights=weights2, nApprox=nApprox, mode=mode, eps=1e-2, pdfMethod="kde", pdfPoints=None, bw_method=0.1)

    # 3. Plot histogram + estimated PDF
    x_grid = np.linspace(data.min() - 1, data.max() + 1, 1000)
    pdf_vals = empDist.pdf(x_grid)
    pdf_vals_digest = empDigest.pdf(x_grid)

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=40, density=True, alpha=0.5, color="gray", label="Original data")
    plt.plot(x_grid, pdf_vals, 'r-', linewidth=2, label="Estimated PDF from EmpDist")
    plt.plot(x_grid, pdf_vals_digest, 'b-', linewidth=2, label="Estimated PDF from EmpDigest")
    plt.title("Original Histogram with Estimated PDF Overlay")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # 4. Plot CDF and inverse CDF (PPF)
    cdf_vals = empDist.cdf(x_grid)
    cdf_vals_digest = empDigest.cdf(x_grid)
    y_grid = np.linspace(0, 1, 1000)
    ppf_vals = empDist.ppf(y_grid)
    ppf_vals_digest = empDigest.ppf(y_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x_grid, cdf_vals, 'r-', label="CDF from EmpDist")
    axes[0].plot(x_grid, cdf_vals_digest, 'b-', label="Approximate CDF from EmpDigest")
    axes[0].set_title("Empirical vs. Approximate CDF")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("F(x)")

    axes[1].plot(y_grid, ppf_vals, 'r-', label="Inverse CDF from EmpDist")
    axes[1].plot(y_grid, ppf_vals_digest, 'b-', label="Approximate Inverse CDF from EmpDigest")
    axes[1].set_title("Empirical vs. Approximate Inverse CDF (PPF)")
    axes[1].set_xlabel("Quantile")
    axes[1].set_ylabel("x")

    plt.tight_layout()
    plt.show()

    # 5. Draw new samples from the empirical distribution
    M = 2000
    sampled = empDist.rvs(size=M)
    sampled_digest = empDigest.rvs(size=M)

    # 6. Compare histograms: original vs resampled
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=40, density=True, color="gray", alpha=0.4, label="Original Data")
    plt.hist(sampled, bins=40, density=True, color="r", alpha=0.4, label="EmpDist samples")
    plt.hist(sampled_digest, bins=40, density=True, color="b", alpha=0.4, label="EmpDigest samples")
    plt.title("Comparison of Original, Empricial and Approximate Histograms")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    
    # 6. add two empirical distribution and digest objects
    empDist_merged = empDist+empDist2
    empDigest_merged = empDigest+empDigest2
    empDigestDist_merged = empDigest+empDist2 # it's even possible to merge a digest and a dist object (usually leads to more precise results!)
    
    x_grid2 = np.linspace(dataSum.min() - 1, dataSum.max() + 1, 1000)
    pdf_vals2 = empDist_merged.pdf(x_grid)
    pdf_vals2_digest = empDigest_merged.pdf(x_grid)
    pdf_vals2_digestdist = empDigestDist_merged.pdf(x_grid)

    plt.figure(figsize=(8, 4))
    plt.hist(dataSum, bins=40, density=True, alpha=0.5, color="gray", label="Original Merged Data")
    plt.plot(x_grid, pdf_vals2, 'r-', linewidth=2, label=f"Estimated PDF from EmpDist ({len(empDist_merged)} Data Points)")
    plt.plot(x_grid, pdf_vals2_digest, 'b-', linewidth=2, label=f"Estimated PDF from EmpDigest ({len(empDigest_merged)} Data Points)")
    plt.plot(x_grid, pdf_vals2_digestdist, 'c-', linewidth=2, label=f"Estimated PDF from EmpDigest merged with EmpDist ({len(empDigestDist_merged)} Data Points)")
    plt.title("Original Histogram with Estimated PDF Overlay of Merged Data")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    
    # 7. compare some quantiles
    print(f"Median (reference: {empDist.icdf(0.5)}) of empDigest:", empDigest.icdf(0.5))
    print(f"Median2 (reference: {empDist_merged.icdf(0.5)}) of empDigest:", empDigest_merged.icdf(0.5))
    print(f"Median2 (reference: {empDist_merged.icdf(0.5)}) of empDigestDist:", empDigestDist_merged.icdf(0.5))
    
    print(f"0p01 (reference: {empDist.icdf(0.01)}) of empDigest:", empDigest.icdf(0.01))
    print(f"0p01 merged (reference: {empDist_merged.icdf(0.01)}) of empDigest:", empDigest_merged.icdf(0.01))
    print(f"0p01 merged (reference: {empDist_merged.icdf(0.01)}) of empDigestDist:", empDigestDist_merged.icdf(0.01))
    
    print(f"0p99 (reference: {empDist.icdf(0.99)}) of empDigest:", empDigest.icdf(0.99))
    print(f"0p99 merged (reference: {empDist_merged.icdf(0.99)}) of empDigest:", empDigest_merged.icdf(0.99))
    print(f"0p99 merged (reference: {empDist_merged.icdf(0.99)}) of empDigestDist:", empDigestDist_merged.icdf(0.99))