import numpy as np

from rasterio.transform import AffineTransformer

from scipy.interpolate import CubicSpline
from scipy import ndimage, stats
from scipy.signal import wiener

from shapely import Point, LineString, make_valid

from skimage.feature import canny
from skimage.morphology import remove_small_holes

from plantcv import plantcv as pcv


def spearman_per_pixel(a, b, min_valid=3):
    """
    a, b : 3‑D arrays (T, R, C)
    Returns:
        rho  : (R, C) array of Spearman ρ
        pval : (R, C) array of two‑tailed p‑values
    """
    # ------------------------------------------------------------------
    # 1. Rank the time series along the *time* axis
    # ------------------------------------------------------------------
    # argsort twice gives ranks (0‑based)
    rnk_a = np.argsort(np.argsort(a, axis=0), axis=0)
    rnk_b = np.argsort(np.argsort(b, axis=0), axis=0)

    # 2. Build a mask of *finite* pairs
    mask = np.isfinite(a) & np.isfinite(b)      # shape: (T, R, C)

    # 3. Count valid points per pixel
    n_valid = mask.sum(axis=0)                  # shape: (R, C)

    # 4. Masked means of the ranks
    #    Use np.nan_to_num after multiplying by mask to avoid NaNs
    sum_a = (rnk_a * mask).sum(axis=0)          # shape: (R, C)
    sum_b = (rnk_b * mask).sum(axis=0)

    # Avoid division by zero for pixels with < min_valid points
    valid_mask = n_valid >= min_valid

    mean_a = np.empty_like(n_valid, dtype=float)
    mean_b = np.empty_like(n_valid, dtype=float)
    mean_a[valid_mask] = sum_a[valid_mask] / n_valid[valid_mask]
    mean_b[valid_mask] = sum_b[valid_mask] / n_valid[valid_mask]

    # 5. Covariance per pixel
    #    (x-mean_x)*(y-mean_y) * mask, summed over time
    cov = ((rnk_a - mean_a) * (rnk_b - mean_b) * mask).sum(axis=0)

    # 6. Standard deviations per pixel
    var_a = ((rnk_a - mean_a)**2 * mask).sum(axis=0)
    var_b = ((rnk_b - mean_b)**2 * mask).sum(axis=0)

    # 7. Final correlation (handle division by zero)
    rho = np.empty_like(cov, dtype=np.float32)
    rho[:] = np.nan
    denom = np.sqrt(var_a * var_b)
    ok = (denom > 0) & valid_mask
    rho[ok] = cov[ok] / denom[ok]

    # 8. Two‑tailed p‑value (t‑distribution)
    #    t = ρ * sqrt((n-2)/(1-ρ²))
    #    p = 2 * tcdf(-|t|, df=n-2)
    df = n_valid - 2
    t = rho * np.sqrt(df / (1 - rho**2))
    pval = np.empty_like(rho, dtype=np.float32)
    pval[:] = np.nan
    pval[ok] = 2 * stats.t.cdf(-np.abs(t[ok]), df=df[ok])

    # 9. Return
    return rho, pval

def reorder_points(start_point: Point, remaining_points: list) -> list:
    """Reorders points in line (nearest neighbours)

    Parameters
    ----------
    start_point : Point
        on of the ends of the line segment
    remaining_points : list
        rest of the points in the line segment

    Returns
    -------
    list
        reordered points
    """    
    points_reordered = []
    points_reordered.append(start_point)
    index = 1
    while len(remaining_points) != 0:
        current_point = points_reordered[index - 1]
        distances = [np.sqrt(np.square(current_point[0] - point[0]) + np.square(current_point[1] - point[1])) for point in remaining_points]
        points_reordered.append(remaining_points[np.argmin(distances)])
        remaining_points.remove(points_reordered[index])
        index = index + 1
    return points_reordered


def raster_to_lines(segments, segment_ends, transform) -> list:
    """Converts skeletonized raster to line strings, handles loops and rings too

    Parameters
    ----------
    segments : dict
        skeletonized prediction segments, output of get_segments
    segment_ends : dict
        pixels that correspond to end of the segments, output of get_segment_ends
    transform : Affine
        affine transform matrix of prediction tile

    Returns
    -------
    list
       line strings
    """    
    transformer = AffineTransformer(transform)
    line_segments = []
    for id, segment in segments.items():
        if id != 0 and len(segment) > 1:
            points = []
            for pixel in segment:
                if (pixel != segment_ends[id]).all():
                    points.append(tuple(pixel))
            points = reorder_points(segment_ends[id], points)
            if len(points) > 1:
                points = [Point(transformer.xy(pixel[0], pixel[1])) for pixel in points]
                line_segments.append(make_valid(LineString(points)))
    return line_segments

def get_segments(filled_mask: np.ndarray) -> dict:
    """Saves the pixels of segments of prediction skeleton to dict, with dict keys corresponding to segment id

    Parameters
    ----------
    filled_mask : np.ndarray
        Filled prediction segment, output from pcv.morphology.fill_segments

    Returns
    -------
    dict
        segments and corresponding pixels
    """    
    segments = {}
    for id in np.unique(filled_mask):
        segments[id] = np.argwhere(filled_mask == id)
    del segments[0]
    return segments

def find_segment_ends(segments: dict, skeleton: np.ndarray) -> dict:
    """Finds one tip of each line segment in skeletonized skeleton. Does not handle closed rings or loops

    Parameters
    ----------
    segments : dict
        pixel positions of the individual segments, output from the function get_segments
    skeleton : np.ndarray
        skeletonized predictions

    Returns
    -------
    dict
        The pixel corresponding to one end of a line segment
    """    
    segment_ends = {}
    max_pixel = skeleton.shape[0] - 1
    for id, segment in segments.items():
        segment_ends[id] = segment[0]
        for pixel in segment:
            if (pixel[0] == 0) or (pixel[1] == 0) or (pixel[0] == max_pixel) or (pixel[1] == max_pixel):
                segment_ends[id] = pixel
            else:
                neighbours = skeleton[pixel[0] - 1:pixel[0] + 2, pixel[1] - 1:pixel[1] + 2]
                if np.sum(neighbours) == 2:
                    segment_ends[id] = pixel
    return segment_ends

def smoothen_line(vector: list, points: np.int16 = 1000, sigma: np.float16 = 10) -> list:
    """Interpolates line coordinates and applies 1D gaussian smoothing

    Parameters
    ----------
    vector : list
        linestrings
    points : np.int16, optional
        total no. of coordinates after interpolation, by default 1000
    sigma : np.float16, optional
        kernel for 1D gaussian, by default 10

    Returns
    -------
    list
        smoothened linestrings
    """    
    interpolated_lines = []
    for line in vector:
        x, y = line.xy
        x = list(x)
        y = list(y)
        values = np.arange(len(x))

        interp_x = CubicSpline(x = values, y = x, bc_type = 'natural')
        interp_y = CubicSpline(x = values, y = y, bc_type = 'natural')
        interpolate_values = np.linspace(values[0], values[-1], points)
        interpolated_x = interp_x(interpolate_values)
        interpolated_y = interp_y(interpolate_values)

        smooth_x = ndimage.gaussian_filter1d(interpolated_x, sigma)
        smooth_y = ndimage.gaussian_filter1d(interpolated_y, sigma)
        interp_x = CubicSpline(x = interpolate_values, y = smooth_x, bc_type = 'natural')
        interp_y = CubicSpline(x = interpolate_values, y = smooth_y, bc_type = 'natural')
        interpolated_x = interp_x(values)
        interpolated_y = interp_y(values)
        interpolated_coordinates = [Point(x, y) for x, y in zip(interpolated_x, interpolated_y)]
        interpolated_line = LineString(interpolated_coordinates)
        interpolated_lines.append(interpolated_line)
    return interpolated_lines

def vectorize_gl(gl_arr, transform):
    pruned, _, segment_objects = pcv.morphology.prune(skel_img = gl_arr, size = 0)
    filled = pcv.morphology.fill_segments(mask = pruned, objects = list(segment_objects))
    segments = get_segments(filled_mask = filled)
    segment_ends = find_segment_ends(segments = segments, skeleton = pruned)
    vectors = raster_to_lines(segments = segments, segment_ends = segment_ends, transform = transform)
    vectors = smoothen_line(vector = vectors, sigma = 1)
    return vectors

def lowpass_filter(image, filter_order = 2, cutoff_freq = 250):
    nan_mask = np.isnan(image)
    image[nan_mask] = 0
    image_fft = np.fft.fft2(image)
    
    rows = np.arange(image.shape[0])
    cols = np.arange(image.shape[1])

    rows_center = image.shape[0] / 2
    cols_center = image.shape[1] / 2

    rows[rows > rows_center] = rows[rows > rows_center] - image.shape[0]
    cols[cols > cols_center] = cols[cols > cols_center] - image.shape[1]

    cols_grid, rows_grid = np.meshgrid(cols, rows)

    euclidean_dist = np.sqrt(rows_grid**2 + cols_grid**2)
    filter_mask = 1/(1 + (euclidean_dist/cutoff_freq)**(2 * filter_order))

    convolved_image = np.multiply(filter_mask, image_fft)
    filtered_image = np.real(np.fft.ifft2(convolved_image))
    filtered_image = wiener(im = filtered_image, mysize = (12, 12))
    filtered_image[nan_mask] = np.nan
    return filtered_image

def get_tmoc_gl(correlation, pval, threshold):
    filtered_correlation = lowpass_filter(image = correlation)
    filtered_pval = lowpass_filter(image = pval)
    filtered_correlation = np.multiply(filtered_correlation, 1 - filtered_pval)
    thresholded_correlation = np.zeros_like(filtered_correlation)
    thresholded_correlation[filtered_correlation >= threshold] = 1
    thresholded_correlation = remove_small_holes(thresholded_correlation.astype('bool'), area_threshold = 64)
    grounding_line = canny(thresholded_correlation, sigma = 1).astype(np.uint8)
    return grounding_line

