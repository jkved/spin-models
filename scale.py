import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import convolve2d


def scale_data(data, scale):
    """Change the scale of given data.

    Input:
        data:
            Two-dimensional matrix of numbers in [0, 1].
            It is assumed that these numbers represent
            fractions of people with certain
            characteristic.
        scale:
            Scale of newly made geographical units. Units
            are assumed to take square shape (i.e., their
            size is [scale x scale]).
    Output:
        scaled_data:
            Two-dimensional matrix which contains
            fractions of people with certain
            characteristic when viewed from a larger
            scale.

    Important:
        Note that the scaled_data will be of smaller size
        as geographical units are not allowed to overlap.

        Note that the border regions are taken into
        account even if they from incomplete (contain less
        than scale*scale smallest units).
    """
    out_x = np.ceil(data.shape[0] / scale).astype(int)
    out_y = np.ceil(data.shape[1] / scale).astype(int)
    out = np.zeros((out_x, out_y))
    for ix, x in enumerate(np.arange(0, data.shape[0], scale)):
        for iy, y in enumerate(np.arange(0, data.shape[1], scale)):
            out[ix, iy] = np.mean(data[x : x + scale, y : y + scale])
    return out


def auto_scale(data, analysis_fn=np.std):
    """Perform analysis at automatically selected scales.

    Input:
        data:
            Two-dimensional matrix of numbers in [0, 1].
            It is assumed that these numbers represent
            fractions of people with certain
            characteristic.
        analysis_fn:
            The function used to perform analysis.

    Output:
        unit_sizes:
            Unit sizes at which analysis was performed.
        values:
            Values obtained by analysis_fn when analyzing
            scaled data. Normalized by the first value.
    """
    scales = (2 ** np.arange(0, np.floor(np.log2(np.min(data.shape))))).astype(int)
    vals = np.array([analysis_fn(scale_data(data, scale)) for scale in scales])
    if vals[0] != 0:
        rel_vals = vals / vals[0]
    else:
        rel_vals = np.zeros(vals.shape)
        rel_vals[0] = 1
    unit_sizes = scales**2
    return (unit_sizes, rel_vals)


def calculate_scaling_index(unit_sizes, data_vals):
    """Calculate scaling index in comparison to some baseline.

    Input:
        unit_sizes:
            Unit sizes at which analysis as performed.
        data_vals:
            Analysis values obtained for the data.

    Output:
        index:
            Value between -1 and 1. The closer values is
            to 0, the less difference between the
            baseline model and data.
    """
    min_vals = np.zeros(unit_sizes.shape)
    min_vals[0] = 1
    minimal_area = trapezoid(min_vals, unit_sizes)
    maximal_area = unit_sizes[-1] - 1
    baseline_area = trapezoid(1 / np.sqrt(unit_sizes), unit_sizes)
    data_area = trapezoid(data_vals, unit_sizes)
    index = 0
    if data_area < baseline_area:
        # In this case data_area will be in [minimal_area,
        # baseline_area] interval. We would like that
        # data_area=minimal_area would imply index=-1, while
        # data_area=baseline_area would imply index=0.
        index = (data_area - minimal_area) / (baseline_area - minimal_area) - 1
    else:
        # In this case data_area will be in [baseline_area,
        # unit_sizes[-1]-1] interval. We would like that
        # data_area=baseline_area would imply index=0, while
        # data_area=maximal_area would imply index=1.
        index = (data_area - baseline_area) / (maximal_area - baseline_area)
    return index


def scale_data_convolve(data, scale):
    """Change the scale of given data using convolution.

    Input:
        data:
            Two-dimensional matrix of numbers in [0, 1].
            It is assumed that these numbers represent
            fractions of people with certain
            characteristic.
        scale:
            Scale of newly made geographical units. Units
            are assumed to take square shape (i.e., their
            size is [scale x scale]).
    Output:
        scaled_data:
            Two-dimensional matrix which contains
            fractions of people with certain
            characteristic when viewed from a larger
            scale.

    Important:
        Note that the newly made geographical units are
        allowed to overlap.
    """
    averaging_kernel = np.ones((scale, scale)) / (scale**2)
    scaled_data = convolve2d(data, averaging_kernel, mode="same", boundary="wrap")
    return scaled_data
