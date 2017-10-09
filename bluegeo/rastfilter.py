from raster import *
import util
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.interpolate import griddata


class FilterError(Exception):
    pass


def eval_op(a, nd, size, func, **kwargs):
    """
    Factory for all numpy statistical function filters
    :param a: input 2-D array
    :param nd: No data value
    :param size: window size for filter
    :param func: numpy stats function
    :param kwargs: additional args for functions
    :return: 2-D output array
    """
    ndMask = binary_dilation(a == nd, structure=numpy.ones(shape=size, dtype='bool'))[1:-1, 1:-1]
    percentile = kwargs.get('percentile', None)
    if percentile is not None:
        A = func(util.stride_hood(a, size), percentile=percentile, axis=(3, 2))
    else:
        A = func(util.stride_hood(a, size), axis=(3, 2))
    A[ndMask] = a[1:-1, 1:-1][ndMask]
    return A


def min_filter(input_raster, size=(3, 3)):
    """
    Perform a minimum filter
    :param size: Window size
    :return: Raster with local minima
    """
    input_raster = raster(input_raster)
    out_raster = input_raster.full(input_raster.nodata)
    if input_raster.useChunks:
        for a, s in input_raster.iterchunks(expand=size):
            s_ = util.truncate_slice(s, size)
            out_raster[s_] = eval_op(a, input_raster.nodata, size, numpy.min)
    else:
        # Calculate over all data
        out_raster[1:-1, 1:-1] = eval_op(input_raster.array, input_raster.nodata, size, numpy.min)

    return out_raster


def max_filter(input_raster, size=(3, 3)):
    """
    Perform a maximum filter
    :param size: Window size
    :return: Raster with local maxima values
    """
    input_raster = raster(input_raster)
    out_raster = input_raster.full(input_raster.nodata)
    if input_raster.useChunks:
        for a, s in input_raster.iterchunks(expand=size):
            s_ = util.truncate_slice(s, size)
            out_raster[s_] = eval_op(a, input_raster.nodata, size, numpy.max)
    else:
        # Calculate over all data
        out_raster[1:-1, 1:-1] = eval_op(input_raster.array, input_raster.nodata, size, numpy.max)

    return out_raster


def mean_filter(input_raster, size=(3, 3)):
    """
    Perform a mean filter
    :param size: Window size
    :return: Raster with local mean values
    """
    # Allocate output
    input_raster = raster(input_raster)
    out_raster = input_raster.full(input_raster.nodata)
    if input_raster.useChunks:
        for a, s in input_raster.iterchunks(expand=size):
            s_ = util.truncate_slice(s, size)
            out_raster[s_] = eval_op(a, input_raster.nodata, size, numpy.mean)
    else:
        # Calculate over all data
        out_raster[1:-1, 1:-1] = eval_op(input_raster.array, input_raster.nodata, size, numpy.mean)

    return out_raster


def std_filter(input_raster, size=(3, 3)):
    """
    Perform a standard deviation filter
    :param size: Window size
    :return: Raster with local standard deviation values
    """
    input_raster = raster(input_raster)
    out_raster = input_raster.full(input_raster.nodata)
    if input_raster.useChunks:
        for a, s in input_raster.iterchunks(expand=size):
            s_ = util.truncate_slice(s, size)
            out_raster[s_] = eval_op(a, input_raster.nodata, size, numpy.std)
    else:
        # Calculate over all data
        out_raster[1:-1, 1:-1] = eval_op(input_raster.array, input_raster.nodata, size, numpy.std)

    return out_raster


def var_filter(input_raster, size=(3, 3)):
    """
    Perform a variance filter
    :param size: Window size
    :return: Raster with local variance values
    """
    input_raster = raster(input_raster)
    out_raster = input_raster.full(input_raster.nodata)
    if input_raster.useChunks:
        for a, s in input_raster.iterchunks(expand=size):
            s_ = util.truncate_slice(s, size)
            out_raster[s_] = eval_op(a, input_raster.nodata, size, numpy.var)
    else:
        # Calculate over all data
        out_raster[1:-1, 1:-1] = eval_op(input_raster.array, input_raster.nodata, size, numpy.var)

    return out_raster


def median_filter(input_raster, size=(3, 3)):
    """
    Perform a median filter
    :param size: Window size
    :return: Raster with local median values
    """
    input_raster = raster(input_raster)
    out_raster = input_raster.full(input_raster.nodata)
    if input_raster.useChunks:
        for a, s in input_raster.iterchunks(expand=size):
            s_ = util.truncate_slice(s, size)
            out_raster[s_] = eval_op(a, input_raster.nodata, size, numpy.median)
    else:
        # Calculate over all data
        out_raster[1:-1, 1:-1] = eval_op(input_raster.array, input_raster.nodata, size, numpy.median)

    return out_raster


def percentile_filter(input_raster, percentile=25, size=(3, 3)):
    """
    Perform a median filter
    :param size: Window size
    :return: Raster with local median values
    """

    input_raster = raster(input_raster)
    out_raster = input_raster.full(input_raster.nodata)
    if input_raster.useChunks:
        for a, s in input_raster.iterchunks(expand=size):
            s_ = util.truncate_slice(s, size)
            out_raster[s_] = eval_op(a, input_raster.nodata, size, numpy.percentile, percentile=percentile)
    else:
        # Calculate over all data
        out_raster[1:-1, 1:-1] = eval_op(input_raster.array, input_raster.nodata, size, numpy.percentile,
                                         percentile=percentile)

    return out_raster


def most_common(input_raster, size=(3, 3)):
    """
    Perform a mode filter
    :param size: Window size
    :return: Raster with most frequent local value
    """
    # Allocate output
    input_raster = raster(input_raster)
    mode_raster = input_raster.empty()
    if input_raster.useChunks:
        # Iterate chunks and calculate mode (memory-intensive, so don't fill cache)
        for a, s in input_raster.iterchunks(expand=size, fill_cache=False):
            s_ = util.truncate_slice(s, size)
            mode_raster[s_] = util.mode(util.window_on_last_axis(a, size), 2)[0]
    else:
        # Calculate over all data
        mode_raster[1:-1, 1:-1] = util.mode(util.window_on_last_axis(input_raster.array, size), 2)[0]

    return mode_raster


def gaussian(input_raster, sigma):
    """
    Perform a gaussian filter with a specific standard deviation
    :param input_raster: Input raster
    :param sigma: Standard deviation
    :return: raster instance
    """
    # Allocate output
    input_raster = raster(input_raster)
    gauss = input_raster.empty()

    # Create mask from nodata values
    m = input_raster.array == input_raster.nodata

    # Regions of no data must be interpolated first
    interp_rast = interpolate_nodata(input_raster)

    # Perform filter
    a = gaussian_filter(interp_rast.array, sigma)
    a[m] = gauss.nodata

    gauss[:] = a
    return gauss


def interpolate_nodata(input_raster, method='nearest'):
    """
    Fill no data values with interpolated values from the edge of valid data
    :param input_raster: input raster
    :param method: interpolation method
    :return: raster instance
    """
    inrast = raster(input_raster)

    # Check if no data values exist
    a = inrast.array
    xi = a == inrast.nodata
    if xi.sum() == 0:
        return inrast

    # Interpolate values
    points = binary_dilation(xi, structure=numpy.ones(shape=(3, 3), dtype='bool')) & ~xi
    values = a[points]
    points = numpy.where(points)
    points = numpy.vstack([points[0] * inrast.csy, points[1] * inrast.csx]).T
    xi = numpy.where(xi)
    if method != 'nearest':
        # Corners of raster must have data to ensure convex hull encompasses entire raster
        index = map(numpy.array, ([0, 0, a.shape[0] - 1, a.shape[0] - 1], [0, a.shape[1] - 1, 0, a.shape[1] - 1]))
        corner_nodata = a[index] == inrast.nodata
        if corner_nodata.sum() != 0:
            index = (index[0][corner_nodata], index[1][corner_nodata])
            points_append = (index[0] * inrast.csy, index[1] * inrast.csx)
            corners = griddata(points, values, points_append, method='nearest')
            values = numpy.append(values, corners)
            points = numpy.append(points, numpy.vstack(points_append).T, axis=0)
            a[index] = corners
            xi = a == inrast.nodata
            xi = numpy.where(xi)

    a[xi] = griddata(points, values, (xi[0] * inrast.csy, xi[1] * inrast.csx), method)

    # Return output
    outrast = inrast.empty()
    outrast[:] = a
    return outrast


def dilate(input_raster, dilate_value=1, iterations=1):
    """
    Perform a region dilation
    :param input_raster: Input raster
    :param dilate_value: Raster value to dilate
    :return: Raster instance
    """
    pass


def edge_detect(input_Raster, detection_value=1):
    """
    Edge detection
    :param input_Raster:
    :param detection_value:
    :return: raster instance
    """
    dilate()
