from raster import *
import util


class FilterError(Exception):
    pass


def min_filter(input_raster, size=(3, 3)):
    """
    Perform a minimum filter
    :param size: Window size
    :return: Raster with local minima
    """
    def eval_min(a, nd):
        # Pad with the maximum value
        ndMask = a == nd
        aMax = a[~ndMask].max()
        a[ndMask] = aMax
        minA = numpy.min(util.stride_hood(a, size, constant_values=aMax), axis=(3, 2))
        minA[ndMask] = nd
        return minA

    # Allocate output
    input_raster = raster(input_raster)
    with input_raster.empty() as min_raster:
        if input_raster.useChunks:
            for a, s in input_raster.iterchunks(expand=size):
                min_raster[s] = eval_min(a, input_raster.nodata)
        else:
            # Calculate over all data
            min_raster[:] = eval_min(input_raster.array, input_raster.nodata)

    return min_raster


def max_filter(input_raster, size=(3, 3)):
    """
    Perform a maximum filter
    :param size: Window size
    :return: Raster with local maxima values
    """
    def eval_max(a, nd):
        # Pad with the minimum value
        ndMask = a == nd
        aMin = a[~ndMask].min()
        a[ndMask] = aMin
        maxA = numpy.max(util.stride_hood(a, size, constant_values=aMin), axis=(3, 2))
        maxA[ndMask] = nd
        return maxA

    # Allocate output
    input_raster = raster(input_raster)
    with input_raster.empty() as max_raster:
        if input_raster.useChunks:
            for a, s in input_raster.iterchunks(expand=size):
                max_raster[s] = eval_max(a, input_raster.nodata)
        else:
            # Calculate over all data
            max_raster[:] = eval_max(input_raster.array, input_raster.nodata)

    return max_raster


def mean_filter(input_raster, size=(3, 3)):
    """
    Perform a mean filter
    :param size: Window size
    :return: Raster with local mean values
    """
    def eval_mean(a, nd):
        ndMask = a == nd
        meanA = numpy.mean(util.stride_hood(a, size, edge_mode='edge'), axis=(3, 2))
        meanA[ndMask] = nd
        return meanA

    # Allocate output
    input_raster = raster(input_raster)
    with input_raster.empty() as mean_raster:
        if input_raster.useChunks:
            for a, s in input_raster.iterchunks(expand=size):
                mean_raster[s] = eval_mean(a, input_raster.nodata)
        else:
            # Calculate over all data
            mean_raster[:] = eval_mean(input_raster.array, input_raster.nodata)

    return mean_raster


def std_filter(input_raster, size=(3, 3)):
    """
    Perform a standard deviation filter
    :param size: Window size
    :return: Raster with local standard deviation values
    """
    def eval_std(a, nd):
        ndMask = a == nd
        stdA = numpy.std(util.stride_hood(a, size, edge_mode='edge'), axis=(3, 2))
        stdA[ndMask] = nd
        return stdA

    # Allocate output
    input_raster = raster(input_raster)
    with input_raster.empty() as std_raster:
        if input_raster.useChunks:
            for a, s in input_raster.iterchunks(expand=size):
                std_raster[s] = eval_std(a, input_raster.nodata)
        else:
            # Calculate over all data
            std_raster[:] = eval_std(input_raster.array, input_raster.nodata)

    return std_raster


def var_filter(input_raster, size=(3, 3)):
    """
    Perform a variance filter
    :param size: Window size
    :return: Raster with local variance values
    """
    def eval_var(a, nd):
        ndMask = a == nd
        varA = numpy.var(util.stride_hood(a, size, edge_mode='edge'), axis=(3, 2))
        varA[ndMask] = nd
        return varA

    # Allocate output
    input_raster = raster(input_raster)
    with input_raster.empty() as var_raster:
        if input_raster.useChunks:
            for a, s in input_raster.iterchunks(expand=size):
                var_raster[s] = eval_var(a, input_raster.nodata)
        else:
            # Calculate over all data
            var_raster[:] = eval_var(input_raster.array, input_raster.nodata)

    return var_raster


def median_filter(input_raster, size=(3, 3)):
    """
    Perform a median filter
    :param size: Window size
    :return: Raster with local median values
    """
    def eval_med(a, nd):
        ndMask = a == nd
        medA = numpy.median(util.stride_hood(a, size, edge_mode='edge'), axis=(3, 2))
        medA[ndMask] = nd
        return medA

    # Allocate output
    input_raster = raster(input_raster)
    with input_raster.empty() as med_raster:
        if input_raster.useChunks:
            for a, s in input_raster.iterchunks(expand=size):
                med_raster[s] = eval_med(a, input_raster.nodata)
        else:
            # Calculate over all data
            med_raster[:] = eval_med(input_raster.array, input_raster.nodata)

    return med_raster


def percentile_filter(input_raster, percentile=25, size=(3, 3)):
    """
        Perform a median filter
        :param size: Window size
        :return: Raster with local median values
        """

    def eval_perc(a, nd):
        ndMask = a == nd
        percA = numpy.percentile(util.stride_hood(a, size, edge_mode='edge'), percentile, axis=(3, 2))
        percA[ndMask] = nd
        return percA

    # Allocate output
    input_raster = raster(input_raster)
    with input_raster.empty() as perc_raster:
        if input_raster.useChunks:
            for a, s in input_raster.iterchunks(expand=size):
                perc_raster[s] = eval_perc(a, input_raster.nodata)
        else:
            # Calculate over all data
            perc_raster[:] = eval_perc(input_raster.array, input_raster.nodata)

    return perc_raster


def most_common(input_raster, size=(3, 3)):
    """
    Perform a mode filter
    :param size: Window size
    :return: Raster with most frequent local value
    """
    # Allocate output
    input_raster = raster(input_raster)
    with input_raster.empty() as mode_raster:
        if input_raster.useChunks:
            # Iterate chunks and calculate mode (memory-intensive, so don't fill cache)
            for a, s in input_raster.iterchunks(expand=size, fill_cache=False):
                s_ = util.truncate_slice(s, size)
                mode_raster[s_] = util.mode(util.window_on_last_axis(a, size), 2)[0]
        else:
            # Calculate over all data
            mode_raster[1:-1, 1:-1] = util.mode(util.window_on_last_axis(input_raster.array, size), 2)[0]

    return mode_raster


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
