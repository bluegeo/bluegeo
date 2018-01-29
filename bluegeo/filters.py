from spatial import *
import util
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.interpolate import griddata
from numba.decorators import jit


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
        A = func(util.stride_hood(a, size), percentile, axis=(3, 2))
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
    input_raster = Raster(input_raster)
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
    input_raster = Raster(input_raster)
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
    input_raster = Raster(input_raster)
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
    input_raster = Raster(input_raster)
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
    input_raster = Raster(input_raster)
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
    input_raster = Raster(input_raster)
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

    input_raster = Raster(input_raster)
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
    input_raster = Raster(input_raster)
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
    :param input_raster: Input Raster
    :param sigma: Standard deviation
    :return: Raster instance
    """
    # Allocate output
    input_raster = Raster(input_raster)
    gauss = input_raster.astype('float32')

    # Create mask from nodata values
    m = input_raster.array == input_raster.nodata

    # Dilate no data regions so sample does not use no data
    distance = int(4 * sigma + 0.5) * max([input_raster.csx, input_raster.csy])
    interp_rast = extrapolate_buffer(input_raster, distance)

    # Perform filter
    a = gaussian_filter(interp_rast.array, sigma)
    a[m] = gauss.nodata

    gauss[:] = a
    return gauss


def extrapolate_buffer(input_raster, distance, method='nearest'):
    """
    Extrapolate outside of data regions over a specified distance
    :param input_raster: Input Raster to be extrapolated
    :param distance: (double) Distance of buffer (will be snapped to a fixed number of grid cells)
    :param method: (str) interpolation method (nearest, bilinear, cubic).  Only the nearest method will ensure values are added outside the convex hull
    :return: Raster instance
    """
    # Read input and find the number of cells to dilate
    input_raster = Raster(input_raster)
    iterations = int(numpy.ceil(distance / min([input_raster.csx, input_raster.csy])))
    a = input_raster.array

    # Create mask for no data values
    m = (a == input_raster.nodata) | numpy.isnan(a)
    # Coordinates to interpolate data
    xi = numpy.where(m & binary_dilation(~m, numpy.ones((3, 3)), iterations))
    # Coordinates to be used in interpolation
    points = numpy.where(binary_dilation(m, numpy.ones((3, 3))) & ~m)
    # Data to be used in interpolation
    values = a[points]
    # Change configuration of interpolation points
    points = numpy.vstack([points[0] * input_raster.csy, points[1] * input_raster.csx]).T
    # Complete and insert interpolation
    a[xi] = griddata(points, values, (xi[0] * input_raster.csy, xi[1] * input_raster.csx), method)

    # Prepare and return output
    output_raster = input_raster.empty()
    output_raster[:] = a
    return output_raster


def interpolate_nodata(input_raster, method='nearest'):
    """
    Fill no data values with interpolated values from the edge of valid data
    :param input_raster: input Raster
    :param method: interpolation method
    :return: Raster instance
    """
    inrast = Raster(input_raster)

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
        # Corners of Raster must have data to ensure convex hull encompasses entire Raster
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


def interpolate_mask(input_raster, mask_raster, method='nearest'):
    """
    Interpolate data from the Raster to the missing values in the mask.
    Note, if using cubic or linear, values will be limited to the coverage of the convex hull
    :param input_raster: Raster to grid across mask
    :param mask_raster: mask to grid values
    :param method: interpolation method ('nearest', 'idw', 'linear')
    :return: Raster instance
    """

    def inverse_distance(pointGrid, xGrid, values):
        """TODO: Reduce boilerplate. This method also exists in bluegeo.terrain.align"""
        @jit(nopython=True, nogil=True)
        def idw(args):
            points, xi, grad, output, mask = args
            i_shape = xi.shape[0]
            point_shape = points.shape[0]
            for i in range(i_shape):
                num = 0.0
                denom = 0.0
                for j in range(point_shape):
                    w = 1 / numpy.sqrt(
                        ((points[j, 0] - xi[i, 0]) ** 2) + ((points[j, 1] - xi[i, 1]) ** 2)
                    ) ** 2
                    denom += w
                    num += w * grad[j]
                output[i] = num / denom
            return output, mask

        # Compute chunk size from memory specification and neighbours
        from multiprocessing import Pool, cpu_count
        chunkSize = int(round(xGrid.shape[0] / (cpu_count() * 4)))
        if chunkSize < 1:
            chunkSize = 1
        chunkRange = range(0, xGrid.shape[0] + chunkSize, chunkSize)

        iterator = []
        totalCalcs = 0
        for fr, to in zip(chunkRange[:-1], chunkRange[1:-1] + [xGrid.shape[0]]):
            xChunk = xGrid[fr:to]
            totalCalcs += pointGrid.shape[0] * xChunk.shape[0]
            iterator.append(
                (pointGrid, xChunk, values, numpy.zeros(shape=(to - fr,), dtype='float32'), (fr, to))
            )
        print "IDW requires {} calculations".format(totalCalcs)

        import time
        now = time.time()
        p = Pool(cpu_count())
        try:
            iterator = list(p.imap_unordered(idw, iterator))
        except Exception as e:
            import sys
            p.close()
            p.join()
            raise e, None, sys.exc_info()[2]
        else:
            p.close()
            p.join()
        print "Completed IDW interpolation in %s minutes" % (round((time.time() - now) / 60, 3))
        return iterator

    inrast = Raster(input_raster)
    a = inrast.array

    # Create a mask from mask Raster
    mask = Raster(mask_raster).match_raster(inrast)
    mask = mask.array != mask.nodata

    # Gather points for interpolation
    in_nodata = a == inrast.nodata
    xi = mask & in_nodata
    if xi.sum == 0:
        # Nothing to interpolate
        return inrast

    # Gather data values for interpolation at the edges only
    points = binary_dilation(in_nodata, numpy.ones((3, 3))) & ~in_nodata
    values = a[points]

    # Create x-y grids from masks
    points = numpy.where(points)
    xi = numpy.where(xi)

    # Interpolate using scipy griddata if method is nearest, cubic, or linear
    if method != 'idw':
        points = numpy.vstack([points[0] * inrast.csy, points[1] * inrast.csx]).T
        a[xi] = griddata(points, values, (xi[0] * inrast.csy, xi[1] * inrast.csx), method)
    else:
        # Use internal idw method- note, this is slow because it completes an entire outer product
        # Points in form ((x, y), (x, y))
        pointGrid = numpy.fliplr(
            numpy.array(util.indices_to_coords(points, inrast.top, inrast.left, inrast.csx, inrast.csy)).T
        )
        # Interpolation grid in form ((x, y), (x, y))
        xGrid = numpy.fliplr(
            numpy.array(util.indices_to_coords(xi, inrast.top, inrast.left, inrast.csx, inrast.csy)).T
        )

        iterator = inverse_distance(pointGrid, xGrid, values)

        # Add output to a using iterator generated in idw
        output = numpy.zeros(shape=xi[0].shape, dtype='float32')
        for i in iterator:
            output[i[1][0]:i[1][1]] = i[0]
        a[xi] = output

    # Create output
    outrast = inrast.empty()
    outrast[:] = a
    return outrast


def dilate(input_raster, dilate_value=1, iterations=1):
    """
    Perform a region dilation
    :param input_raster: Input Raster
    :param dilate_value: Raster value to dilate
    :return: Raster instance
    """
    pass


def normalize(input_raster):
    """
    Normalize the range of Raster data values (make range from 0 to 1)
    :param input_raster:
    :return:
    """
    min_val = rastmin(input_raster)
    return (Raster(input_raster) - min_val) / (rastmax(input_raster) - min_val)


def inverse(input_raster):
    """Return the inverse of data values"""
    inrast = Raster(input_raster)
    outrast = inrast.empty()
    a = inrast.array
    mask = a != inrast.nodata
    a_min, a_max = a[mask].min(), a[mask].max()
    m, b = numpy.linalg.solve([[a_max, 1.], [a_min, 1.]], [a_min, a_max])
    a[mask] = (a[mask] * m) + b
    outrast[:] = a
    return outrast


def convolve(input_raster, kernel):
    """
    Perform convolution using the specified kernel of weights
    :param input_raster: Raster to perform convolution
    :param kernel: numpy 2-d array of floats
    :return: Raster instance
    """
    if kernel.size == 1:
        return input_raster * numpy.squeeze(kernel)

    # Create a padded array
    inrast = Raster(input_raster)
    padding = (map(int, ((kernel.shape[0] - 1.) / 2, numpy.ceil((kernel.shape[0] - 1.) / 2))),
               map(int, ((kernel.shape[1] - 1.) / 2, numpy.ceil((kernel.shape[1] - 1.) / 2))))
    a = inrast.array
    mask = a == inrast.nodata
    a[mask] = 0
    a = numpy.pad(a.astype('float32'), padding, 'constant')

    # Perform convolution
    views = util.get_window_views(a, kernel.shape)  # Views into a over the kernel
    local_dict = util.window_local_dict(views)  # Views turned into a pointer dictionary for numexpr
    output = numpy.zeros(shape=inrast.shape, dtype='float32')  # Allocate output
    # ne.evaluate only allows 32 arrays in one expression.  Need to chunk it up.
    keys = ['a{}_{}'.format(i, j) for i in range(len(views)) for j in range(len(views[0]))]  # preserve order
    kernel_len = len(keys)
    keychunks = range(0, len(local_dict) + 31, 31)
    keychunks = zip(keychunks[:-1],
                    keychunks[1:-1] + [len(keys)])
    kernel = kernel.ravel()
    for ch in keychunks:
        new_local = {k: local_dict[k] for k in keys[ch[0]: ch[1]]}
        expression = '+'.join(['{}*{}'.format(prod_1, prod_2)
                               for prod_1, prod_2 in zip(new_local.keys(), kernel[ch[0]: ch[1]])])
        output += ne.evaluate(expression, local_dict=new_local)

    # Allocate output
    outrast = inrast.astype('float32')

    # Mask previous nodata and write to output
    output[mask] = outrast.nodata
    outrast[:] = output

    return outrast


def edge_detect(input_Raster, detection_value=1):
    """
    Edge detection
    :param input_Raster:
    :param detection_value:
    :return: Raster instance
    """
    dilate()
