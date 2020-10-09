# Utilities for bluegeo

import math
import numpy
import os
from tempfile import _get_candidate_names, gettempdir
from scipy import ndimage
from osgeo import osr


class BlueUtilError(Exception):
    pass


def generate_name(parent_path, suffix, extension):
    """Generate a unique file name"""
    if parent_path is None:
        path_dir = gettempdir()
        path_str = next(_get_candidate_names())
    else:
        path_dir = os.path.dirname(parent_path)
        path_str = os.path.basename(parent_path)


    path = ('%s_%s_%s.%s' %
            (''.join(path_str.split('.')[:-1])[:20], suffix,
             next(_get_candidate_names()), extension)
            )

    return os.path.join(path_dir, path)


def parse_projection(projection):
    """Return a wkt from some argument"""
    def raise_re():
        raise BlueUtilError('Unable to determine projection from %s' %
                            projection)
    if isinstance(projection, str):
        sr = osr.SpatialReference()
        sr.ImportFromWkt(projection)
        outwkt = sr.ExportToWkt()
    elif isinstance(projection, osr.SpatialReference):
        return projection.ExportToWkt()
    elif isinstance(projection, int):
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(projection)
        outwkt = sr.ExportToWkt()
    elif projection is None or projection == '':
        outwkt = ''
    else:
        raise_re()
    return outwkt


def compare_projections(proj1, proj2):
    osr_proj1 = osr.SpatialReference()
    osr_proj2 = osr.SpatialReference()

    osr_proj1.ImportFromWkt(parse_projection(proj1))
    osr_proj2.ImportFromWkt(parse_projection(proj2))

    return osr_proj1.IsSame(osr_proj2)


def isclose(input, values, tolerance):
    values = [(val - tolerance, val + tolerance) for val in values]
    if any([lower < input < upper for lower, upper in values]):
        return True
    else:
        return False


def transform_points(points, inproj, outproj):
    """
    Transform a list of points [(x, y), (x, y)...]
    :param points:
    :param inproj: projection of points as wkt
    :param outproj: output projection of points as wkt
    :return: projected points
    """
    if compare_projections(inproj, outproj):
        return points

    insr = osr.SpatialReference()
    insr.ImportFromWkt(parse_projection(inproj))
    outsr = osr.SpatialReference()
    outsr.ImportFromWkt(parse_projection(outproj))

    # Ensure resulting axes are still in the order x, y
    outsr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    coordTransform = osr.CoordinateTransformation(insr, outsr)
    return [coordTransform.TransformPoint(x, y)[:2] for x, y in points]


def truncate_slice(s, size):
    i, j = size
    ifr = int(math.ceil((i - 1) / 2.))
    jfr = int(math.ceil((j - 1) / 2.))
    ito = int((i - 1) / 2.)
    jto = int((j - 1) / 2.)
    s_ = s[0].start + ifr
    s__ = s[1].start + jfr
    _s = s[0].stop - ito
    __s = s[1].stop - jto
    return (slice(s_, _s), slice(s__, __s))


def get_window_views(a, size):
    '''
    Get a "shaped" list of views into "a" to retrieve
    neighbouring cells in the form of "size."
    Note: asymmetrical shapes will be propagated
        up and left.
    '''
    i_offset = (size[0] - 1) * -1
    j_offset = (size[1] - 1) * -1
    output = []
    for i in range(i_offset, 1):
        output.append([])
        _i = abs(i_offset) + i
        if i == 0:
            i = None
        for j in range(j_offset, 1):
            _j = abs(j_offset) + j
            if j == 0:
                j = None
            output[-1].append(a[_i:i, _j:j])
    return output


def window_on_last_axis(a, size):
    """
    Return a 3-dimensional array with all neighbours on the last axis
    """
    views = get_window_views(a, size)
    tarr = numpy.copy(views[0][0])
    try:
        tarr = tarr.reshape(tarr.shape[0], tarr.shape[1], 1).repeat(size[0] * size[1], axis=2)
    except MemoryError:
        raise BlueUtilError('Could not allocate enough memory to create a 3D array of all neighbours')
    cnt = -1
    for i in range(len(views)):
        for j in range(len(views[0])):
            cnt += 1
            tarr[:, :, cnt] = views[i][j]
    return tarr


def window_local_dict(views, prefix='a'):
    '''
    Create a local dictionary with variable names to craft a numexpr
    expression using offsets of a moving window
    '''
    return {'%s%s_%s' % (prefix, i, j): views[i][j]
            for i in range(len(views))
            for j in range(len(views[i]))}


def indices_to_coords(indices, top, left, csx, csy):
    """
    Convert a tuple of ([i...], [j...]) indices to coordinates
    :param indices:
    :param top:
    :param left:
    :param csx:
    :param csy:
    :return: Coordinates: ([y...], [x...])
    """
    i, j = numpy.asarray(indices[0]), numpy.asarray(indices[1])
    return ((top - (csy / 2.)) - (i * csy),
            (left + (csx / 2.)) + (j * csx)
            )


def intersect_mask(coords, top, left, csx, csy, shape):
    """
    Generate a mask of coordinates that intersect a domain
    :param coords: Tuple of coordinates in the form ([x...], [y...])
    :param top: Top coordinate of array
    :param left: Left coordinate of array
    :param csx: Cell size in the x-direction
    :param csy: Cell size in the y-direction
    :param shape: Shape of array (for bounds)
    :return: 1-d mask where points intersect domain
    """
    x, y = numpy.asarray(coords[0]), numpy.asarray(coords[1])
    i = numpy.int64((top - y) / csy)
    j = numpy.int64((x - left) / csx)
    return (i > 0) & (j > 0) & (i < shape[0]) & (j < shape[1])


def coords_to_indices(coords, top, left, csx, csy, shape, preserve_out_of_bounds=False):
    """
    Convert coordinates to array indices using the given specs.
    Coordinates outside of the shape are not returned.
    :param coords: Tuple of coordinates in the form ([x...],    [y...])
    :param top: Top coordinate of array
    :param left: Left coordinate of array
    :param csx: Cell size in the x-direction
    :param csy: Cell size in the y-direction
    :param shape: Shape of array (for bounds)
    :return: tuple of indices in the form ([i...], [j...])
    """
    x, y = numpy.asarray(coords[0]), numpy.asarray(coords[1])
    i = numpy.int64((top - y) / csy)
    j = numpy.int64((x - left) / csx)
    if preserve_out_of_bounds:
        return i, j
    else:
        m = (i >= 0) & (j >= 0) & (i < shape[0]) & (j < shape[1])
        return i[m], j[m]


def kernel_from_distance(distance, csx, csy):
    """
    Calculate a kernel mask using distance
    :param distance: Radius for kernel
    :param csx:
    :param csy:
    :return: kernel mask
    """
    num_cells_x = numpy.ceil(round((distance * 2.) / csx)) + 1
    num_cells_y = numpy.ceil(round((distance * 2.) / csy)) + 1
    centroid = (int((num_cells_y - 1) / 2.), int((num_cells_x - 1) / 2.))
    kernel = numpy.ones(shape=(int(num_cells_y), int(num_cells_x)), dtype='bool')
    kernel[centroid] = 0
    dt = ndimage.distance_transform_edt(kernel, (csy, csx))
    return dt <= distance


def stride_hood(a, window=(3, 3)):
    """
    Return views into array over a given neighbourhood window
    :param a: ndarray
    :param window: window shape
    :param edge_mode: method to expand array to fit window
    :param constant_values: if edge mode is 'constant', value to use
    :return: views into a over with window as a last two axes
    """

    def rolling_window_lastaxis(a, window):
        if window < 1:
            raise ValueError("window must be at least 1.")
        if window > a.shape[-1]:
            raise ValueError("window is too long.")
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        a = a.swapaxes(i, -1)
        a = rolling_window_lastaxis(a, win)
        a = a.swapaxes(-2, i)
    return a



def mode(ndarray, axis=0):
    # Check inputs
    ndarray = numpy.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return ndarray[0], 1
    elif ndarray.size == 0:
        raise BlueUtilError('Cannot compute mode on empty array')
    try:
        axis = list(range(ndarray.ndim))[axis]
    except:
        raise BlueUtilError('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 numpy.unique will suffice
    if all([ndim == 1,
            int(numpy.__version__.split('.')[0]) >= 1,
            int(numpy.__version__.split('.')[1]) >= 9]):
        modals, counts = numpy.unique(ndarray, return_counts=True)
        index = numpy.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = numpy.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = numpy.roll(numpy.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = numpy.concatenate([numpy.zeros(shape=shape, dtype='bool'),
                                 numpy.diff(sort, axis=axis) == 0,
                                 numpy.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = numpy.cumsum(strides)
    counts[~strides] = numpy.concatenate([[0], numpy.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = numpy.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[slices] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = numpy.ogrid[slices]
    index.insert(axis, numpy.argmax(counts, axis=axis))
    return sort[index], counts[index]
