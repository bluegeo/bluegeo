'''
Utilities for bluegeo
'''
import math
import numpy
from scipy import ndimage


class BlueUtilError(Exception):
    pass


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
    return ((top - (csy / 2.)) - (indices[0] * csy),
            (left + (csx / 2.)) + (indices[1] * csx)
            )


def kernel_from_distance(distance, csx, csy):
    """
    Calculate a kernel mask using distance
    :param distance: Radius for kernel
    :param csx: 
    :param csy: 
    :return: kernel mask
    """
    num_cells_x = numpy.ceil(round((distance * 2.) / csx))
    num_cells_y = numpy.ceil(round((distance * 2.) / csy))
    centroid = (int(num_cells_y / 2.), int(num_cells_x / 2.))
    kernel = numpy.ones(shape=(num_cells_y, num_cells_x), dtype='bool')
    kernel[centroid] = 0
    dt = ndimage.distance_transform_edt(kernel, (csy, csx))
    return dt <= distance



## OLD stuff ##
# def parseInput(data):
#     if not isinstance(data, raster):
#         raise Exception('Expected a raster instance, got a %s' % type(data))
#     data.read()
#
#
# def windowAs3D(a, window):
#     '''
#     Get a 3-dimensional array with all neighbours
#     in the thrid dimension. Memory intensive!
#     '''
#     tarr = a.reshape(a.shape[0], a.shape[1], 1)
#     tarr = tarr.repeat(window[0] * window[1], axis=2)
#     cnt = -1
#     iedge = (window[0] - 1) / 2
#     jedge = (window[1] - 1) / 2
#     ito = int(math.ceil((window[0] - 1) / 2.))
#     jto = int(math.ceil((window[1] - 1) / 2.))
#     for i in range(iedge * -1, ito + 1):
#         for j in range(jedge * -1, jto + 1):
#             if i == 0 and j == 0:
#                 continue
#             cnt += 1
#             if i < 0 and j < 0:
#                 tarr[:i, :j, cnt] = a[i * -1:, j * -1:]
#             elif i < 0 and j == 0:
#                 tarr[:i, :, cnt] = a[i * -1:, :]
#             elif i < 0 and j > 0:
#                 tarr[:i, j:, cnt] = a[i * -1:, :j * -1]
#             elif i == 0 and j < 0:
#                 tarr[:, :j, cnt] = a[:, j * -1:]
#             elif i == 0 and j > 0:
#                 tarr[:, j:, cnt] = a[:, :j * -1]
#             elif i > 0 and j < 0:
#                 tarr[i:, :j, cnt] = a[:i * -1, j * -1:]
#             elif i > 0 and j == 0:
#                 tarr[i:, :, cnt] = a[:i * -1, :]
#             elif i > 0 and j > 0:
#                 tarr[i:, j:, cnt] = a[:i * -1, :j * -1]
#     return tarr
#
#
# def getWindowViews(a, window):
#     '''
#     Get a "shaped" list of views into "a" to retrieve
#     neighbouring cells in the form of "window."
#     Note: asymmetrical shapes will be propagated
#         up and left.
#     '''
#     i_offset = (window[0] - 1) * -1
#     j_offset = (window[1] - 1) * -1
#     output = []
#     for i in range(i_offset, 1):
#         output.append([])
#         _i = abs(i_offset) + i
#         if i == 0:
#             i = None
#         for j in range(j_offset, 1):
#             _j = abs(j_offset) + j
#             if j == 0:
#                 j = None
#             output[-1].append(a[_i:i, _j:j])
#     return output
#
#
# def getViewInsertLocation(nbrhood):
#     itop = (nbrhood[0] - 1) / 2
#     if itop == 0 and nbrhood[0] > 1:
#         itop = 1
#     ibot = (nbrhood[0] - 1) / 2
#     if ibot == 0:
#         ibot = None
#     else:
#         ibot *= -1
#     jleft = (nbrhood[1] - 1) / 2
#     if jleft == 0 and nbrhood[1] > 1:
#         jleft = 1
#     jright = (nbrhood[1] - 1) / 2
#     if jright == 0:
#         jright = None
#     else:
#         jright *= -1
#     return itop, ibot, jleft, jright
#
#
# def rolling_window_lastaxis(a, window):
#     if window < 1:
#         raise Exception("window must be at least 1.")
#     if window > a.shape[-1]:
#         raise Exception("window is too long.")
#     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
#     strides = a.strides + (a.strides[-1],)
#     return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
#
#
# def strideHood(a, window, edge_method='reflect'):
#     acopy = numpy.zeros(shape=(a.shape[0] + 2, a.shape[1] + 2), dtype=a.dtype)
#     if edge_method == 'reflect':
#         acopy[1:-1, 1:-1] = a
#         acopy[1:-1, 0] = a[:, 0]
#         acopy[1:-1, -1] = a[:, -1]
#         acopy[0, :] = acopy[1, :]
#         acopy[-1, :] = acopy[-2, :]
#     if not hasattr(window, '__iter__'):
#         return rolling_window_lastaxis(acopy, window)
#     for i, win in enumerate(window):
#         if win > 1:
#             acopy = acopy.swapaxes(i, -1)
#             acopy = rolling_window_lastaxis(acopy, win)
#             acopy = acopy.swapaxes(-2, i)
#     return acopy
#
#
# def tileHood(tile, nbrhood, shape):
#     '''Safely expand a tile to accommodate a neighborhood'''
#     itop, ibot, jleft, jright = getViewInsertLocation(nbrhood)
#     if ibot is None:
#         ibot = 0
#     else:
#         ibot *= -1
#     if jright is None:
#         jright = 0
#     else:
#         jright *= -1
#     xoff, yoff, win_xsize, win_ysize = tile
#     if yoff - itop < 0:
#         i = yoff
#         i_ = 0
#     else:
#         i = yoff - itop
#         i_ = itop
#     if yoff + win_ysize + ibot > shape[0]:
#         ispan = shape[0]
#         _i = None
#     else:
#         ispan = yoff + win_ysize + ibot
#         _i = ibot * -1
#     if xoff - jleft < 0:
#         j = xoff
#         j_ = 0
#     else:
#         j = xoff - jleft
#         j_ = jleft
#     if xoff + win_xsize + jright > shape[1]:
#         jspan = shape[1]
#         _j = None
#     else:
#         jspan = xoff + win_xsize + jright
#         _j = jright * -1
#     return slice(i, ispan), slice(j, jspan), slice(i_, _i), slice(j_, _j)
#
#
# def getSlice(i, j, shape, edges=(1, 1, 1, 1)):
#     '''
#     Return a safe slice based on shape.
#     edges are: ifrom, ito, jfrom, jto'''
#     if i - edges[0] < 0:
#         i_ = 0
#     else:
#         i_ = i - edges[0]
#     if i + edges[1] + 1 > shape[0]:
#         _i = shape[0]
#     else:
#         _i = i + edges[1] + 1
#     if j - edges[2] < 0:
#         j_ = 0
#     else:
#         j_ = j - edges[2]
#     if j + edges[3] + 1 > shape[1]:
#         _j = shape[1]
#     else:
#         _j = j + edges[3] + 1
#     return i_, _i, j_, _j
