'''
Library for segmentation and label measurement using rasters

blueGeo 2017
'''
from raster import *
from skimage.measure import label as sklabel
from skimage.graph import MCP_Geometric


class MeasurementError(Exception):
    pass


def label(data, return_map=False):
    """
    Label contiguous regions in a raster or an array
    :param data: raster or numpy array
    :param return_map: Return a dictionary of cell indices associated with each label
    :return: output labelled raster, and map of the flattened labels if return_map is True
    """
    if isinstance(data, numpy.ndarray):
        a = data
        background = 0
    else:
        rast = raster(data)
        a = rast.array
        background = rast.nodata

    a = sklabel(a, background=background)
    with rast.astype(a.dtype.name) as outrast:
        outrast.nodataValues = [0]
        outrast[:] = a

    if return_map:
        a = a.ravel()
        indices = numpy.argsort(a)
        bins = numpy.bincount(a)
        indices = numpy.split(indices, numpy.cumsum(bins[bins > 0][:-1]))
        _map = dict(zip(numpy.unique(a), [numpy.unravel_index(ind, outrast.shape) for ind in indices]))
        try:
            del _map[0]
        except KeyError:
            pass
        return outrast, _map
    else:
        return outrast


def centroid():
    pass


def zonal():
    pass


def cost_surface(sources, cost, reverse=False):
    """
    Generate a cost surface using a source raster and a cost raster
    :return:
    """
    # Generate cost surface
    cost = raster(cost).astype('float32')
    sources = raster(sources).match_raster(cost)
    sources = sources.array != sources.nodata
    _cost = cost.array
    m = _cost != cost.nodata
    data = _cost[m]
    dataMin = data.min()
    data = (data - dataMin) / (data.max() - dataMin)
    if reverse:
        data = 1. - data
    _cost[m] = data
    _cost[~m] = data.max()  # Fill no data with large values
    _cost[sources] = 0

    # Compute cost network
    mcp = MCP_Geometric(_cost, sampling=(cost.csy, cost.csx))
    cost_network, traceback = mcp.find_costs(numpy.array(numpy.where(sources)).T)

    # Prepare output
    out = cost.empty()
    cost_network[numpy.isnan(cost_network) | numpy.isinf(cost_network) | ~m] = out.nodata
    out[:] = cost_network

    return out
