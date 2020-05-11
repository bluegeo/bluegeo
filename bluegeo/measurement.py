'''
Library for segmentation and label measurement using rasters

blueGeo 2017
'''
from .spatial import *
from skimage.measure import label as sklabel
from skimage.graph import MCP_Geometric
from scipy.ndimage import distance_transform_edt


class MeasurementError(Exception):
    pass


def label(data, return_map=False, raster_template=None):
    """
    Label contiguous regions in a Raster or an array
    :param data: Raster or numpy array
    :param return_map: Return a dictionary of cell indices associated with each label
    :param raster_template: Template Raster to use if using an array
    :return: output labelled Raster or array (if no template), and map of labels if return_map is True
    """
    array_only = False
    if isinstance(data, numpy.ndarray):
        a = data
        background = 0
        if raster_template is not None:
            rast = Raster(raster_template)
            if any([rast.shape[0] != data.shape[0], rast.shape[1] != data.shape[1]]):
                raise MeasurementError("Input Raster template does not match array")
        else:
            array_only = True
    else:
        rast = Raster(data)
        a = rast.array
        background = rast.nodata

    a = sklabel(a, background=background, return_num=False).astype('uint32')
    if array_only:
        outrast = a
    else:
        outrast = rast.astype('uint32')
        outrast.nodataValues = [0]
        outrast[:] = a

    if return_map:
        a = a.ravel()
        indices = numpy.argsort(a)
        bins = numpy.bincount(a)
        indices = numpy.split(indices, numpy.cumsum(bins[bins > 0][:-1]))
        _map = dict(list(zip(numpy.unique(a), [numpy.unravel_index(ind, outrast.shape) for ind in indices])))
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


def distance(sources):
    """
    Calculate distance to sources everywhere in the dataset
    :param sources: Raster with sources as legitimate data
    :return: distance array
    """
    r = Raster(sources)
    out = r.astype('float32')
    out[:] = distance_transform_edt(r.array == r.nodata, [r.csx, r.csy])
    return out


def cost_surface(sources, cost, reverse=False):
    """
    Generate a cost surface using a source Raster and a cost Raster
    :return:
    """
    # Generate cost surface
    cost = Raster(cost).astype('float32')
    sources = Raster(sources).match_raster(cost)
    sources = sources.array != sources.nodata
    _cost = cost.array
    m = _cost != cost.nodata

    if reverse:
        data = _cost[m]
        _cost[m] = data.max() - data

    _cost[~m] = numpy.inf # Fill no data with infinity
    _cost[sources] = 0

    # Compute cost network
    mcp = MCP_Geometric(_cost, sampling=(cost.csy, cost.csx))
    cost_network, traceback = mcp.find_costs(numpy.array(numpy.where(sources)).T)

    # Prepare output
    out = cost.astype('float32')
    cost_network[numpy.isnan(cost_network) | numpy.isinf(cost_network) | ~m] = out.nodata
    out[:] = cost_network

    return out
