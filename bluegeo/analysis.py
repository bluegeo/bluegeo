'''
General functions for hydrologic analysis
Devin Cairns, 2016
'''
from raster import *
from terrain import *
from skimage.measure import label
from scipy import ndimage
import math
import time


def region_label(input_raster, return_map=False):
    """
    Label contiguous regions in a raster
    :param input_raster: 
    :param return_map: Return a dictionary of cell indices associated with each label
    :return: output labelled raster, and map of the flattened labels if return_map is True
    """
    rast = raster(input_raster)
    a = rast.array
    a = label(a, background=rast.nodata)
    outrast = rast.astype(a.dtype)
    outrast[:] = a

    if return_map:
        indices = numpy.argsort(a.ravel())
        bins = numpy.bincount(a.ravel())
        indices = numpy.split(indices, numpy.cumsum(bins[bins > 0][:-1]))
        return outrast, dict(zip(numpy.unique(a.ravel()), indices))


def sinuosity(**kwargs):
    """
    Calculate sinuosity from a dem or streams
    :param kwargs: dem=path to dem _or_ stream_order=path to strahler stream order raster
    :return: sinuosity as a ratio
    """
    # Collect as raster of streams
    dem = kwargs.get('dem', None)
    stream_order = kwargs.get('stream_order')
    distance = kwargs.get('sample_distance', 100)  # Default measurement distance is 100m
    if distance == 0:
        raise Exception('Sinuosity sampling distance must be greater than 0')
    if stream_order is not None:
        stream_order = raster(stream_order)
    elif dem is not None:
        min_contrib_area = kwargs.get('min_contrib_area')
        tempdir = kwargs.get('tempdir', None)
        if min_contrib_area is None:
            raise Exception('Minimum contributing area required if deriving streams from DEM')
        stream_order = watershed(dem, tempdir=tempdir).stream_order(min_contrib_area)
    else:
        raise Exception('Sinuosity needs needs either dem or  ')

    # Convolution kernel created using distance
    kernel = util.kernel_from_distance(distance, stream_order.csx,
                                       stream_order.csy)
    # Maximum length of a straight reach
    straight_segment = distance / min(stream_order.csx, stream_order.csy)

    # Label and map stream order
    stream_order, stream_map = region_label(stream_order)

    # Iterate stream orders and calculate sinuosity
    sinuosity_raster = stream_order.copy().astype('float32')
    outa = sinuosity_raster.array
    datamask = outa != sinuosity_raster.nodata
    for region, indices in stream_map:
        # Count cells in neighbourhood
        count_arr = numpy.zeros(shape=sinuosity_raster.shape, dtype='int32').ravel()
        count_arr[indices] = 1
        count_arr = count_arr.reshape(sinuosity_raster.shape)
        count_arr = ndimage.convolve(count_arr, kernel, mode='constant')
        outa[datamask] = count_arr[datamask] / straight_segment
    sinuosity_raster[:] = outa

    return sinuosity_raster


def combine_bathymetry(dem, bathymetry, csv_path):
    print "Aligning datasets"

    # Match the rasters
    bath = raster(bathymetry)
    dem = raster(dem)
    bath.interpolationMethod = 'cubic spline'
    bath.match_raster(dem)
    a = dem.array
    b = bath.array

    # Compute slope to avoid overwriting good elevation data
    a = numpy.pad(a, 1, 'reflect')
    a = numpy.ma.masked_equal(a, dem.nodata)
    dx = ((a[:-2, :-2] + (2 * a[1:-1, :-2]) + a[2:, :-2]) -
          (a[2:, 2:] + (2 * a[1:-1, 2:]) + a[:-2, 2:])) / (8 * dem.csx)
    dy = ((a[:-2, :-2] + (2 * a[:-2, 1:-1]) + a[:-2, 2:]) -
          (a[2:, :-2] + (2 * a[2:, 1:-1]) + a[2:, 2:])) / (8 * dem.csy)
    dx = numpy.arctan(numpy.sqrt((dx**2) + (dy**2))) * (180. / math.pi)
    dx = numpy.ma.filled(dx, dem.nodata)
    a = numpy.ma.filled(a, dem.nodata)

    lake_set = (dx < 0.001) & (bath.array != -9999) & (dx != dem.nodata)
    labels, num = ndimage.label(lake_set, numpy.ones(shape=(3, 3),
                                                     dtype='bool'))
    label_set = numpy.arange(1, num + 1)
    label_ = label_set[numpy.argmax(ndimage.sum(lake_set, labels, label_set))]
    lake_set = labels == label_
    a = a[1:-1, 1:-1]

    # Find cells at the edge of the lake
    edge = ndimage.binary_dilation(lake_set, numpy.ones(shape=(3, 3),
                                                        dtype='bool'))
    edge = edge & ~lake_set & (b != bath.nodata)

    # Compute the difference and create delta surface
    delta = (a[edge] - b[edge]).astype('float32')
    y, x = numpy.where(edge)
    top_c = dem.top - (dem.csy / 2.)
    left_c = dem.left + (dem.csx / 2.)
    y_ = top_c - (y.astype('float32') * dem.csy)
    x_ = left_c + (x.astype('float32') * dem.csx)
    y, x = numpy.where(lake_set)
    y = top_c - (y * dem.csy).astype('float32')
    x = left_c + (x * dem.csx).astype('float32')
    surface_set = numpy.zeros(shape=x.shape, dtype='float32')
    now = time.time()
    # Would do a vectorized outer product, but too memory-intensive
    for i in range(x.shape[0]):
        xi, yi = x[i], y[i]
        dx = ne.evaluate('abs(xi-x_)')
        dy = ne.evaluate('abs(yi-y_)')
        dxs = dx.sum()
        dys = dy.sum()
        dxs = ne.evaluate('sum((dx/dxs)*delta)')
        dys = ne.evaluate('sum((dy/dys)*delta)')
        surface_set[i] = (dxs + dys) / 2
    print "Performed outer product distance-weighted bilinear interpolation in %s seconds" % (time.time() - now)
    delta_surface = numpy.zeros(shape=dem.shape, dtype='float32')
    delta_surface[lake_set] = surface_set

    out = raster(dem)
    out[:] = delta_surface
    out.nodataValues = [0]
    out.save_gdal_raster(r'C:/users/devin/desktop/delta_surface2.tif')

    # Adjust bathymetry using delta
    print "Adjusting bathymetry to align with DEM"
    a[lake_set] = b[lake_set] + delta_surface[lake_set]

    print "Creating output raster"
    out = raster(dem)
    out[:] = a
    out.save_gdal_raster(r'T:\AB_Data\Bathymetry\DIG_2008_0379\bich_bathy.tif')

    m = a == dem.nodata

    def generate_curve():
        start = numpy.round(numpy.min(a[lake_set]))
        stop = numpy.round(numpy.max(a[lake_set]) + 50)
        curve = []
        area = dem.csx * dem.csy
        print ("Generating stage-area-volume curve for elevations ranging from"
               " %s to %s" % (start, stop))
        for i in numpy.linspace(start, stop, ((stop - start) / 0.1) + 1):
            blobs = label(a <= i, background=0)
            blobs[m] = 0
            if numpy.sum(blobs) == 0:
                continue
            unique_blobs = numpy.unique(blobs[blobs != 0])
            if unique_blobs.size == 1:
                # Only one blob
                bloc = blobs != 0
                bloc = numpy.where(bloc)
                if (numpy.any(bloc[0] == a.shape[0] - 1) or
                        numpy.any(bloc[1] == a.shape[1] - 1)):
                    # Lake has reached the edge of the raster
                    print ("Reached edge of raster while generating"
                           " curve")
                    return curve, blobs
                curve.append((i, bloc[0].size * area,
                              numpy.sum(i - a[bloc]) * area))
            else:
                # Only use blobs connected to original lake extent
                areas, vols = [], []
                for blob in unique_blobs:
                    bloc = blobs == blob
                    if numpy.sum(bloc & lake_set) > 0:
                        bloc = numpy.where(bloc)
                        if (numpy.any(bloc[0] == a.shape[0] - 1) or
                                numpy.any(bloc[1] == a.shape[1] - 1)):
                            # Lake has reached the edge of the raster
                            print ("Reached edge of raster while generating"
                                   " curve")
                            return curve, blobs
                        areas.append(bloc[0].size * area)
                        vols.append(numpy.sum(i - a[bloc]) * area)
                curve.append((i, numpy.sum(areas), numpy.sum(vols)))
        return curve, None

    curve, blobs = generate_curve()
    if blobs is not None:
        blobs = blobs.astype(dem.dtype)
        blobs[blobs == 0] = dem.nodata
        out = raster(dem)
        out[:] = blobs
        out.save_gdal_raster(r'C:/Users/Devin/Desktop/lake_level_2.tif')
        del out
    with open(csv_path, 'w') as f:
        f.write('Stage (masl),Surface Area (m2),Volume (m3)\n')
        f.write('\n'.join([','.join(map(str, line)) for line in curve]))


def h60(dem, basins, output_raster):
    '''
    Further divide basins into additional regions based on the H60 line.
    Returns the indices of H60 regions.
    '''
    # Read DEM
    dem = raster(dem)
    a = dem.load('r')
    # Read basins and create output dataset
    bas = raster(basins)
    output = raster(bas)
    loadout = output.load('r+')
    # Create an index for basins
    out = label(loadout, connectivity=2, background=bas.nodata[0],
                return_num=False)
    cursor = numpy.max(out)
    h60basins = []
    # Compute H60 Elevation at each region
    out = out.ravel()
    a = a.ravel()
    indices = numpy.argsort(out)
    bins = numpy.bincount(out)
    bins = numpy.concatenate([[0], numpy.cumsum(bins[bins > 0])])
    for lab, start, stop in zip(numpy.unique(out), bins[:-1], bins[1:]):
        if lab == 0:
            continue
        inds = indices[start:stop]
        elevset = a[inds]
        m = elevset != dem.nodata[0]
        elev_range = numpy.max(elevset[m]) - numpy.min(elevset[m])
        cursor += 1
        if elev_range < 300:
            out[inds] = cursor
            h60basins.append(cursor)
        else:
            h60basins.append(cursor)
            h60elev = numpy.sort(elevset)[int(round((stop - start) * .4))]
            outset = out[inds]
            outset[m & (elevset >= h60elev)] = cursor
            out[inds] = outset
    out = out.reshape(loadout.shape)
    out[out == 0] = output.nodata[0]
    loadout[:] = out
    loadout.flush()
    output.saveRaster(output_raster)
    del a, loadout
    return h60basins
