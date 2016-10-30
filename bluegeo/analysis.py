'''
General functions for hydrologic analysis
Devin Cairns, 2016
'''
from bluegeo.raster import *
from skimage.measure import label


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
