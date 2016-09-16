'''
Functions for completing custom spatial filters
'''

from raster import *
import util
import scipy.ndimage as scifilter


def edgedetecttask(args):
    af, tile, nodata, inside, outside, val = args
    xoff, yoff, win_xsize, win_ysize = tile
    insidemma = numpy.load(inside, mmap_mode='r+')
    outsidemma = numpy.load(outside, mmap_mode='r+')
    mma = numpy.load(af, mmap_mode='r')
    islice, jslice, insislice, insjslice = util.tileHood(tile, (3, 3),
                                                         mma.shape)
    a = numpy.copy(mma[islice, jslice])
    inmask = numpy.zeros(shape=a.shape, dtype='bool')
    outmask = numpy.zeros(shape=a.shape, dtype='bool')
    view = util.getWindowViews(a, (3, 3))
    for i in range(3):
        for j in range(3):
            m = view[i][j] == val
            inmask[1:-1, 1:-1][m] = 1
            outmask[1:-1, 1:-1][~m] = 1
    outmask = a == val
    inmask = inmask & outmask
    insidemma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] =\
        outmask[insislice, insjslice] & inmask[insislice, insjslice]
    outsidemma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] =\
        ~outmask[insislice, insjslice] & inmask[insislice, insjslice]
    insidemma.flush()
    outsidemma.flush()


def edgeDetect(data, edge_value):
    '''
    Compute the edges of data where edge_value exists.
    Returns two boolean rasters (inside and outside).
    '''
    util.parseInput(data)
    inside = data.writeFile(dtype='bool', shape=data.shape)
    outside = data.writeFile(dtype='bool', shape=data.shape)
    data.runTiles(edgedetecttask, args=(inside, outside, edge_value))
    return (numpy.load(inside, mmap_mode='r+'),
            numpy.load(outside, mmap_mode='r+'))


def locatereclasstask_one(args):
    af, tile, nodata, minval = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r+')
    a = numpy.copy(mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize])
    a[a == 0] = minval - 1
    a[a == nodata] = 0
    mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] = a
    mma.flush()


def locatereclasstask_two(args):
    af, tile, nodata = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r+')
    a = numpy.copy(mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize])
    a[a == 0] = nodata
    mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] = a
    mma.flush()


def locate(data):
    '''
    Locate clusters of contiguous cells with the same value.
    Returns a raster with regions indexed.
    '''
    util.parseInput(data)
    output = raster(data)
    output.changeDataType('int32')
    # Replace 0 values with another value if necessary
    if output.nodata[0] != 0:
        minval = data.rastermin()
        output.runTiles(locatereclasstask_one, args=(minval,))
    a = output.load('r+')
    # Implement using scipy measurements label:
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html#scipy.ndimage.label
    scifilter.label(a, numpy.ones(shape=(3, 3), dtype='bool'), output=a)
    a.flush()
    if output.nodata[0] != 0:
        output.runTiles(locatereclasstask_two)
    return output


def expand(data, value, num_cells):
    '''
    Expand regions of "value" by a width of num_cells.
    Warning: high memory usage may result for
    large rasters
    '''
    util.parseInput(data)
    num_cells = int(num_cells)
    output = raster(data)
    mma = output.load('r+')
    a = numpy.copy(mma)
    m = a == value
    if numpy.sum(m) == 0:
        raise Exception('The value %s does not exist in the input raster' %
                        value)
    m = scifilter.binary_dilation(m, numpy.ones(shape=(3, 3), dtype='bool'),
                                  iterations=num_cells)
    a[m] = value
    mma[:] = a
    mma.flush()
    return output


def distance(data, value):
    '''Compute the distance to the nearest cell with the "value" everywhere'''
    util.parseInput(data)
    output = raster(data)
    if value != 0:
        mma = output.load('r+')
        a = numpy.copy(mma)
        a[a == 0] = value + 1
        a[a == value] = 0
        mma[:] = a
        mma.flush()
        del a
    output.changeDataType('float64')
    mma = output.load('r+')
    a = numpy.copy(mma)
    scifilter.distance_transform_edt(a, (output.csx, output.csy), distances=a)
    mma[:] = a
    mma.flush()
    output.nodata = [0]
    return output


def meantask(args):
    af, tile, nodata, nbrhood, iterations, sa = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r+')
    ina = numpy.load(sa, mmap_mode='r+')
    # set up neighborhood
    islice, jslice, insislice, insjslice = util.tileHood(tile, nbrhood,
                                                         mma.shape)
    a = numpy.copy(ina[islice, jslice])
    if numpy.all(a == nodata):
        return
    nodatamask = a == nodata
    view = util.getWindowViews(a, nbrhood)
    itop, ibot, jleft, jright = util.getViewInsertLocation(nbrhood)
    out = numpy.zeros(shape=a.shape, dtype='float32')
    cnt = numpy.zeros(shape=a.shape, dtype='int8')
    for i in range(nbrhood[0]):
        for j in range(nbrhood[1]):
            m = view[i][j] != nodata
            out[itop:ibot, jleft:jright][m] += view[i][j][m]
            cnt[itop:ibot, jleft:jright][m] += 1
    m = ~nodatamask & (cnt != 0)
    out[m] /= cnt[m]
    out[nodatamask] = nodata
    mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] =\
        out[insislice, insjslice]
    mma.flush()


def meanfilter(data, neighborhood=(3, 3), iterations=1):
    '''
    Compute a mean (uniform) filter.
    The neighborhood is the filter window
    '''
    util.parseInput(data)
    output = raster(data)
    output.changeDataType('float32')
    # Collect edges for replacement
    a = output.load('r')
    edges = (a[0, :].copy(), a[-1, :].copy(), a[:, 0].copy(), a[:, -1].copy())
    for i in range(iterations):
        output.runTiles(meantask, args=(neighborhood, iterations, data.array))
    a = output.load('r+')
    a[0, :], a[-1, :], a[:, 0], a[:, -1] = edges
    a.flush()
    print "Completed a mean filter %i times" % (iterations)
    return output


def modetask(args):
    af, tile, nodata, nbrhood, iterations, sa = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r+')
    ina = numpy.load(sa, mmap_mode='r')
    # set up neighborhood
    islice, jslice, insislice, insjslice = util.tileHood(tile, nbrhood,
                                                         mma.shape)
    ndarray = numpy.copy(ina[islice, jslice])
    if numpy.all(ndarray == nodata):
        return
    ndarray = util.windowAs3D(ndarray, nbrhood)
    axis = 2
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Attempted to find mode on an empty array!')
    axis = [i for i in range(ndarray.ndim)][axis]
    srt = numpy.sort(ndarray, axis=axis)
    dif = numpy.diff(srt, axis=axis)
    shape = [i for i in dif.shape]
    shape[axis] += 2
    indices = numpy.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1, -1)
                   for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)
    bins = numpy.diff(indices, axis=axis)
    location = numpy.argmax(bins, axis=axis)
    mesh = numpy.indices(bins.shape)
    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [mesh[i][index].ravel() if i != axis else location.ravel()
             for i in range(bins.ndim)]
    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)
    mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] =\
        modals[insislice, insjslice]
    mma.flush()


def modefilter(data, neighborhood=(3, 3), iterations=1):
    '''
    Compute a mode moving filter over
    the neigborhood window
    '''
    util.parseInput(data)
    output = raster(data)
    # Hack tile size to be smaller due to memory intensiveness
    prevts = output.tilesize
    output.tilesize = 0.05
    output.createTileDefn()
    for i in range(iterations):
        output.runTiles(modetask, args=(neighborhood, iterations, data.array))
    print "Completed a mode filter %i times" % (iterations)
    output.tilesize = prevts
    output.createTileDefn()
    return output
