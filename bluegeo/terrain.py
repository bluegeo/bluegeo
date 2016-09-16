'''
Terrain analysis functions
'''

from raster import *
import util
import spatial
import numpy.ma as ma
import math


def slopetask(args):
    af, tile, nodata, sa, output, exag, csx, csy = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r+')
    ina = numpy.load(sa, mmap_mode='r')
    # set up neighborhood
    islice, jslice, insislice, insjslice = util.tileHood(tile, (3, 3),
                                                         mma.shape)
    a = numpy.copy(ina[islice, jslice])
    if numpy.all(a == nodata):
        return
    a = ma.masked_equal(a, nodata)
    a *= float(exag)
    #  ___ ___ ___   ___ ___ ___
    # |_dx|___|dx_| |_dy|2dy|dy_|
    # |2dx|___|2dx| |___|___|___|
    # |_dx|___|dx_| |_dy|2dy|dy_|
    #
    dx = ((a[:-2, :-2] + (2 * a[1:-1, :-2]) + a[2:, :-2]) -
          (a[2:, 2:] + (2 * a[1:-1, 2:]) + a[:-2, 2:])) / (8 * csx)
    dy = ((a[:-2, :-2] + (2 * a[:-2, 1:-1]) + a[:-2, 2:]) -
          (a[2:, :-2] + (2 * a[2:, 1:-1]) + a[2:, 2:])) / (8 * csy)
    if output == 'percent rise':
        dx = numpy.sqrt((dx**2) + (dy**2)) * 100
    else:
        dx = numpy.arctan(numpy.sqrt((dx**2) + (dy**2))) * (180. / math.pi)
    a[1:-1, 1:-1] = dx
    a[numpy.isnan(a)] = nodata
    mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] =\
        ma.filled(a[insislice, insjslice], nodata)
    mma.flush()


def slope(surface, exag=1, units='degrees'):
    '''compute slope in "degrees" or "percent rise"'''
    util.parseInput(surface)
    slope = raster(surface)
    slope.changeDataType('float32')
    slope.runTiles(slopetask, args=(surface.array, units, exag, slope.csx,
                                    slope.csy))
    a = slope.load('r+')
    a[0, :] = slope.nodata[0]
    a[-1, :] = slope.nodata[0]
    a[:, 0] = slope.nodata[0]
    a[:, -1] = slope.nodata[0]
    a.flush()
    return slope


def aspecttask(args):
    '''
    Compute aspect as a compass (360 > a >= 0)
    or pi (pi > a >= -pi)
    '''
    af, tile, nodata, sa, output, exag, csx, csy = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r+')
    ina = numpy.load(sa, mmap_mode='r')
    # set up neighborhood
    islice, jslice, insislice, insjslice = util.tileHood(tile, (3, 3),
                                                         mma.shape)
    a = numpy.copy(ina[islice, jslice])
    if numpy.all(a == nodata):
        return
    a = ma.masked_equal(a, nodata)
    a *= float(exag)
    #  ___ ___ ___   ___ ___ ___
    # |_dx|___|dx_| |_dy|2dy|dy_|
    # |2dx|___|2dx| |___|___|___|
    # |_dx|___|dx_| |_dy|2dy|dy_|
    #
    dx = ((a[:-2, :-2] + (2 * a[1:-1, :-2]) + a[2:, :-2]) -
          (a[2:, 2:] + (2 * a[1:-1, 2:]) + a[:-2, 2:])) / (8 * csx)
    dy = ((a[:-2, :-2] + (2 * a[:-2, 1:-1]) + a[:-2, 2:]) -
          (a[2:, :-2] + (2 * a[2:, 1:-1]) + a[2:, 2:])) / (8 * csy)
    dx = numpy.arctan2(dx, dy)
    if output == 'compass':
        dx = numpy.degrees(dx) * -1
        dx[dx < 0] += 360
    a[1:-1, 1:-1] = dx
    a[numpy.isnan(a)] = nodata
    mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] =\
        ma.filled(a[insislice, insjslice], nodata)
    mma.flush()


def aspect(surface, exag=1, units='compass'):
    util.parseInput(surface)
    dem = raster(surface)
    dem.changeDataType('float32')
    dem.runTiles(aspecttask, args=(surface.array, units, exag, dem.csx,
                                   dem.csy))
    a = dem.load('r+')
    a[0, :] = dem.nodata[0]
    a[-1, :] = dem.nodata[0]
    a[:, 0] = dem.nodata[0]
    a[:, -1] = dem.nodata[0]
    a.flush()
    return dem


def roughnesstask(args):
    af, tile, nodata, sa, method, nbrhood = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r+')
    ina = numpy.load(sa, mmap_mode='r')
    # set up neighborhood
    islice, jslice, insislice, insjslice = util.tileHood(tile, nbrhood,
                                                         mma.shape)
    a = numpy.copy(ina[islice, jslice])
    if numpy.all(a == nodata):
        return
    nodatamask = a == nodata
    view = util.getWindowViews(a, nbrhood)
    itop, ibot, jleft, jright = util.getViewInsertLocation(nbrhood)
    # Normalize elevation over neighborhood
    max_ = numpy.zeros(shape=a.shape, dtype='float32')
    min_ = numpy.zeros(shape=a.shape, dtype='float32')
    min_[~nodatamask] = numpy.max(a[~nodatamask])
    # Max/Min filter
    for i in range(nbrhood[0]):
        for j in range(nbrhood[1]):
            m = (view[i][j] != nodata) & (view[i][j] > max_[itop:ibot,
                                                            jleft:jright])
            max_[itop:ibot, jleft:jright][m] = view[i][j][m]
            m = (view[i][j] != nodata) & (view[i][j] < min_[itop:ibot,
                                                            jleft:jright])
            min_[itop:ibot, jleft:jright][m] = view[i][j][m]
    # Calculate mean over normalized elevations
    rge = numpy.zeros(shape=a.shape, dtype='float32')
    rge[~nodatamask] = max_[~nodatamask] - min_[~nodatamask]
    del max_
    mean = numpy.zeros(shape=a.shape, dtype='float32')
    modal = numpy.zeros(shape=a.shape, dtype='int8')
    rgemask = rge != 0
    for i in range(nbrhood[0]):
        for j in range(nbrhood[1]):
            m = (view[i][j] != nodata) & rgemask[itop:ibot, jleft:jright]
            mean[itop:ibot, jleft:jright][m] +=\
                (view[i][j][m] -
                 min_[itop:ibot, jleft:jright][m]) / rge[itop:ibot,
                                                         jleft:jright][m]
            modal[itop:ibot, jleft:jright][m] += 1
    repl = ~nodatamask & (modal != 0)
    mean[repl] /= modal[repl]
    # Calculate standard deviation over normalized elevations
    std = numpy.zeros(shape=a.shape, dtype='float32')
    for i in range(nbrhood[0]):
        for j in range(nbrhood[1]):
            m = (view[i][j] != nodata) & rgemask[itop:ibot, jleft:jright]
            std[itop:ibot, jleft:jright][m] +=\
                (((view[i][j][m] - min_[itop:ibot, jleft:jright][m]) /
                  rge[itop:ibot, jleft:jright][m]) - mean[itop:ibot,
                                                          jleft:jright][m])**2
    std[repl] /= modal[repl]
    std[repl] = numpy.sqrt(std[repl])
    std[~rgemask] = 0
    std[nodatamask] = nodata
    mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] =\
        std[insislice, insjslice]
    mma.flush()


def roughness(surface, method='std-elev', neighborhood=(5, 5), smooth=False,
              smooth_sigma=7):
    '''
    Compute the roughness of a surface.
    Methods are:
    "std-elev": standard deviation of locally normalized elevation
    '''
    util.parseInput(surface)
    noise = raster(surface)
    noise.changeDataType('float32')
    noise.runTiles(roughnesstask, args=(surface.array, method, neighborhood))
    a = noise.load('r+')
    a[0, :] = noise.nodata[0]
    a[-1, :] = noise.nodata[0]
    a[:, 0] = noise.nodata[0]
    a[:, -1] = noise.nodata[0]
    a.flush()
    if smooth:
        from scipy.ndimage import gaussian_filter
        mma = noise.load('r+')
        gaussian_filter(noise, smooth_sigma, output=noise)
    return noise
