'''
Compute riparian connectivity and risk
1.  def implement from naman paper- cumulative effectiveness index

    def bankful- top of channel- elevation above stream

2.  def steep slope contributing- sensitvity
'''

from ..raster import *
from ..terrain import *
from ..spatial import *
from . import util
import math


def dilateAttribute(attr, iterations):
    '''
    Perform an approximate mean dilation of stream attributes
    '''
    return meanDilate(attr, iterations=iterations)


def streamSlope(streams, dem):
    '''
    Compute the slope from cell to cell in streams
    '''
    util.parseInput(streams)
    util.parseInput(dem)
    output = raster(streams)
    a = output.load('r')
    m = a != streams.nodata[0]
    output.changeDataType('float32')
    a = output.load('r+')
    elev = dem.load('r')
    # Compute stream slope
    inds = numpy.where(m)
    diag = math.sqrt(streams.csx**2 + streams.csy**2)
    run = numpy.array([[diag, streams.csy, diag],
                       [streams.csx, 1, streams.csx],
                       [diag, streams.csy, diag]])

    def compute(i, j):
        i_, _i, j_, _j = util.getSlice(i, j, elev.shape)
        base = elev[i, j]
        rise = numpy.abs(base - elev[i_:_i, j_:_j][m[i_:_i, j_:_j]])
        run_ = run[m[i_:_i, j_:_j]]
        run_ = run_[rise != 0]
        rise = rise[rise != 0]
        if run_.size == 0:
            return 0
        else:
            return numpy.mean(numpy.degrees(numpy.arctan(rise / run_)))
    slopefill = [compute(inds[0][i], inds[1][i])
                 for i in range(inds[0].shape[0])]
    a[m] = slopefill
    a.flush()
    del a
    print "Computed stream slope"
    return output


def aggradation(streams, dem, slope_surface=None, stream_deriv_threshold=75,
                min_slope=3):
    '''
    Compute zones of stream aggradation
    '''
    util.parseInput(streams)
    util.parseInput(dem)
    strmderiv = float(stream_deriv_threshold)
    minslope = float(min_slope)
    if slope_surface is None:
        slope_surface = slope(dem)
    else:
        util.parseInput(slope_surface)
    # Compute change in stream slopes
    strslope = streamSlope(streams, dem)
    strslope = streamSlope(strslope, strslope)
    # Identify seed cells for growth
    seeds = numpy.where(strslope.load('r') > strmderiv)
    if seeds[0].size == 0:
        a = strslope.load('r')
        m = a != strslope.nodata[0]
        raise Exception('No seeds could be created with the specified stream'
                        ' derivative threshold.  Computed values range from %s'
                        ' to %s.' % (numpy.min(a[m]), numpy.max(a[m])))
    # Prepare output
    regions = raster(streams)
    a = regions.load('r+')
    a.fill(0)
    a.flush()
    regions.nodata = [0]
    regions.changeDataType('bool')
    # Iterate seeds and recursively delineate regions
    a = regions.load('r+')
    elev = dem.load('r')
    m = slope_surface.load('r') >= minslope
    for seed in range(seeds[0].size):
        inds = [seeds[0][seed], seeds[1][seed]]
        if a[inds]:
            continue
        while len(inds[0]) > 0:
            i_, _i, j_, _j = util.getSlice(inds[0][0], inds[1][0], a.shape)
            curelev = elev[inds[0][0], inds[1][0]]
            del inds[0][0], inds[1][0]
            find = numpy.where(m[i_:_i, j_:_j] & (elev[i_:_i, j_:_j] < curelev) & ~
                               a[i_:_i, j_:_j])
            if find[0].size == 0:
                continue
            find = [(find[0] + i_).tolist(), (find[1] + j_).tolist()]
            a[find] = 1
            inds[0] += find[0]
            inds[1] += find[1]
    a.flush()
    del a
    return regions


def generateStreams(flow_accumulation, min_contrib_area):
    '''
    Generate a network of streams from a flow accumulation surface
    (reclassifies using the minimum contributing area)
    '''
    util.parseInput(flow_accumulation)
    area = float(min_contrib_area)
    streams = raster(flow_accumulation)
    a = streams.load('r+')
    m = (a >= area / (streams.csx * streams.csy)) &\
        (a != streams.nodata[0])
    a[m] = 1
    a[~m] = 0
    a.flush()
    del a
    streams.changeDataType('bool')
    streams.nodata = [0]
    return streams


def expandStreams(streams, dem, slope_surface=None, slope_threshold=3,
                  stream_slope_cutoff=None, flow_accumulation=None,
                  elevation_tolerance=0.01):
    '''
    Expand the extent of streams based on slope and elevation
    using existing streams.  If a stream slope cutoff is used,
    streams above regions of that slope will not be included. If
    flow_accumulation is included with stream slope cutoff, the
    slope will be normalized by the contributing area, such that
    larger streams have less of a change of being truncated by
    the slope cutoff.
    Elevation tolerance helps the algorithm propagate better
    when elevations are rough.
    '''
    util.parseInput(streams)
    util.parseInput(dem)
    thresh = float(slope_threshold)
    if slope_surface is None:
        slope_surface = slope(dem)
    else:
        util.parseInput(slope_surface)
    elev = dem.load('r')
    # Prepare slope data
    sl = slope_surface.load('r')
    slm = sl <= thresh
    # Prepare stream data
    streams = raster(streams)
    a = streams.load('r')
    m = a != streams.nodata[0]
    if stream_slope_cutoff is not None:
        # Compute stream slope
        a[~m] = streams.nodata[0]
        a.flush()
        streams = streamSlope(streams, dem)
        a = streams.load('r+')
        if flow_accumulation is not None:
            # Normalize by contributing area
            util.parseInput(flow_accumulation)
            fa = flow_accumulation.load('r')
            a[m] = a[m] * (1 - (fa[m] / numpy.max(fa[m])))
            a.flush()
        stream_slope_cutoff = float(stream_slope_cutoff)
        m = m & (a <= stream_slope_cutoff)
    # Prepare output
    output = raster(streams)
    a = output.load('r+')
    a.fill(0)
    a.flush()
    output.changeDataType('bool')
    output.nodata = [0]
    a = output.load('r+')
    # Prepare indices for iteration
    inds = numpy.where(m & (elev == elev[m].min()))
    a[inds] = 1
    inds = [inds[0].tolist(), inds[1].tolist()]
    # Recursively delineate streams
    while len(inds[0]) > 0:
        i_, _i, j_, _j = util.getSlice(inds[0][0], inds[1][0], a.shape)
        curelev = elev[inds[0][0], inds[1][0]] - elevation_tolerance
        del inds[0][0], inds[1][0]
        find = numpy.where((m[i_:_i, j_:_j] |
                            (slm[i_:_i, j_:_j] &
                             (elev[i_:_i, j_:_j] >= curelev))) & ~
                           a[i_:_i, j_:_j])
        if find[0].size == 0:
            continue
        find = [(find[0] + i_).tolist(), (find[1] + j_).tolist()]
        a[find] = 1
        inds[0] += find[0]
        inds[1] += find[1]
    a.flush()
    # Compute a mode filter
    m = a != 0
    output = modefilter(output)
    a = output.load('r+')
    a[m] = 1
    a.flush()
    del a
    print "Completed stream extent delineation"
    return output


def streamBuffer(streams, distance, method='fast'):
    '''
    Calculate a buffer of cells around streams. If method "fast"
    is used, the buffer may be blocky and accurate to the nearest
    cell size. If "slow" is used, the buffer will look more
    realistic, but the algorithm will be (aptly) slower.
    '''
    if method == 'slow':
        streams = streamDistance(streams)
        mma = streams.load('r+')
        a = numpy.copy(mma)
        m = a > distance
        a[m] = streams.nodata[0]
        a[~m] = 1
        mma[:] = a
        mma.flush()
        del mma
    else:
        # Read input
        util.parseInput(streams)
        # Allocate binary input/output
        streams = raster(streams)
        # Assign binary mask for growth
        a = streams.load('r+')
        m = a != streams.nodata[0]
        a[m] = 1
        a[~m] = 0
        a.flush()
        del a
        # Compute buffer width
        num_cells = int(math.ceil(distance / min([streams.csx, streams.csy])))
        # Allocate buffer
        streams = expand(streams, 1, num_cells)
    print "Stream Buffer Complete"
    return streams


def streamDistance(streams):
    '''Compute the distance to the nearest stream- everywhere.'''
    # Read input
    util.parseInput(streams)
    # Allocate mask input
    strdist = raster(streams)
    mma = strdist.load('r+')
    a = numpy.copy(mma)
    a[:] = (a == strdist.nodata[0]).astype('float64')
    mma[:] = a
    mma.flush()
    del mma
    # Allocate distance
    strdist = distance(strdist, 0)
    "Distance to Stream Calculation Complete"
    return strdist


def riparianDelineation(dem, slope_raster, streams, min_cut_slope,
                        number_benches=1):
    '''
    Define zones of riparian connectivity.

    min_cut_slope is the lowest slope of cut point slopes.
    max_bench_slope is the maximum slope a bench can be.
    number_benches is the number of sequential benches to delineate as part of
    the riparian zone.
    '''

    # Match rasters to streams
    util.parseInput(dem)
    util.parseInput(slope_raster)
    # Create stream mask for iteration
    str_mask = raster(streams)
    str_mask.changeDataType('bool')
    mma = str_mask.load('r+')
    st = streams.load('r')
    mma[:] = st != streams.nodata[0]
    mma.flush()
    str_mask.nodata = [0]
    str_mask = streamBuffer(str_mask, min([str_mask.csx, str_mask.csy]))
    benches = raster(str_mask)
    benches.changeDataType('int8')
    mma = benches.load('r+')
    mma.fill(0)
    mma.flush()
    for i in range(1, number_benches * 2, 2):
        # Cutslope phase
        lowslope = raster(slope_raster)
        sl = lowslope.load('r+')
        sl[sl < min_cut_slope] = lowslope.nodata[0]
        sl.flush()
        lowslope = locate(lowslope)
        sl = lowslope.load('r')
        st = str_mask.load('r+')
        be = benches.load('r+')
        unique_regions = numpy.unique(sl[st])
        unique_regions = unique_regions[unique_regions != lowslope.nodata[0]]
        print "%i high slope regions found" % (unique_regions.size)
        be[numpy.in1d(sl.ravel(), unique_regions).reshape(benches.shape)] = i
        be.flush()
        st[be == i] = 1
        st.flush()
        # Bench phase
        lowslope = raster(slope_raster)
        sl = lowslope.load('r+')
        sl[sl >= min_cut_slope] = lowslope.nodata[0]
        sl.flush()
        lowslope = locate(lowslope)
        sl = lowslope.load('r')
        st = str_mask.load('r+')
        be = benches.load('r+')
        unique_regions = numpy.unique(sl[st])
        unique_regions = unique_regions[unique_regions != lowslope.nodata[0]]
        print "%i low slope regions found" % (unique_regions.size)
        be[numpy.in1d(sl.ravel(), unique_regions).reshape(benches.shape)] =\
            i + 1
        be.flush()
        st[be == i + 1] = 1
        st.flush()
    del mma, st, be
    return benches


def cumefftask(args):
    af, tile, nodata, th, sa = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r+')
    ina = numpy.load(sa, mmap_mode='r')
    a = numpy.copy(ina[yoff:yoff + win_ysize, xoff:xoff + win_xsize])
    a /= th
    a[(a > 1) | (a == 0)] = nodata
    mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] = a
    mma.flush()


def cumulativeEffectiveness(stream_raster, tree_height=45):
    '''Define zone of riparian cumulative effectiveness'''
    # Read input
    util.parseInput(stream_raster)
    # Compute distance
    dist = streamDistance(stream_raster)
    dist.changeDataType('float32')
    root = raster(dist)
    litter = raster(dist)
    shade = raster(dist)
    coarse = raster(dist)
    # Shade
    shade.runTiles(cumefftask, args=(tree_height, dist.array))
    # Coarse Wood
    coarse.runTiles(cumefftask, args=(tree_height, dist.array))
    # Litter Fall
    litter.runTiles(cumefftask, args=(tree_height * 0.6, dist.array))
    # Root Strength
    root.nodata = [-1]
    a = root.load('r+')
    a[a < 0.25 * tree_height] = 0
    a[a >= 0.25 * tree_height] = 1
    s = dist.load(in_memory=True)
    a[s == 0] = root.nodata[0]
    a[s > tree_height] = root.nodata[0]
    a.flush()
    del a
    print "Complete Cumulative Effectiveness Calculations"
    return root, litter, shade, coarse


def vegEffectiveness(stream_raster, canopy_height_surface, tree_height=45):
    '''
    Calculate the effectiveness surrounding a stream using
    a buffer dictated by tree height, and a canopy height raster
    '''
    util.parseInput(stream_raster)
    util.parseInput(canopy_height_surface)
    # Compute stream buffers using distance and
    #   clip and reclass canopy_height using buffers
    canopy_rec = raster(canopy_height_surface)
    a = canopy_rec.load('r+')
    a[(a != canopy_rec.nodata[0]) & (a > tree_height)] = tree_height
    a.flush()
    shade = streamDistance(stream_raster)
    litter_height = raster(shade)
    root_height = raster(shade)
    a = shade.load('r+')
    a[(a > tree_height) | (a == 0)] = shade.nodata[0]
    m = a != shade.nodata[0]
    canopy = canopy_rec.load('r')
    a[m] = canopy[m]
    max_height = a[m].max()
    if max_height > 0:
        a[m] = a[m] / max_height
    else:
        a[m] = 0
    a.flush()
    coarse = raster(shade)
    a = litter_height.load('r+')
    a[(a > (tree_height * 0.6)) | (a == 0)] = litter_height.nodata[0]
    m = a != litter_height.nodata[0]
    a[m] = canopy[m]
    max_height = a[m].max()
    if max_height > 0:
        a[m] = a[m] / max_height
    else:
        a[m] = 0
    a.flush()
    a = root_height.load('r+')
    a[(a < (tree_height * 0.25)) | (a > tree_height)] = root_height.nodata[0]
    m = a != root_height.nodata[0]
    a[m] = canopy[m]
    max_height = a[m].max()
    if max_height > 0:
        a[m] = a[m] / max_height
    else:
        a[m] = 0
    a.flush()
    del a
    return root_height, litter_height, shade, coarse

# Junkyard function pasted for parts
def riparianRisk(dem, slope, streams, outpath, maxdist=60, max_flat_slope=3, min_highslope_thresh=20):
    a = raster(streams)
    a.numpyarray()
    a.array[a.array == a.nodata[0]] = 0
    a.array[a.array != 0] = 1
    inds = numpy.where(a.array.astype(bool))
    ash = a.array.shape
    del a
    dem = raster(dem)
    dem.numpyarray()
    nodata = dem.nodata
    dem = dem.array
    s = raster(slope)
    s.numpyarray()
    if s.array.shape != dem.shape or s.array.shape != ash:
        print "Shapes are wrong:", s.array.shape, dem.shape, ash
    out = numpy.zeros(shape=dem.shape,dtype='float32')
    out[dem == nodata] = -1
    out[inds] =  min_highslope_thresh / ((s.csx + abs(s.csy) + math.sqrt((s.csx**2) + (abs(s.csy)**2))) / 3)
    inds = numpy.array([inds[0],inds[1]]).swapaxes(0,1)
    p = 0.1
    for ind in range(inds.shape[0]):
        pc = round(float(ind) / inds.shape[0],1)
        if pc == p:
            print '%s percent complete' % (p*100)
            p += 0.1
        tc = inds[ind].reshape(1,2)
        while len(tc) > 0:
            i, j = tc[0]
            tc = tc[1:,:]
            if i - 1 < 0: i_ = 0
            else: i_ = i - 1
            if j - 1 < 0: j_ = 0
            else: j_ = j - 1
            if i + 2 > dem.shape[0] - 1: i__ = dem.shape[0] - 1
            else: i__ = i + 2
            if j + 2 > dem.shape[1] - 1: j__ = dem.shape[1] - 1
            else: j__ = j + 2
            d = findDist(inds[ind][0],inds[ind][1],numpy.mgrid[i_:i__,j_:j__],s.csx,abs(s.csy))
            wind = s.array[i_:i__,j_:j__]
            elev = dem[i_:i__,j_:j__]
            m = (out[i_:i__,j_:j__] == 0) & (elev > dem[i,j]) & ((d < maxdist) | ((wind > min_highslope_thresh) | ((0 <= wind) & (wind <= max_flat_slope))))
            out[i_:i__,j_:j__] = numpy.maximum(out[i_:i__,j_:j__],wind / d)
            if numpy.all(~m): continue
            ai, aj = numpy.where(m)
            newinds = numpy.zeros(shape=(tc.shape[0] + ai.shape[0],2),dtype='int64')
            newinds[tc.shape[0]:,0] = ai + (i - 1)
            newinds[tc.shape[0]:,1] = aj + (j - 1)
            newinds[:tc.shape[0],:] = tc
            tc = newinds
    out[out == 0] = -1
    s.array = out
    s.nodata = -1
    try:
        s.save(outpath,s.array)
    except Exception:
        s.save(outpath.rstrip('.tif') + '01.tif',s.array)


def findDist(i, j, inds, csx, csy):
    a = numpy.sqrt(((numpy.abs(inds[0] - i)*csy)**2) + ((numpy.abs(inds[1] - j)*csx)**2))
    a[a == 0] = (csx + csy) / 2
    return a
