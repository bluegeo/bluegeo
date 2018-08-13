'''
Hydrologic analysis library

Blue Geosimulation, 2018
'''
from terrain import *
from filters import *
from measurement import *
import bluegrass
from scipy.ndimage import binary_dilation


class WaterError(Exception):
    pass


class HruError(Exception):
    pass


def delineate_watersheds(points, dem=None, fd=None, fa=None, as_vector=True, snap_tolerance=1E6):
    """
    Delineate watersheds from pour points
    :param points: Vector or list of coordinate tuples in the form [(x1, y1), (x2, y2),...(xn, yn)]
    :param dem: digital elevation model Raster (if no flow direction surface is available)
    :param fd: Flow direction surface (if available)
    :param fa: Flow accumulation surface (if available). This will only be used for snapping pour points
    :param as_vector: Return a polygon vector with a different feature for each watershed
    :param snap_tolerance: Snap the pour points to a minimum basin size. Use 0 to omit this argument
    :return: Vector (if as_vector is True) or Raster (if as_vector is False), and the snapped points if specified
    """
    if fd is None:
        if dem is None:
            raise WaterError('One of either a DEM or Flow Direction must be specified')
        fd, fa = bluegrass.watershed(dem)

    if snap_tolerance > 0:
        if fa is None:
            if dem is None:
                raise WaterError('Either flow accumulation or a DEM must be specified if snapping pour points')
            fd, fa = bluegrass.watershed(dem)

        points = snap_pour_points(points, fd, fa, snap_tolerance)

    else:
        if isinstance(points, basestring) or isinstance(points, Vector):
            # Convert the vector to a list of coordinates in the raster map projection
            points = Vector(points).transform(Raster(fd).projection).vertices[:, [0, 1]]

    if not as_vector:
        return bluegrass.water_outlet(points, direction=fd)

    basins = []
    for point in points:
        basins += bluegrass.water_outlet([point], direction=fd).polygonize()[:]

    # Sort basins by area (largest to smallest)
    srt = numpy.argsort([ogr.CreateGeometryFromWkb(b).Area() for b in basins])[::-1]
    basins = [basins[i] for i in srt]

    return Vector(basins, mode='w', projection=Raster(fd).projection)


def wetness(dem, minimum_area):
    """
    Calculate a wetness index using streams of a minimum contributing area
    :param dem: dem (Raster)
    :param minimum_area: area in units^2
    :return: Raster instance
    """
    return normalize(inverse(cost_surface(bluegrass.stream_order(dem, minimum_area), topo(dem).slope())))


def convergence(size=(11, 11), fd=None):
    """
    Compute the relative convergence of flow vectors (uses directions 1 to
    8, which are derived from flow direction)

    TODO: needs to be fixed because moved out of class method
    :param size:
    :param fd:
    :return:
    """
    def eval_conv(a):
        nd = fd.nodata
        mask = (a > 0) & (a != nd)

        # Convert a into angles
        x, y = numpy.mgrid[0:self.csy * 2:3j, 0:self.csx * 2:3j]
        ang = (numpy.arctan2(y - self.csy, x - self.csx) * -1) + numpy.pi
        a = ne.evaluate('where(mask,a-1,0)')
        a = ang[(0, 0, 0, 1, 2, 2, 2, 1), (2, 1, 0, 0, 0, 1, 2, 2)][a]
        a[~mask] = nd

        # Get neighbours as views and create output
        b = util.window_local_dict(util.get_window_views(a, size), 'a')
        x, y = numpy.mgrid[0:(a.shape[0] - 1) * self.csy:a.shape[0] * 1j,
                           0:(a.shape[1] - 1) * self.csx:a.shape[1] * 1j]
        b.update(util.window_local_dict(util.get_window_views(x, size), 'x'))
        b.update(util.window_local_dict(util.get_window_views(y, size), 'y'))
        pi = numpy.pi
        b.update({'pi': pi, 'nd': nd})
        c = '%s_%s' % ((size[0] - 1) / 2, (size[1] - 1) / 2)
        conv = numpy.zeros(shape=b['a%s' % c].shape, dtype='float32')

        # Iterate neighbours and compute convergence
        size_scale = (size[0] * size[1]) - 1
        for i in range(size[0]):
            for j in range(size[1]):
                if i == int(c[0]) and j == int(c[2]):
                    continue
                at2 = ne.evaluate('where(a%i_%i!=nd,abs(((arctan2(y%i_%i-'
                                  'y%s,x%i_%i-x%s)*-1)+pi)-a%i_%i),nd)' %
                                  (i, j, i, j, c, i, j, c, i, j),
                                  local_dict=b)
                conv = ne.evaluate('where(at2!=nd,conv+((where(at2>pi,(2*'
                                   'pi)-at2,at2)/pi)/size_scale),conv)')
        conv[b['a%s' % c] == nd] = nd
        return conv

    # Calculate fa if not specified
    if fd is None:
        fa, fd = self.route()
    else:
        fd = Raster(fd)
        if 'int' not in fd.dtype:
            fd = fd.astype('int32')
    # Allocate output
    conv = self.empty()
    if fd.useChunks:
        # Iterate chunks and calculate convergence
        for a, s in fd.iterchunks(expand=size):
            s_ = util.truncate_slice(s, size)
            conv[s_] = eval_conv(a).astype('float32')
    else:
        # Calculate over all data
        conv[:] = eval_conv(fd.array)

    return watershed(conv, tempdir=self.tempdir)


def stream_slope(dem, streams, units='degrees'):
    """
    Compute the slope from cell to cell in streams with a minimum
    contributing area.  If streams are specified, they will not be
    computed.
    :param streams:
    :param units:
    :return:
    """
    dem = Raster(dem)
    dem.interpolationMethod = 'bilinear'

    with dem.match_raster(streams) as dem:
        elev = dem.array
    strms = Raster(streams)
    m = strms.array != strms.nodata

    # Compute stream slope
    inds = numpy.where(m)
    diag = math.sqrt(dem.csx**2 + dem.csy**2)
    run = numpy.array([[diag, dem.csy, diag],
                       [dem.csx, 1, dem.csx],
                       [diag, dem.csy, diag]])
    ish, jsh = dem.shape

    def compute(i, j):
        s = (slice(max([0, i - 1]),
                   min([i + 2, ish])),
             slice(max([0, j - 1]),
                   min([j + 2, jsh])))
        base = elev[i, j]
        loc_ind = m[s]
        rise = numpy.abs(base - elev[s][loc_ind])
        run_ = run[:loc_ind.shape[0], :loc_ind.shape[1]][loc_ind]
        run_ = run_[rise != 0]
        rise = rise[rise != 0]
        if run_.size == 0:
            return 0
        else:
            if units == 'degrees':
                return numpy.mean(numpy.degrees(numpy.arctan(rise / run_)))
            else:
                return numpy.mean(rise / run_) * 100

    output = dem.empty()
    a = numpy.full(output.shape, output.nodata, output.dtype)
    slopefill = [compute(inds[0][i], inds[1][i])
                 for i in range(inds[0].shape[0])]
    a[m] = slopefill
    output[:] = a
    return output


def aggradation(stream_slope, slope_thresh=6, stream_slope_thresh=5):
    """
    Use the derivative of stream slope to determine regions of
    aggradation to predict alluvium deposition.  The input slope threshold
    is used as a cutoff for region delineation, which the stream slope
    threshold is the required stream slope to initiate deposition.
    Uses the dem as an input
    surface, and accepts (or they will be derived):

    streams: a streams Raster
    min_contrib_area: minimum contributing area to define streams
    slope: a slope surface used to control the region delineation
    :param stream_slope:
    :param slope_thresh:
    :param stream_slope_thresh:
    :return:
    """
    # Get or compute necessary datasets
    strslo = Raster(stream_slope)
    seeds = set(zip(*numpy.where(strslo.array != strslo.nodata)))
    slope = topo(self).slope().array

    # Create output tracking array
    track = numpy.zeros(shape=self.shape, dtype='uint8')

    # Recursively propagate downstream and delineate alluvium
    # Load elevation data into memory
    ish, jsh = self.shape
    dem = self.array
    dem_m = dem != self.nodata
    streams = strslo.array
    while True:
        try:
            seed = seeds.pop()
        except:
            break
        s = (slice(max(0, seed[0] - 1),
                   min(seed[0] + 2, ish)),
             slice(max(0, seed[1] - 1),
                   min(seed[1] + 2, jsh)))
        str_mask = streams[s] != strslo.nodata
        if (streams[seed] != strslo.nodata) & (track[seed] == 0):
            # If stream exists check slope and initiate growth
            if streams[seed] > stream_slope_thresh:
                track[seed] = 2
            else:
                track[seed] = 1
        # High slope: erosion- directed propagation at higher slopes
        g = (dem[seed] - dem[s]).ravel()
        mask = numpy.argsort(g)
        if track[seed] == 2:
            # Create a mask with correct gradient directions
            mask = (mask > 5).reshape(str_mask.shape)
            mask = mask & (slope[s] < slope_thresh)
            track_add = 2
        # Low slope: aggradation- fan outwards at shallower slopes
        else:
            mask = (mask > 3).reshape(str_mask.shape)
            mask = mask & (slope[s] < (slope_thresh / 2))
            track_add = 1

        # Update track with non-stream cells
        mask = mask & ~str_mask & (track[s] == 0) & dem_m[s]
        s_i, s_j = numpy.where(mask)
        s_i = s_i + s[0].start
        s_j = s_j + s[1].start
        track[(s_i, s_j)] = track_add

        # Update the stack with new stream and other cells
        mask[str_mask & (track[s] == 0)] = 1
        s_i, s_j = numpy.where(mask)
        s_i = s_i + s[0].start
        s_j = s_j + s[1].start
        seeds.update(zip(s_i, s_j))

    alluv_out = self.astype('uint8')
    alluv_out[:] = track
    alluv_out.nodataValues = [0]
    return watershed(alluv_out, tempdir=self.tempdir)


def channel_density(streams, sample_distance=50):
    """
    Compute channel density- poor man's sinuosity
    :param streams: stream Raster
    :param sample_distance: distance to sample density
    :return: Raster instance
    """
    # Allocate output as a Raster cast to 32-bit floating points
    streams = Raster(streams)

    i = numpy.ceil(sample_distance / streams.csy)
    if i < 1:
        i = 1
    j = numpy.ceil(sample_distance / streams.csx)
    if j < 1:
        j = 1
    shape = map(int, (i, j))
    weights = numpy.ones(shape=shape, dtype='float32') / (shape[0] * shape[1])

    # Streams must be a mask
    _streams = streams.empty()
    _streams[:] = (streams.array != streams.nodata).astype(streams.dtype)
    _streams.nodataValues = [0]

    return convolve(_streams, weights)


def sinuosity(dem, stream_order, sample_distance=100):
    """
    Calculate sinuosity from a dem or streams
    :param kwargs: dem=path to dem _or_ stream_order=path to strahler stream order Raster
        distance=search distance to calculate sinuosity ratio
    :return: sinuosity as a ratio

    Updated October 25, 2017
    """
    # Collect as Raster of streams
    stream_order = Raster(stream_order)
    distance = sample_distance
    radius = distance / 2.
    if distance <= 0:
        raise WaterError('Sinuosity sampling distance must be greater than 0')

    # Remove connecting regions to avoid over-counting
    m = min_filter(stream_order).array != max_filter(stream_order).array
    a = stream_order.array
    a[m] = stream_order.nodata

    # Label and map stream order
    stream_labels, stream_map = label(a, True)
    # Get window kernel using distance
    kernel = util.kernel_from_distance(radius, stream_order.csx, stream_order.csy)

    # Iterate stream orders and calculate sinuosity
    @jit(nopython=True)
    def calc_distance(a, csx, csy, output):
        """Brute force outer min distance"""
        diag = numpy.sqrt((csx ** 2) + (csy ** 2))
        iInds, jInds = numpy.where(a)
        for ind in range(iInds.shape[0]):
            i = iInds[ind]
            j = jInds[ind]
            iFr, jFr = i - ((kernel.shape[0] - 1) / 2), j - ((kernel.shape[1] - 1) / 2)
            if iFr < 0:
                kiFr = abs(iFr)
                iFr = 0
            else:
                kiFr = 0
            if jFr < 0:
                kjFr = abs(jFr)
                jFr = 0
            else:
                kjFr = 0
            iTo, jTo = i + ((kernel.shape[0] - 1) / 2) + 1, j + ((kernel.shape[1] - 1) / 2) + 1
            if iTo > a.shape[0]:
                kiTo = kernel.shape[0] - (iTo - a.shape[0])
                iTo = a.shape[0]
            else:
                kiTo = kernel.shape[0]
            if jTo > a.shape[1]:
                kjTo = kernel.shape[1] - (jTo - a.shape[1])
                jTo = a.shape[1]
            else:
                kjTo = kernel.shape[1]
            iInner, jInner = numpy.where(a[iFr:iTo, jFr:jTo] & kernel[kiFr:kiTo, kjFr:kjTo])
            distance = 0
            connected = numpy.empty(iInner.shape, numpy.int64)
            for _ind in range(iInner.shape[0]):
                connected[_ind] = -1
            for _ind in range(iInner.shape[0]):
                localMin = 1E38
                localMinInd = -1
                for _outind in range(iInner.shape[0]):
                    if connected[_outind] != _ind:
                        d = numpy.sqrt((((iInner[_ind] - iInner[_outind]) * csy)**2) +
                                       (((jInner[_ind] - jInner[_outind]) * csx)**2))
                        if d < localMin and d != 0 and d <= diag:
                            localMin = d
                            localMinInd = _outind
                if localMinInd != -1:
                    connected[_ind] = localMinInd
                    distance += localMin
                else:
                    continue
            output[i, j] = distance
        return output

    outa = stream_order.array
    nodata_mask = outa == stream_order.nodata
    sinuosity_raster = stream_order.astype('float32')
    outa = outa.astype('float32')
    cnt = 0
    for region, indices in stream_map.iteritems():
        cnt += 1
        # Create slices using index
        i, j = indices
        iSlice, jSlice = (slice(i.min(), i.max() + 1), slice(j.min(), j.max() + 1))
        i = i - i.min()
        j = j - j.min()
        sinu = numpy.zeros(shape=(iSlice.stop - iSlice.start, jSlice.stop - jSlice.start), dtype='bool')
        sinu[i, j] = True
        count_arr = numpy.zeros(shape=sinu.shape, dtype='float32')
        if sinu.sum() > 1:
            # Count cells in neighbourhood
            count_arr = calc_distance(sinu, stream_order.csx, stream_order.csy, count_arr)
        else:
            count_arr[sinu] = distance
        # Avoid false negatives where a full reach does not exist
        count_arr[count_arr < distance] = distance
        outa[iSlice, jSlice][sinu] = count_arr[sinu]

    sinuosity_raster.nodataValues = [-1]
    outa[nodata_mask] = sinuosity_raster.nodata
    outaVals = outa[~nodata_mask]
    outaMin = outaVals.min()
    outa[~nodata_mask] = (outaVals - outaMin) / (outaVals.max() - outaMin)
    sinuosity_raster[:] = outa

    return sinuosity_raster


def eca(tree_height, disturbance, curve, basins):
    """
    Calculate Equivalent Clearcut Area percentage at each basin
    :param tree_height: Tree height (raster or vector)
        If this is a vector:
            -The height must be the first field
            -Polygons must only include forested ares
        If this is a raster:
            -The values must be tree height
            -Regions that are not classified as forests must have no data values
    :param disturbance:
        Disturbance mask (raster or vector). Regions with disturbance will have a hydrologic recovery of 0.
    :param curve:
        Hydrologic recovery vs tree height curve in the form [(x, y), (x, y),...]. The hydrologic recovery is
        linearly interpolated between points.
    :param basins:
        Basin boundaries  (enumerated raster or vector) used to summarize ECA into a percentage
    :return: Basin vector with an ECA percentage attribute
    """
    @jit(nopython=True)
    def eca_curve(data):
        for i in range(data.shape[0]):
            for j in range(curve.shape[0]):
                if curve[j, 0] <= data[i] < curve[j, 1]:
                    data[i] = data[i] * curve[j, 2] + curve[j, 3]
                elif data[i] < curve[0, 0]:
                    data[i] = curve[0, 0] * curve[0, 2] + curve[0, 3]
                elif data[i] >= curve[-1, 1]:
                    data[i] = curve[-1, 1] * curve[-1, 2] + curve[-1, 3]
        return data

    # Create polygons from basins to hold the ECA percentage output
    basins = assert_type(basins)(basins)
    if isinstance(basins, Raster):
        basins = basins.polygonize()
    basins.mode = 'r+'

    # Calculate linear regression constants for each node in the curve
    curve = numpy.array(curve).T
    x = zip(curve[0][:-1], curve[0][1:])
    y = zip(curve[1][:-1], curve[1][1:])
    curve = numpy.array([(x[0], x[1]) + numpy.linalg.solve([[x, 1.], [x[1], 1]], [y[0], y[1]]) for x, y in zip(x, y)])

    # Calculate ECA from area and hydrologic recovery (derived from tree height and the curve)
    tree_height = assert_type(tree_height)(tree_height)
    disturbance = assert_type(disturbance)(disturbance)

    if isinstance(tree_height, Raster):
        # Calculate ECA at each grid cell
        A = tree_height.csx * tree_height.csy
        height_data = tree_height != tree_height.nodata
        ECA = A * (1. - eca_curve(tree_height.array[height_data], curve))

        # Create an output dataset - the absence of eca data means eca is 0
        output_eca = numpy.zeros(shape=height_data.shape, dtype='float32')
        output_eca[height_data] = ECA

        # Disturbance data must be applied as an array, and will have a value of the area [A * (1 - 0) = A]
        if isinstance(disturbance, Vector):
            disturbance = disturbance.rasterize(tree_height).array
        output_eca[disturbance] = A

        # Summarize by sub-basin
        eca_perc = []
        basins['ID'] = numpy.arange(1, basins.featureCount + 1)
        basins, basin_map = label(basins.rasterize(tree_height, 'ID'), True)
        for basin, inds in basin_map.iteritems():
            eca_perc.append(output_eca[inds].sum() / (inds[0].size * A))
    else:
        # Calculate hydrologic recovery at each polygon
        geos = [shpwkb.loads(geo) for geo in tree_height[:]]
        HR = 1. - eca_curve(tree_height[tree_height.fieldNames[0]], curve)

        # Create a spatial index of geos
        def gen_idx():
            """Generator for spatial index"""
            for i, geo in enumerate(geos):
                yield (i, geo.bounds, None)

        idx = index.Index(gen_idx())

        # Iterate basins and intersect eca polygons and disturbance polygons
        if isinstance(disturbance, Raster):
            disturbance = disturbance.mask.polygonize()

        eca_perc = []
        for basin_num, basin in enumerate(basins[:]):
            print "Working on basin {}".format(basin_num)
            # Start with an ECA of 0, and gradually increase it using intersections
            ECA = 0
            basin_geo = shpwkb.loads(basin)
            # Perform an intersect operation on geos that intersect the basin
            for i in idx.intersection(basin_geo.bounds):
                intersect = basin_geo.intersection(geos[i])
                if intersect.is_empty:
                    continue
                ECA += intersect.area * HR[i]

            # Add disturbance
            for dist in disturbance[:]:
                dist_geo = shpwkb.loads(dist)
                intersect = basin_geo.intersection(dist_geo)
                if intersect.is_empty:
                    continue
                ECA += intersect.area

            eca_perc.append(ECA / basin_geo.area)

    basins['eca_perc'] = eca_perc
    return basins


def h60(dem, basins):
    '''
    Further divide basins into additional regions based on the H60 line.
    Returns the indices of H60 regions.
    '''
    labels, basin_map = label(basins, True)
    a = dem.array
    for basin, inds in basin_map:
        elev_set = a[inds]
        elev_set = elev_set[elev_set != dem.nodata]
        elev = numpy.sort(elev_set)[numpy.int64(inds[0].size * .4)]


def snap_pour_points(points, sfd, fa, min_contrib_area=1E7):
    """
    Snap pour points to a cell with a specified minimum contributing area.
    Points are recursively routed down slope until the minimum contributing area is reached.
    :param points: Vector or list of coordinate tuples in the form [(x1, y1), (x2, y2),...(xn, yn)]
    :param sfd: Single flow direction raster
    :param min_contrib_area: Minimum contributing area in map units (default is 10 km ** 2)
    :return: coordinate tuples in the form [(x1, y1), (x2, y2),...(xn, yn)]
    """
    # Map of flow direction: downstream index
    #
    downstream = {1: (-1, 1),
                  2: (-1, 0),
                  3: (-1, -1),
                  4: (0, -1),
                  5: (1, -1),
                  6: (1, 0),
                  7: (1, 1),
                  8: (0, 1)}

    # Make sure SFD and FA are read into Raster instances
    sfd = Raster(sfd)
    fa = Raster(fa)

    # Check that the sfd and fa maps align
    if not sfd.aligns(fa):
        raise WaterError('Input flow accumulation and single flow direction rasters must align spatially')

    if isinstance(points, basestring) or isinstance(points, Vector):
        # Convert the vector to a list of coordinates in the raster map projection
        points = Vector(points).transform(Raster(sfd).projection).vertices[:, [0, 1]]

    # Convert the coordinates to raster map indices
    points = map(tuple, [[p[0] for p in points], [p[1] for p in points]])
    indices = util.coords_to_indices(points, sfd.top, sfd.left, sfd.csx, sfd.csy, sfd.shape)

    # Collect the area as a unit of number of cells
    num_cells = min_contrib_area / (sfd.csx * sfd.csy)

    snapped_points = []
    point_index = -1  # For warning prints
    for i, j in zip(indices[0], indices[1]):
        point_index += 1
        snapped = True
        while fa[i, j] < num_cells:
            try:
                o_i, o_j = downstream[int(numpy.squeeze(sfd[i, j]))]
                i += o_i
                j += o_j
            except KeyError:
            # except (KeyError, IndexError):  Commented out for testing- revert if you see this
                print "Warning: unable to snap point at index {}".format(point_index)
                snapped = False
                break

        if snapped:
            snapped_points.append((i, j))

    snapped_points = map(tuple, [[pt[0] for pt in snapped_points], [pt[1] for pt in snapped_points]])
    y, x = indices_to_coords(snapped_points, sfd.top, sfd.left, sfd.csx, sfd.csy)

    return zip(x, y)


class HRU(object):
    """
    An HRU instance is used to create spatial units and to calculate summary stats in a model domain

    Datasets may be added in one of 3 ways:
        1. As a "spatial" dataset, which is used to both spatially discretize the domain,
            and provide data for each HRU
        2. As a "zonal" dataset, which is simply summarized to a single value within each
            spatial HRU using a statistical method
        3. Used to "split" HRU's using an area proportion. This is designed to create
            additional HRU's within spatial boundaries using another dataset, such as
            landcover.

    Example:
    ================================================================================================
        # Create an instance of the hru class
        # The domain inherits the properties of the input raster, and is masked by a mask dataset
        # Note, the mask optional (in the case that the watershed is comprised of the DEM data)
        hrus = hru('path_to_dem.tif', 'path_to_mask.shp')

        # Split HRU's by adding a sub-basin file, using the field "name" to assign values,
        #   and call the .rvh heading "SUB_BASIN"
        hrus.add_spatial_data('sub_basins.shp', 'SUB_BASINS', 'mode', vector_attribute='name')

        # Add elevation as a spatial discretization dataset, and split it using an interval of 250m
        # A fixed number, explicit breakpoints, or discrete values (as is the case in the basins
        #   line above) may also be used instead of an interval
        hrus.add_elevation(250)

        # Split HRU's into 4 classes of solar radiation, calling the attribute "SOLRAD"
        hrus.add_spatial_data('solar_radiation.tif', 'SOLRAD', number=4)

        # Remove spatial HRU's with areas less than 1 km**2
        #   Note, all desired spatial datasets must be added (except for split) before using this function.
        #   If not, this process will be reversed.
        hrus.simplify_by_area(1E6)

        # Add aspect only as an attribute using zonal stats
        hrus.add_aspect(only_zonal=True)

        # Add slope only as an attribute using zonal stats
        hrus.add_slope(only_zonal=True)

        # Add landcover by splitting spatial HRU's, and do not include covers with areas < 1 km**2.
        # NOTE, splitting must be done only after adding all spatial and zonal datasets, because those
        #   values are used when repeating HRU's. This will be reversed if those functions are called again.
        #   Also, any of the "simplify" functions must be called prior to using split.
        hrus.split('landcover.shp', 'COVER', vector_attribute='COVER_CLASS', minimum_areaa=1E6))

        # Any raster data that are added as arguements may include a correlation dictionary to dictate
        #   what the output (.rvh, or .csv) names are, for example:
        # hrus.add_zonal_data('landcover.tif', summary_method='mode', dataset_interpolation='nearest',
        #                     correlation_dict={1: 'Trees', 2: 'Grassland', 3: 'water', 4: 'Alpine'})

        # Write to an output .rvh using a template
        hrus.write_raven_rvh('template_file.rvh', 'output_file.rvh')
    """
    def __init__(self, dem, basin_mask=None, output_srid=4269):
        """
        HRU instance for dynamic HRU creation tasks
        :param dem: (str or Raster) Digital Elevation Model
        :param basin_mask: (str, Vector or Raster) mask to use for the overall basin. If None, it is assumed that
            the the comprises the watershed extent.
        :param output_srid: spatial reference for the output centroids
        """
        # Prepare dem using mask
        dem = Raster(dem)

        if basin_mask is not None:
            mask = assert_type(basin_mask)(basin_mask)
            if isinstance(mask, Raster):
                # Reduce the DEM to the necessary data
                mask = mask.match_raster(dem)
                m = mask.array
                d = dem.array
                d[m == mask.nodata] = dem.nodata
                dem = dem.empty()
                dem[:] = d
                self.dem = dem.clip_to_data()
            else:
                # Clip the dem using a polygon
                self.dem = dem.clip(mask)

        else:
            self.dem = dem

        self.mask = self.dem.array != self.dem.nodata

        self.srid = output_srid

        self.wkdir = os.path.dirname(self.dem.path)
        self.spatialData = {}
        self.zonalData = {}
        self.hrus = self.dem.full(0).astype('uint64')
        self.hrus.nodataValues = [0]

        self.regen_spatial = True  # Flag to check if regeneration necessary
        self.regen_zonal = True

    def collect_input_data(self, dataset, vector_attribute, dataset_interpolation):
        """
        INTERNAL method used to prepare input datasets
        :param dataset:
        :return:
        """
        data = assert_type(dataset)(dataset)
        if isinstance(data, Vector) and vector_attribute is None:
            raise HruError('If a Vector is used to add spatial data, an attribute field name must be specified')

        # A correlation dictionary may be generated
        correlation_dict = None

        # Rasterize or align the input data
        if isinstance(data, Vector):
            rasterized_data = data.rasterize(self.dem, vector_attribute)
            if isinstance(rasterized_data, tuple):
                # A correlation dict was returned because the field was text
                ds, correlation_dict = rasterized_data
            else:
                ds = rasterized_data
        else:
            data.interpolationMethod = dataset_interpolation
            ds = data.match_raster(self.dem)

        return ds, correlation_dict

    def add_spatial_data(self, dataset, name, summary_method='mean', interval=0, number=0, bins=[],
                         dataset_interpolation='bilinear', vector_attribute=None, correlation_dict=None):
        """
        Split spatial HRU's using a dataset and zones
        If the bins argument is used, it will override the other interval argumants.
        Similarly, if the number argument is not 0 it will override the interval argument.
        If neither of bins, interval, or number are specified, the discrete values will be used as regions.
        :param dataset: Vector or Raster
        :param name: Name to be used for output HRU's
        :param summary_method: Method used to summarize original data within bins
        :param interval: float: Interval in units to divide into HRU's
        :param number: Number of regions to split the dataset into
        :param bins: Manual bin edges used to split the dataset into regions
        :param dataset_interpolation: Method used to interpolate the dataset
        :param vector_attribute: Attribute field to use for data values if the dataset is a Vector
        :param correlation_dict: dictionary used to correlate the attributed value with text
        :return: None
        """
        # Check arguments
        summary_method = str(summary_method).lower()
        if summary_method not in ['mean', 'mode', 'min', 'max', 'std']:
            raise HruError("Invalid summary method {}".format(summary_method))

        # Add to spatial datasets and add original to zonal datasets
        if name in self.spatialData.keys():
            print "Warning: Existing spatial dataset {} will be overwritten".format(name)
        if name in self.zonalData.keys():
            print "Warning: Existing zonal dataset {} will be overwritten".format(name)

        ds, new_c_dict = self.collect_input_data(dataset, vector_attribute, dataset_interpolation)
        if correlation_dict is None:
            correlation_dict = new_c_dict

        # Read data and create mask
        spatial_data = ds.array
        data_mask = (spatial_data != ds.nodata) & self.mask & ~numpy.isnan(spatial_data) & ~numpy.isinf(spatial_data)
        a = spatial_data[data_mask]
        spatial_data = numpy.full(spatial_data.shape, 0, 'uint64')

        # Digitize
        digitize = True
        if len(bins) != 0:
            pass
        elif number > 0:
            bins = numpy.linspace(a.min(), a.max(), number + 1)
        elif interval > 0:
            # Snap upper and lower bounds to interval
            lower = a.min()
            lower = lower - (lower % interval)
            lower -= interval / 2.
            upper = a.max()
            _ceil = interval - (upper % interval)
            if _ceil == interval:
                _ceil = 0
            upper += _ceil
            upper += interval / 2.
            bins = numpy.linspace(lower, upper, int((upper - lower) / interval) + 1)
        else:
            # Use discrete values
            digitize = False
            spatial_data[data_mask] = a

        if digitize:
            spatial_data[data_mask] = numpy.digitize(a, bins) + 1

        # Update spatial HRU datasets with labeled data and original data
        out = self.hrus.empty()
        out[:] = label(spatial_data)

        self.spatialData[name] = out
        self.zonalData[name] = (ds, summary_method, correlation_dict)

        self.regen_spatial = True
        self.regen_zonal = True

        print "Successfully added {} to spatial data".format(name)

    def add_zonal_data(self, dataset, name, summary_method='mean',
                       dataset_interpolation='bilinear', vector_attribute=None, correlation_dict=None):
        """
        Prepare a dataset for zonal statistics while creating HRUs
        :param dataset: Instance of the Raster class
        :param name: Name of the dataset to be used in the HRU set
        :param summary_method: Statistical method to be applied
        :param dataset_interpolation: Method used to interpolate the dataset
        :param vector_attribute: Attribute field to use for data values if the dataset is a Vector
        :param correlation_dict: dictionary used to correlate the attributed value with text
        :return: None
        """
        summary_method = str(summary_method).lower()
        if summary_method not in ['mean', 'mode', 'min', 'max', 'std']:
            raise HruError("Invalid summary method {}".format(summary_method))

        if name in ['Area', 'Centroid']:
            raise HruError("Name cannot be 'Area' or 'Centroid', as these are used when writing HRU's.")

        if name in self.zonalData.keys():
            print "Warning: Existing zonal dataset {} will be overwritten".format(name)

        ds, new_c_dict = self.collect_input_data(dataset, vector_attribute, dataset_interpolation)
        if correlation_dict is None:
            correlation_dict = new_c_dict

        a = ds.array
        a[~self.mask] = ds.nodata
        ds[:] = a

        # Add to spatial datasets
        self.zonalData[name] = (ds, summary_method, correlation_dict)

        self.regen_zonal = True

        print "Successfully added {} to zonal data".format(name)

    def build_spatial_hrus(self):
        """
        Create HRU set using spatial data
        :return: None
        """
        if len(self.spatialData) == 0:
            raise HruError('No spatial datasets have been added yet')

        # Iterate spatial datasets and create HRUs
        names = self.spatialData.keys()
        hrua = numpy.zeros(shape=self.hrus.shape, dtype='uint64')
        for name in names[:-1]:
            print "Splitting by {}".format(name)
            a = self.spatialData[name].array
            m = a != 0
            hrua[m] = hrua[m] + a[m] + hrua.max()
            hrua = label(hrua)

        # Add last dataset separately in order to create map
        name = names[-1]
        print "Splitting by {}".format(name)
        a = self.spatialData[name].array
        m = a != 0
        hrua[m] = hrua[m] + a[m] + hrua.max()
        self.hrus[:], self.hru_map = label(hrua, return_map=True)

        print "{} spatial HRU's built".format(len(self.hru_map))

        self.regen_spatial = False
        self.regen_zonal = True

    def split(self, dataset, name, vector_attribute=None, dataset_interpolation='nearest',
              correlation_dict=None, minimum_area=0, exclude_from_area_filter=[]):
        """
        Split existing hru's into more using coverage of another dataset
        :param dataset: Vector or Raster
        :param name: Name for dataset in header
        :param vector_attribute: name of the attribute field to use if the dataset is a vector
        :param dataset_interpolation: Interpolation method to use for raster resampling
        :param correlation_dict: Raster attribute table dictionary
        :param minimum_area: minimum threshold area to disclude HRU's
        :param exclude_from_area_filter: List of names that will not be removed with the area filter
        :return: None
        """
        def collect_name_attr(d):
            try:
                return correlation_dict[d]
            except KeyError:
                raise KeyError('The value {} does not exist in the correlation '
                               'dictionary for {}'.format(data, name))

        if self.regen_spatial:
            self.build_spatial_hrus()
        if self.regen_zonal:
            self.compute_zonal_data()  # Only self.hru_attributes are used

        print "Creating additional HRU's based on {}...".format(name)

        ds, new_c_dict = self.collect_input_data(dataset, vector_attribute, dataset_interpolation)
        if correlation_dict is None:
            correlation_dict = new_c_dict

        a = ds.array
        nd = ds.nodata

        new_hrus = {}
        cnt = -1

        for id, ind in self.hru_map.iteritems():
            data = a[ind]
            data = data[(data != nd) & ~numpy.isinf(data) & ~numpy.isnan(data)]

            # No data here, simply record an HRU with [None] for this attribute
            if data.size == 0:
                cnt += 1
                new_hrus[cnt] = {key: val for key, val in self.hru_attributes[id].iteritems()}
                new_hrus[cnt].update({name: '[None]', 'MAP_HRU': id})
                continue

            # Split data into unique values with respective areas (converted to proportions)
            data, areas = numpy.unique(data, return_counts=True)
            areas = areas.astype('float32') * self.dem.csx * self.dem.csy
            areas /= areas.sum()

            # Apply minimum proportion argument
            current_area = self.hru_attributes[id]['AREA']
            keep_area = areas * current_area >= minimum_area
            # Check exclude list
            if correlation_dict is not None:
                data_names = [collect_name_attr(d) for d in data]
            else:
                data_names = data
            keep_area = keep_area | [d in exclude_from_area_filter for d in data_names]
            # If all types are below the proportion use the dominant type
            if keep_area.size == 0:
                keep_area = numpy.zeros(areas.shape, 'bool')
                keep_area[numpy.argmax(areas)] = True
            # Filter and re-normalize
            data = data[keep_area]
            areas = areas[keep_area]
            areas /= areas.sum()

            # Create additional HRU's
            for d, area_prop in zip(data, areas):
                cnt += 1
                new_hrus[cnt] = {key: val for key, val in self.hru_attributes[id].iteritems()}
                if correlation_dict is not None:
                    d = collect_name_attr(d)
                new_hrus[cnt].update({name: d, 'AREA': current_area * area_prop, 'MAP_HRU': id})

        print "...Created {} additional HRU's based on {}".format(
            len(new_hrus) - len(self.hru_attributes), name
        )
        self.hru_attributes = new_hrus

    def compute_zonal_data(self):
        """
        Use domain.zonalData to produce zonal summary statistics for output.
        Centroids and areas are also added implicitly
        :return: dict of hru id's and the value of each column
        """
        if self.regen_spatial:
            self.build_spatial_hrus()

        methods = {'mean': numpy.mean,
                   'mode': util.mode,
                   'min': numpy.min,
                   'max': numpy.max,
                   'std': numpy.std}  # Add more as needed...

        # Rebuild HRU attributes
        self.hru_attributes = {id: {} for id in self.hru_map.keys()}

        print "Computing LONGITUDE and LATITUDE"
        self.compute_centroids()
        print "Computing AREA"
        self.compute_area()

        for name, zoneData in self.zonalData.iteritems():
            print "Computing {}".format(name)
            rast, method, corr_dict = zoneData
            a = rast.array
            nd = rast.nodata
            method = methods[method]
            for id in self.hru_map.keys():
                data = a[self.hru_map[id]]
                data = data[(data != nd) & ~numpy.isinf(data) & ~numpy.isnan(data)]
                if data.size == 0:
                    self.hru_attributes[id][name] = '[None]'
                    continue
                data = method(data)
                if method == util.mode:
                    data = data[0]
                if corr_dict is not None:
                    try:
                        data = corr_dict[data]
                    except KeyError:
                        raise KeyError('The value {} does not exist in the correlation '
                                       'dictionary for {}'.format(data, name))
                self.hru_attributes[id][name] = data

        self.regen_zonal = False

    def write_raven_rvh(self, template_file, output_name):
        """
        Write an .rvh file to be used in the Raven Hydrological Modal
        :param template_file: Path to a file to use as a template to write an output .rvh file
        :param output_name: path to output file
        :return: None
        """
        # Create HRUs and add data if needed
        if self.regen_spatial:
            self.build_spatial_hrus()
        if self.regen_zonal:
            self.compute_zonal_data()

        potential_order = ['AREA', 'ELEVATION', 'LATITUDE', 'LONGITUDE', 'BASIN_ID', 'LAND_USE_CLASS', 'VEG_CLASS',
                           'SOIL_PROFILE', 'AQUIFER_PROFILE', 'TERRAIN_CLASS', 'SLOPE', 'ASPECT']

        # TODO: Incorporate order or static headings mapping into writing of .rvh

        # Read template
        with open(template_file, 'r') as f:
            lines = f.readlines()
        with open(output_name, 'w') as out:
            w = False
            for line in lines:
                if ':HRUs' in line:
                    out.write(line)
                    w = True
                    continue
                if w:
                    keys = self.hru_attributes[self.hru_attributes.keys()[0]].keys()
                    write = ['  :Attributes,ID'] + map(str, keys)
                    out.write(','.join(write) + '\n')
                    out.write('  :Units <-- manually enter units -->\n')
                    for hru in range(1, max(self.hru_attributes.keys()) + 1):
                        write = ','.join(map(str, [hru] + [self.hru_attributes[hru][key] for key in keys]))
                        out.write(write + '\n')
                    out.write(':EndHRUs')
                    break
                else:
                    out.write(line)

        print "Successfully wrote output file {}".format(output_name)

    def write_csv(self, output_name):
        """
        Write the HRU's to a .csv
        :param output_name: path to output csv
        :return: None
        """
        if self.regen_spatial:
            self.build_spatial_hrus()
        if self.regen_zonal:
            self.compute_zonal_data()

        keys = self.hru_attributes[self.hru_attributes.keys()[0]].keys()
        with open(output_name, 'wb') as f:
            f.write(','.join(['HRU_ID'] + keys) + '\n')
            for hru in range(1, max(self.hru_attributes.keys()) + 1):
                write = ','.join(map(str, [hru] + [self.hru_attributes[hru][key] for key in keys]))
                f.write(write + '\n')

        print "Successfully wrote output csv {}".format(output_name)

    def add_elevation(self, interval=100, number=0, bins=[], only_zonal=False):
        """
        Add elevation bands to the zonal data, or both zonal and spatial data
        :param interval: see add_spatial_data
        :param number: see add_spatial_data
        :param bins: see add_spatial_data
        :param only_zonal: Only add elevation to the zonal datasets
        :return: None
        """
        # Create elevation bands
        if only_zonal:
            self.add_zonal_data(self.dem, 'ELEVATION')
        else:
            self.add_spatial_data(self.dem, 'ELEVATION', interval=interval, number=number, bins=bins)

    def add_aspect(self, interval=0, number=4, bins=[], only_zonal=False):
        """
        Compute aspect and add to spatial HRU set
        :param interval: see add_spatial_data
        :param number: see add_spatial_data
        :param bins: see add_spatial_data
        :return: None
        """
        # Compute aspect and add to HRU set
        if only_zonal:
            self.add_zonal_data(topo(self.dem).aspect(), 'ASPECT', 'mode')
        else:
            self.add_spatial_data(topo(self.dem).aspect(), 'ASPECT', interval=interval, number=number, bins=bins)

    def add_slope(self, interval=0, number=4, bins=[], only_zonal=False):
        """
        Compute slope and add to spatial HRU set
        :param interval: see add_spatial_data
        :param number: see add_spatial_data
        :param bins: see add_spatial_data
        :return: None
        """
        # Compute aspect and add to HRU set
        if only_zonal:
            self.add_zonal_data(topo(self.dem).slope(), 'SLOPE')
        else:
            self.add_spatial_data(topo(self.dem).slope(), 'SLOPE', interval=interval, number=number, bins=bins)

    def simplify_by_area(self, min_area):
        """
        Remove spatial HRU's with areas below the specified min_area
        :param min_area: Minimum area in domain units to remove HRU's
        :return: None
        """
        if self.regen_spatial:
            self.build_spatial_hrus()

        a = self.hrus.array
        cnt = 0
        for id, inds in self.hru_map.iteritems():
            area = inds[0].size * self.dem.csx * self.dem.csy

            if area < min_area:
                # Apply no data to the hrus
                cnt += 1
                a[inds] = self.hrus.nodata

        # Interpolate the newly formed gaps with the neighbours
        self.hrus[:] = a
        self.hrus = interpolate_nodata(self.hrus)

        # Apply mask and relabel
        a = self.hrus.array
        a[~self.mask] = self.hrus.nodata
        self.hrus[:], self.hru_map = label(a, return_map=True)

        print "{} HRU's below {} [units] removed".format(cnt, min_area)

    def simplify_by_filter(self, iterations):
        """
        Remove small segments of HRU's.  Applies an iterative mode filter.
        :param iterations: Number of iterations to smooth dataset
        :return: None
        """
        if self.regen_spatial:
            self.build_spatial_hrus()

        previous = max(self.hru_map.keys())

        for i in range(iterations):
            print "Performing filter {} of {}".format(i + 1, iterations)
            self.hrus = most_common(self.hrus)

        self.hrus[:], self.hru_map = label(self.hrus.array, return_map=True)

        print "HRU count reduced from {} to {}".format(previous, max(self.hru_map.keys()))

    def compute_centroids(self):
        """
        Compute the centre of mass centroid of a specific HRU
        :param id: hru id
        :return: None
        """
        # Change to output srid
        insr = osr.SpatialReference()
        insr.ImportFromWkt(self.dem.projection)
        outsr = osr.SpatialReference()
        outsr.ImportFromEPSG(self.srid)
        coordTransform = osr.CoordinateTransformation(insr, outsr)
        for id, inds in self.hru_map.iteritems():
            # Centre of mass in spatial reference of dem
            y = self.dem.top - ((numpy.mean(inds[0]) + 0.5) * self.dem.csy)
            x = self.dem.left + ((numpy.mean(inds[1]) + 0.5) * self.dem.csx)
            x, y, _ = coordTransform.TransformPoint(x, y)
            self.hru_attributes[id]['LONGITUDE'] = x
            self.hru_attributes[id]['LATITUDE'] = y

    def compute_area(self):
        """
        Compute area in the units of the dem spatial reference
        :param id: HRU index
        :return: None
        """
        for id, inds in self.hru_map.iteritems():
            self.hru_attributes[id]['AREA'] = inds[0].size * self.dem.csx * self.dem.csy

    def save_hru_raster(self, output_name):
        """
        Save the current HRU set as a Raster
        :param output_name: name of the output Raster
        :return: None
        """
        # Create HRUs and add data if needed
        if self.regen_spatial:
            self.build_spatial_hrus()

        if output_name.split('.')[-1].lower() != 'tif':
            output_name += '.tif'
        self.hrus.save(output_name)

    def __repr__(self):
        if self.regen_spatial:
            write = 'Uncomputed HRU instance comprised of the following spatial datasets:\n'
            write += '\n'.join(self.spatialData.keys()) + '\n'
            write += 'And the following zonal datasets:\n'
        else:
            write = "HRU instance with {} spatial HRU's, and the following zonal datasets (which have {}" \
                    "been computed):\n".format(max(self.hru_map.keys()), 'not ' if self.regen_zonal else '')
        write += '\n'.join(['{} of {}'.format(method[1], name)
                            for name, method in self.zonalData.iteritems()])
        return write


class riparian(object):
    """Objects and methods for the delineation and calculation of sensitivity of the riparian"""
    def __init__(self, dem):
        self.dem = Raster(dem)
        self.dem.interpolationMethod = 'bilinear'

        self.update_region = False  # Used to track changes in the riparian delineation

    def smooth_dem(self, sigma=2):
        """Use a gaussian filter with the specified sigma to smooth the DEM if it is coarse"""
        self.dem = gaussian(self.dem, sigma=sigma)
        self.dem.interpolationMethod = 'bilinear'

    def generate_streams(self, minimum_contributing_area):
        if minimum_contributing_area is None:
            minimum_contributing_area = 1E6  # Default is 1km2
        if not hasattr(self, 'fa'):
            print "Calculating flow accumulation"
            self.fa = bluegrass.watershed(self.dem, flow_direction='MFD', positive_fd=False, change_nodata=False)[1]

        self.streams = bluegrass.stream_extract(self.dem, minimum_contributing_area=minimum_contributing_area,
                                                accumulation=self.fa.path)

    def calculate_width(self):
        """
        Calculate the width of the riparian within the buffer
        :return:
        """
        if not hasattr(self, 'region'):
            self.delineate_using_topo()

        # Calculate distance to streams
        print "Creating distance transform"
        d = distance(self.streams)

        # Update to include only values on outer edges
        a = d.array
        m = self.region.array
        m = m & binary_dilation(~m, numpy.ones((3, 3)))
        a[~m] = 0
        d[:] = a
        d.nodataValues = [0]

        # Interpolate throughout region
        self.width = interpolate_mask(d, self.region, 'idw')

    def delineate_using_topo(self, reclass_percentile=6, minimum_contributing_area=None,
                             streams=None, scale_by_area=0):
        """
        Delineate the riparian using only terrain
        :param minimum_contributing_area:
        :param reclass_percentile:
        :param streams: stream data source
        :param scale_by_area: (float) Scale the cost using contributing area as a proportion
        :return: None
        """
        if streams is not None:
            streams = assert_type(streams)(streams)
            if isinstance(streams, Vector):
                print "Rasterizing streams"
                self.streams = streams.rasterize(self.dem)
            else:
                print "Matching stream Raster to study area"
                self.streams = streams.match_raster(self.dem)
        elif not hasattr(self, 'streams'):
            print "Delineating streams"
            self.generate_streams(minimum_contributing_area)

        if not hasattr(self, 'cost'):
            print "Calculating cost surface"
            self.cost = normalize(cost_surface(self.streams, topo(self.dem).slope()))

        if scale_by_area:
            if not hasattr(self, 'fa'):
                print "Calculating flow accumulation"
                self.fa = bluegrass.watershed(self.dem, flow_direction='MFD', positive_fd=False, memory_manage=True)[1]

            print "Scaling cost using contributing area"

            # Get rid of nans
            fa = self.fa.copy()
            for a, s in fa.iterchunks():
                a[numpy.isnan(a) | numpy.isinf(a) | (a == fa.nodata)] = numpy.finfo('float32').min
                fa[s] = a
            fa.nodataValues = [numpy.finfo('float32').min]

            # Dilate contributing area and scale
            cont_area = interpolate_nodata(normalize(inverse((fa * (fa.csx * fa.csy)).clip(self.streams))))
            m, b = numpy.linalg.solve([[0, 1], [1, 1]], [1 - scale_by_area, 1.])
            cost = self.cost * (cont_area * m + b)

        else:
            cost = self.cost

        print "Clipping to region"
        a = cost.array
        m = a != cost.nodata
        p = numpy.percentile(a[m], reclass_percentile)
        self.region = cost.astype('bool')
        self.region[:] = m & (a <= p)
        self.region.nodataValues = [0]

        self.update_region = True

    def delineate_using_sensitivity(self):
        pass

    def create_sensitivity_zones(self, breaks='percentile', percentiles=(33.3, 66.7)):
        if not hasattr(self, 'sensitivity'):
            self.update_sensitivity()

        a = self.sensitivity.array
        m = a != self.sensitivity.nodata

        # Collect breakpoints
        if breaks.lower() == 'percentile':
            p1 = numpy.percentile(a[m], percentiles[0])
            p2 = numpy.percentile(a[m], percentiles[1])

        elif breaks.lower() == 'jenks':
            import jenkspy
            breaks = jenkspy.jenks_breaks(a[m], nb_class=3)
            p1, p2 = breaks[1], breaks[2]

        elif breaks.lower() == 'equal':
            breaks = numpy.linspace(a[m].min(), a[m].max(), 4)
            p1, p2 = breaks[1], breaks[2]

        zones = numpy.full(a.shape, 0, 'uint8')

        zones[m & (a <= p1)] = 1
        zones[m & (a > p1) & (a <= p2)] = 2
        zones[m & (a > p2)] = 3

        self.sensitivity_zones = self.sensitivity.astype('uint8')
        self.sensitivity_zones.nodataValues = [0]
        self.sensitivity_zones[:] = zones

    def update_sensitivity(self, cost_weight=2):
        """
        Update the sensitivity surface within the buffer
        :return:
        """
        if not hasattr(self, 'region'):
            self.delineate_using_topo()

        if not hasattr(self, 'sinuosity') or self.update_region:
            print "Calculating sinuosity"
            self.sinuosity = interpolate_mask(channel_density(self.streams), self.region, 'idw')

        if not hasattr(self, 'channel_slope') or self.update_region:
            print "Calculating channel slope"
            self.channel_slope = interpolate_mask(stream_slope(self.dem, self.streams), self.region, 'idw')

        if not hasattr(self, 'contributing_area') or self.update_region:
            print "Calculating contributing area"
            if not hasattr(self, 'fa'):
                self.fa = bluegrass.watershed(self.dem, memory_manage=True)[1]
            a = self.fa.array
            # Sometimes the no data values is nan for flow accumulation
            a[numpy.isnan(a) | (a == self.fa.nodata)] = numpy.finfo('float32').min
            fa = self.fa.empty()
            fa.nodataValues = [numpy.finfo('float32').min]
            a[self.streams.array == self.streams.nodata] = fa.nodata
            fa[:] = a
            self.contributing_area = interpolate_mask(fa, self.region, 'idw')

        # TODO: Add land cover analysis here (coarsewood recruitment, shade, litter, root strength, etc.)
        # i.e. cumulative_effectiveness(canopy, landcover)

        print "Aggregating sensitivity parameters"
        region = self.region.array

        # Create sensitivity from region and cost
        cost = self.cost.empty()
        sensitivity = self.cost.array
        sensitivity[~region] = cost.nodata
        cost[:] = sensitivity

        sensitivity = inverse(normalize(cost)).array * cost_weight
        sensitivity[~region] = -9999

        modals = region.astype('uint8') * cost_weight

        # Normalize and invert stream slope
        # a = (self.channel_slope / 90).array
        ch_sl = normalize(inverse(self.channel_slope))
        a = ch_sl.array
        m = (a != ch_sl.nodata) & region
        sensitivity[m] += a[m]
        modals += m

        # Use sinuosity directly
        sinu = normalize(self.sinuosity)
        a = sinu.array
        m = (a != sinu.nodata) & region
        sensitivity[m] += a[m]
        modals += m

        # Normalize contributing area using y = 5.7E-05x, where y is the width and x is the contributing area
        if not hasattr(self, 'width'):
            self.calculate_width()
        width_ratio = normalize((self.contributing_area * 5.7E-05) / self.width)
        a = width_ratio.array
        m = (a != width_ratio.nodata) & region
        a = a[m]
        a[a > 1] = 1
        print "Min width ratio: {}\nMax width ratio: {}\nMean width ratio: {}".format(a.min(), a.max(), a.mean())
        sensitivity[m] += a
        modals += m

        # Divide by modals and fill in nodata values
        m = modals != 0
        sensitivity[m] /= modals[m]

        # Create output
        self.sensitivity = self.dem.astype('float32')
        self.sensitivity.nodataValues = [-9999]
        self.sensitivity[:] = sensitivity

        self.update_region = False

    def cumulative_effectiveness(self, canopy_height, landcover=None):
        """
        Define zones of riparian cumulative effectiveness.
        Updated Oct 21, 2017
        """
        # Compute distance to streams
        dist = distance(stream_raster)
        distA = dist.array
        # Create mask where distance is less than tree height and scale to index from 0-1
        m = distA < tree_height
        distA[m] = (distA[m] - distA[m].min()) / (distA[m].max() - distA[m].min())
        nodata = dist.nodata

        # Shade is equal to the inverse of dist
        a = distA.copy()
        a[m] = 1. - a[m]
        a[~m] = nodata
        shade = dist.empty()
        shade[:] = a

        # Litter
        a = distA.copy()
        litter_mask = m & (a <= 0.6)
        a[litter_mask] /= 0.6
        a[litter_mask] = (1. - a[litter_mask])
        a[~litter_mask] = nodata
        litter = dist.empty()
        litter[:] = a

        # Coarse
        coarse = shade.copy()

        # Root
        a = distA.copy()
        a = dist.array
        root_mask = m & (a >= 0.25)
        a[root_mask] = 1.
        a[~root_mask] = nodata
        root = dist.empty()
        root[:] = a

        return root, litter, shade, coarse

    def save(self, dir_path):
        if os.path.isdir(dir_path):
            raise Exception("The directory {} already exists".format(dir_path))
        os.mkdir(dir_path)
        for key, attr in self.__dict__.iteritems():
            if isinstance(attr, Raster):
                attr.save(os.path.join(dir_path), '{}.h5'.format(key))

    def load(self, dir_path):
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        self.__dict__.update({os.path.basename(f).split('.')[0]: Raster(f) for f in files})

    def __repr__(self):
        return "Riparian delineation and sensitivity instance with:\n" + '\n'.join(self.__dict__.keys())


def segment_water(dem, slope_threshold=0, slope=None):
    """
    Segment lakes from a dem using slope
    :param dem:
    :param filter:
    :return:
    """
    if slope is None:
        slope = topo(dem).slope()
    else:
        slope = Raster(slope)

    labels = label(slope <= slope_threshold, True)[1]

    # Create an output dataset
    water = slope.astype('bool').full(0)
    water.nodataValues = [0]
    outa = numpy.zeros(shape=water.shape, dtype='bool')

    # Iterate labels and isolate sinks
    print "Identified {} potential waterbodies".format(len(labels))
    cnt = 0
    for id, inds in labels.iteritems():
        cnt += 1
        outa[inds] = 1

    print "Filtered to {} waterbodies".format(cnt)
    water[:] = outa

    return water


def bankfull(dem, average_annual_precip=250, contributing_area=None, flood_factor=3,
             streams=None, min_stream_area=None):
    """
    Calculate a bankfull depth using the given precipitation and flood factor
    :param dem: Input elevation Raster
    :param average_annual_precip: Average annaul precipitation (cm) as a scalar, Vector, or Raster
    :param contributing_area: A contributing area (km**2) Raster. It will be calculated using the DEM if not provided.
    :param flood_factor: Coefficient to amplify the bankfull depth
    :param streams: Input stream Vector or Raster.  They will be calculated using the min_stream_area if not provided
    :param min_stream_area: If no streams are provided, this is used to derived streams.  Units are m**2
    :return: Raster instance of the bankful depth
    """
    dem = Raster(dem)

    # Grab the streams
    if streams is not None:
        streams = assert_type(streams)(streams)
        if isinstance(streams, Vector):
            streams = streams.rasterize(dem)
        elif isinstance(streams, Raster):
            streams = streams.match_raster(dem)
    else:
        if min_stream_area is None:
            raise WaterError('Either one of streams or minimum stream contributing area must be specified')
        streams = bluegrass.stream_extract(dem, min_stream_area)

    streams = streams.array != streams.nodata

    # Check if contributing area needs to be calculated
    if contributing_area is None:
        contrib = bluegrass.watershed(dem)[1] * (dem.csx * dem.csy / 1E6)  # in km**2
    else:
        contrib = Raster(contributing_area)

    # Parse the precip input and create the precip variable
    if any([isinstance(average_annual_precip, t) for t in [int, float, numpy.ndarray]]):
        # Scalar or array
        precip = dem.full(average_annual_precip) ** 0.355
    else:
        precip = assert_type(average_annual_precip)(average_annual_precip) ** 0.355

    # Calculate bankfull depth
    bankfull = (contrib ** 0.280) * 0.196
    bankfull = bankfull * precip
    bankfull = bankfull ** 0.607 * 0.145
    bankfull *= flood_factor

    # Add the dem to the bankfull depth where streams exists, and extrapolate outwards
    bnkfl = bankfull.array
    bnkfl[~streams] = bankfull.nodata
    bankfull[:] = bnkfl
    del bnkfl
    bankfull += dem
    bankfull = interpolate_nodata(bankfull)

    # Smooth using a mean filter 3 times
    for i in range(3):
        bankfull = mean_filter(bankfull)

    # Create a flood depth by subtracting the dem
    bankfull -= dem
    bnkfl = bankfull.array
    bnkfl[bnkfl < 0] = bankfull.nodata
    bankfull[:] = bnkfl

    return bankfull


def valley_confinement(dem, min_stream_area, cost_threshold=2500, streams=None, waterbodies=None,
                       average_annual_precip=250, slope_threshold=9, use_flood_option=True, flood_factor=3,
                       max_width=False, minimum_drainage_area=0, min_stream_length=100, min_valley_bottom_area=10000):
    """
     Valley Confinement algorithm based on https://www.fs.fed.us/rm/pubs/rmrs_gtr321.pdf
    :param dem: (Raster) Elevation Raster
    :param min_stream_area: (float) Minimum contributing area to delineate streams if they are not provided.
    :param cost_threshold: (float) The threshold used to constrain the cumulative cost of slope from streams
    :param streams: (Vector or Raster) A stream Vector or Raster.
    :param waterbodies: (Vector or Raster) A Vector or Raster of waterbodies. If this is not provided, they will be segmented from the DEM.
    :param average_annual_precip: (float, ndarray, Raster) Average annual precipitation (in cm)
    :param slope_threshold: (float) A threshold (in percent) to clip the topographic slope to.  If False, it will not be used.
    :param use_flood_option: (boolean) Determines whether a bankfull flood Extent will be used or not.
    :param flood_factor: (float) A coefficient determining the amplification of the bankfull
    :param max_width: (float) The maximum valley width of the bottoms.
    :param minimum_drainage_area: (float) The minimum drainage area used to filter streams (km**2).
    :param min_stream_length: (float) The minimum stream length (m) used to filter valley bottom polygons.
    :param min_valley_bottom_area: (float) The minimum area for valey bottom polygons.
    :return: Raster instance (of the valley bottom)
    """
    # Create a Raster instance from the DEM
    dem = Raster(dem)

    # The moving mask is a mask of input datasets as they are calculated
    moving_mask = numpy.zeros(shape=dem.shape, dtype='bool')

    # Calculate slope
    print "Calculating topographic slope"
    slope = topo(dem).slope('percent_rise')

    # Add slope to the mask
    if slope_threshold is not False:
        moving_mask[(slope <= slope_threshold).array] = 1

    # Calculate cumulative drainage (flow accumulation)
    fa = bluegrass.watershed(dem)[1]
    fa.mode = 'r+'
    fa *= fa.csx * fa.csy / 1E6

    # Calculate streams if they are not provided
    if streams is not None:
        streams = assert_type(streams)(streams)
        if isinstance(streams, Vector):
            streams = streams.rasterize(dem)
        elif isinstance(streams, Raster):
            streams = streams.match_raster(dem)
    else:
        streams = bluegrass.stream_extract(dem, min_stream_area)

    # Remove streams below the minimum_drainage_area
    if minimum_drainage_area > 0:
        a = streams.array
        a[fa < minimum_drainage_area] = streams.nodata
        streams[:] = a

    # Calculate a cost surface using slope and streams, and create a mask using specified percentile
    print "Calculating cost"
    cost = cost_surface(streams, slope)
    moving_mask = moving_mask & (cost < cost_threshold).array

    # Incorporate max valley width arg
    if max_width is not False:  # Use the distance from the streams to constrain the width
        # Calculate width if necessary
        moving_mask = moving_mask & (distance(streams) < (max_width / 2)).array

    # Flood calculation
    if use_flood_option:
        print "Calculating bankfull"
        flood = bankfull(dem, streams=streams, average_annual_precip=average_annual_precip,
                         contributing_area=fa, flood_factor=flood_factor).mask
        moving_mask = moving_mask & flood.array

    # Remove waterbodies
    # Segment water bodies from the DEM if they are not specified in the input
    print "Removing waterbodies"
    if waterbodies is not None:
        waterbodies = assert_type(waterbodies)(waterbodies)
        if isinstance(waterbodies, Vector):
            waterbodies = waterbodies.rasterize(dem)
        elif isinstance(waterbodies, Raster):
            waterbodies = waterbodies.match_raster(dem)
    else:
        waterbodies = segment_water(dem, slope=slope)
    moving_mask[waterbodies.array] = 0

    # Create a Raster from the moving mask and run a mode filter
    print "Applying a mode filter"
    valleys = dem.astype('bool')
    valleys[:] = moving_mask
    valleys.nodataValues = [0]
    valleys = most_common(valleys)

    # Label the valleys and remove those below the specified area or where stream lenght is too small
    print "Filtering by area and stream length"
    stream_segment = numpy.mean([dem.csx, dem.csy, numpy.sqrt(dem.csx**2 + dem.csy**2)])
    valley_map = label(valleys, True)[1]
    a = numpy.zeros(shape=valleys.shape, dtype='bool')
    sa = streams.array
    for _, inds in valley_map.iteritems():
        length = (sa[inds] != streams.nodata).sum() * stream_segment
        if inds[0].size * dem.csx * dem.csy >= min_valley_bottom_area and length >= min_stream_length:
            a[inds] = 1

    # Write to output and return a Raster instance
    valleys[:] = a
    print "Completed successfully"
    return valleys
