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


class hruError(Exception):
    pass


def wetness(dem, minimum_area):
    """
    Calculate a wetness index using streams of a minimum contributing area
    :param dem: dem (raster)
    :param minimum_area: area in units^2
    :return: raster instance
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
        fd = raster(fd)
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
    dem = raster(dem)
    dem.interpolationMethod = 'bilinear'

    with dem.match_raster(streams) as dem:
        elev = dem.array
    strms = raster(streams)
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

    streams: a streams raster
    min_contrib_area: minimum contributing area to define streams
    slope: a slope surface used to control the region delineation
    :param stream_slope:
    :param slope_thresh:
    :param stream_slope_thresh:
    :return:
    """
    # Get or compute necessary datasets
    strslo = raster(stream_slope)
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
    :param streams: stream raster
    :param sample_distance: distance to sample density
    :return: raster instance
    """
    # Allocate output as a raster cast to 32-bit floating points
    streams = raster(streams)

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
    :param kwargs: dem=path to dem _or_ stream_order=path to strahler stream order raster
        distance=search distance to calculate sinuosity ratio
    :return: sinuosity as a ratio

    Updated October 25, 2017
    """
    # Collect as raster of streams
    stream_order = raster(stream_order)
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
    out = sklabel(loadout, connectivity=2, background=bas.nodata[0],
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


class hru(object):
    """
    Create a model domain instance that is a child of the raster class
    Example:
        >>> # Create an instance of the hru class, and call is "hrus"
        >>> hrus = hru('path_to_dem.tif', 'path_to_mask.shp')  # Note, the mask may be a raster or a vector
        >>> # Add elevation as a spatial discretization dataset, and split it using an interval of 250m
        >>> hrus.add_elevation(250)
        Successfully added ELEVATION to spatial data
        >>> # Split HRU's into 4 classes of solar radiation, calling the attribute "SOLRAD"
        >>> hrus.add_spatial_data('solar_radiation.tif', 'SOLRAD', number=4)
        Successfully added SOLRAD to spatial data
        >>> # Add landcover as an attribute using a vector file and the attribute field "COVER_CLASS".  Specify a mode filter to compute zonal stats.
        >>> hrus.add_zonal_data('landcover.shp', 'COVER', 'mode', vector_attribute='COVER_CLASS')
        Successfully added COVER to zonal data
        >>> # Add aspect only as an attribute using zonal stats
        >>> hrus.add_aspect(only_zonal=True)
        Successfully added ASPECT to zonal data
        >>> # Add slope only as an attribute using zonal stats
        >>> hrus.add_slope(only_zonal=True)
        Successfully added SLOPE to zonal data
        >>> # Use a spatial mode filter 4 times to simplify the spatial HRU's
        >>> hrus.simplify(4)  # Note, the build_spatial_hrus() method must normally be called, but it is done implicitly in most cases
        Splitting by ELEVATION
        Splitting by SOLRAD
        HRU count reduced from 1246 to 424
        >>> # Write to an output file for raven
        >>> hrus.write_raven_rvh('template_file.rvh', 'output_file.rvh')  # Note, just as with spatial data, zonal data are also implicity added (compute_zonal_data() is called)
        Computing LATITUDE and LONGITUDE
        Computing AREA
        Computing ELEVATION
        Computing SOLRAD
        Computing COVER
        Computing ASPECT
        Computing SLOPE
        Successfully wrote output file output_file.rvh
        >>>
    """
    def __init__(self, dem, basin_mask, output_srid=4269):
        """
        HRU instance for dynamic HRU creation tasks
        :param dem: (str or raster) Digital Elevation Model
        :param basin_mask: (str, vector or raster) mask to use for the overall basin
        :param output_srid: spatial reference for the output centroids
        """
        # Prepare dem using mask
        dem = raster(dem)
        mask = assert_type(basin_mask)(basin_mask)
        if isinstance(mask, raster):
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

        self.mask = self.dem.array != self.dem.nodata

        self.srid = output_srid

        self.wkdir = os.path.dirname(self.dem.path)
        self.spatialData = {}
        self.zonalData = {}
        self.hrus = self.dem.full(0).astype('uint64')
        self.hrus.nodataValues = [0]

        self.regen_spatial = True  # Flag to check if regeneration necessary
        self.regen_zonal = True

    def add_spatial_data(self, dataset, name, summary_method='mean', interval=0, number=0, bins=[],
                         dataset_interpolation='bilinear', vector_attribute=None, correlation_dict=None):
        """
        Split spatial HRU's using a dataset and zones
        If the bins argument is used, it will override the other interval argumants.
        Similarly, if the number argument is not 0 it will override the interval argument.
        If neither of bins, interval, or number are specified, the discrete values will be used as regions.
        :param dataset: vector or raster
        :param name: Name to be used for output HRU's
        :param summary_method: Method used to summarize original data within bins
        :param interval: float: Interval in units to divide into HRU's
        :param number: Number of regions to split the dataset into
        :param bins: Manual bin edges used to split the dataset into regions
        :param dataset_interpolation: Method used to interpolate the dataset
        :param vector_attribute: Attribute field to use for data values if the dataset is a vector
        :param correlation_dict: dictionary used to correlate the attributed value with text
        :return: None
        """
        # Check arguments
        summary_method = str(summary_method).lower()
        if summary_method not in ['mean', 'mode', 'min', 'max', 'std']:
            raise hruError("Invalid summary method {}".format(summary_method))

        # Add to spatial datasets and add original to zonal datasets
        if name in self.spatialData.keys():
            print "Warning: Existing spatial dataset {} will be overwritten".format(name)
        if name in self.zonalData.keys():
            print "Warning: Existing zonal dataset {} will be overwritten".format(name)

        data = assert_type(dataset)(dataset)
        if isinstance(data, vector) and vector_attribute is None:
            raise hruError('If a vector is used to add spatial data, an attribute field name must be specified')

        # Rasterize or align the input data
        if isinstance(data, vector):
            ds = data.rasterize(self.dem, vector_attribute)
        else:
            ds = data.match_raster(self.dem)

        # Read data and create mask
        spatial_data = ds.array
        data_mask = (spatial_data != ds.nodata) & self.mask
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
        :param dataset: Instance of the raster class
        :param name: Name of the dataset to be used in the HRU set
        :param summary_method: Statistical method to be applied
        :param dataset_interpolation: Method used to interpolate the dataset
        :param vector_attribute: Attribute field to use for data values if the dataset is a vector
        :param correlation_dict: dictionary used to correlate the attributed value with text
        :return: None
        """
        summary_method = str(summary_method).lower()
        if summary_method not in ['mean', 'mode', 'min', 'max', 'std']:
            raise hruError("Invalid summary method {}".format(summary_method))

        if name in ['Area', 'Centroid']:
            raise hruError("Name cannot be 'Area' or 'Centroid', as these are used when writing HRU's.")

        if name in self.zonalData.keys():
            print "Warning: Existing zonal dataset {} will be overwritten".format(name)

        data = assert_type(dataset)(dataset)
        if isinstance(data, vector) and vector_attribute is None:
            raise hruError('If a vector is used to add zonal attributes, an attribute field name must be specified')

        # Rasterize or align the input data
        if isinstance(data, vector):
            ds = data.rasterize(self.dem, vector_attribute)
        else:
            ds = data.match_raster(self.dem)
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
            raise hruError('No spatial datasets have been added yet')

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

        self.regen_spatial = False

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
                data = data[data != nd]
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

        # TODO: Incorporate order into writing of .rvh

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
        with open(output_name, 'w') as f:
            f.write(','.join(['ID'] + keys) + '\n')
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

    def simplify(self, iterations):
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
        Save the current HRU set as a raster
        :param output_name: name of the output raster
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
        self.dem = raster(dem)
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
            if isinstance(streams, vector):
                print "Rasterizing streams"
                self.streams = streams.rasterize(self.dem)
            else:
                print "Matching stream raster to study area"
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
                self.fa = bluegrass.watershed(self.dem, flow_direction='MFD', positive_fd=False)[1]

            # Dilate contributing area and scale
            cont_area = normalize(inverse((self.fa * (self.fa.csx * self.fa.csy)).clip(self.streams)))
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
                self.fa = bluegrass.watershed(self.dem)[1]
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
            if isinstance(attr, raster):
                attr.save(os.path.join(dir_path), '{}.h5'.format(key))

    def load(self, dir_path):
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        self.__dict__.update({os.path.basename(f).split('.')[0]: raster(f) for f in files})

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
        slope = raster(slope)

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
    :param dem: Input elevation raster
    :param average_annual_precip: Average annaul precipitation (cm) as a scalar, vector, or raster
    :param contributing_area: A contributing area (km**2) raster. It will be calculated using the DEM if not provided.
    :param flood_factor: Coefficient to amplify the bankfull depth
    :param streams: Input stream vector or raster.  They will be calculated using the min_stream_area if not provided
    :param min_stream_area: If no streams are provided, this is used to derived streams.  Units are m**2
    :return: raster instance of the bankful depth
    """
    dem = raster(dem)

    # Grab the streams
    if streams is not None:
        streams = assert_type(streams)(streams)
        if isinstance(streams, vector):
            streams = streams.rasterize(dem)
        elif isinstance(streams, raster):
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
        contrib = raster(contributing_area)

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
    :param dem: (raster) Elevation raster
    :param min_stream_area: (float) Minimum contributing area to delineate streams if they are not provided.
    :param cost_threshold: (float) The threshold used to constrain the cumulative cost of slope from streams
    :param streams: (vector or raster) A stream vector or raster.
    :param waterbodies: (vector or raster) A vector or raster of waterbodies. If this is not provided, they will be segmented from the DEM.
    :param average_annual_precip: (float, ndarray, raster) Average annual precipitation (in cm)
    :param slope_threshold: (float) A threshold (in percent) to clip the topographic slope to.  If False, it will not be used.
    :param use_flood_option: (boolean) Determines whether a bankfull flood extent will be used or not.
    :param flood_factor: (float) A coefficient determining the amplification of the bankfull
    :param max_width: (float) The maximum valley width of the bottoms.
    :param minimum_drainage_area: (float) The minimum drainage area used to filter streams (km**2).
    :param min_stream_length: (float) The minimum stream length (m) used to filter valley bottom polygons.
    :param min_valley_bottom_area: (float) The minimum area for valey bottom polygons.
    :return: raster instance (of the valley bottom)
    """
    # Create a raster instance from the DEM
    dem = raster(dem)

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
        if isinstance(streams, vector):
            streams = streams.rasterize(dem)
        elif isinstance(streams, raster):
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
        if isinstance(waterbodies, vector):
            waterbodies = waterbodies.rasterize(dem)
        elif isinstance(waterbodies, raster):
            waterbodies = waterbodies.match_raster(dem)
    else:
        waterbodies = segment_water(dem, slope=slope)
    moving_mask[waterbodies.array] = 0

    # Create a raster from the moving mask and run a mode filter
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

    # Write to output and return a raster instance
    valleys[:] = a
    print "Completed successfully"
    return valleys
