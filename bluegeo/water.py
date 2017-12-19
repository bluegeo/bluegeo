'''
Hydrologic analysis library

Blue Geosimulation, 2017
'''
from terrain import *
from filters import *
from measurement import *
import bluegrass
from skimage.measure import label as sklabel


class WatershedError(Exception):
    pass

class hruError(Exception):
    pass


class watershed(raster):
    """
    Topographic routing and watershed delineation primarily using grass
    """
    def __init__(self, surface, tempdir=None):
        # Open and change to float if not already
        self.tempdir = tempdir
        super(watershed, self).__init__(surface)
        # Change interpolation method unless otherwise specified
        self.interpolationMethod = 'bilinear'

    def create_external_datasource(self):
        path = util.generate_name('copy', 'tif')
        self.save(path)
        return path

    def route(self):
        '''
        Return a single (D8) flow direction from 1 to 8, and positive flow
        accumulation surface
        '''
        # Save a new raster if the data source format is not gdal
        if self.format != 'gdal':
            external = self.create_external_datasource()
        else:
            external = self.path

        # Get paths for the outputs
        fd_outpath = self.generate_name('fl_dir', 'tif')
        fa_outpath = self.generate_name('fl_acc', 'tif')

        # Perform analysis using grass session
        with GrassSession(external, temp=self.tempdir):
            from grass.pygrass.modules.shortcuts import raster as graster
            from grass.script import core as grass
            graster.external(input=external, output='surface')
            grass.run_command('r.watershed', elevation='surface',
                              drainage='fd', accumulation='fa', flags='s')
            graster.out_gdal('fd', format="GTiff", output=fd_outpath)
            graster.out_gdal('fa', format="GTiff", output=fa_outpath)

        return watershed(fd_outpath, tempdir=self.tempdir), watershed(fa_outpath, tempdir=self.tempdir)

    def convergence(self, size=(11, 11), fd=None):
        '''
        Compute the relative convergence of flow vectors (uses directions 1 to
        8, which are derived from flow direction)
        '''
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

    def stream_order(self, min_contrib_area):
        '''
        Return streams with a contributing area greate than the specified
        threshold.  The resulting dataset includes strahler order for each
        stream
        '''
        # Save a new raster if the data source format is not gdal
        if self.format != 'gdal':
            external = self.create_external_datasource()
        else:
            external = self.path
        # Create output path
        str_path = self.generate_name('streams', 'tif')

        with GrassSession(external, temp=self.tempdir):
            from grass.pygrass.modules.shortcuts import raster as graster
            from grass.script import core as grass
            graster.external(input=external, output='dem')

            # Compute flow accumulation threshold based on min area
            thresh = min_contrib_area / (self.csx * self.csy)
            grass.run_command('r.stream.extract', elevation='dem',
                              threshold=thresh, stream_raster='streams',
                              direction='fd')
            grass.run_command('r.stream.order', stream_rast='streams',
                              direction='fd', strahler="strlr")
            graster.out_gdal('strlr', format="GTiff", output=str_path)

        return watershed(str_path, tempdir=self.tempdir)

    def stream_reclass(self, fa, min_contrib_area):
        '''
        Reclassify a flow accumulation dataset
        '''
        fa_rast = raster(fa)
        st = fa_rast.astype('uint8')
        st[:] = (fa_rast.array > (min_contrib_area /
                                  (st.csx * st.csy))).astype('uint8')
        st.nodataValues = [0]
        return st

    def stream_slope(self, streams, units='degrees'):
        '''
        Compute the slope from cell to cell in streams with a minimum
        contributing area.  If streams are specified, they will not be
        computed.
        '''
        with self.match_raster(streams) as dem:
            elev = dem.array
        strms = raster(streams)
        m = strms.array != strms.nodata

        # Compute stream slope
        inds = numpy.where(m)
        diag = math.sqrt(self.csx**2 + self.csy**2)
        run = numpy.array([[diag, self.csy, diag],
                           [self.csx, 1, self.csx],
                           [diag, self.csy, diag]])
        ish, jsh = self.shape

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

        output = self.empty()
        a = numpy.full(output.shape, output.nodata, output.dtype)
        slopefill = [compute(inds[0][i], inds[1][i])
                     for i in range(inds[0].shape[0])]
        a[m] = slopefill
        output[:] = a
        return output

    def alluvium(self, stream_slope, slope_thresh=6, stream_slope_thresh=5):
        '''
        Use the derivative of stream slope to determine regions of
        aggradation to predict alluvium deposition.  The input slope threshold
        is used as a cutoff for region delineation, which the stream slope
        threshold is the required stream slope to initiate deposition.
        Uses the dem as an input
        surface, and accepts (or they will be derived):

        streams: a streams raster
        min_contrib_area: minimum contributing area to define streams
        slope: a slope surface used to control the region delineation
        '''
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

    def alluvium2(self, min_dep=0, max_dep=5, benches=1, **kwargs):
        '''
        Stuff
        '''
        min_contrib_area = kwargs.get('min_contrib_area', 1E04)
        slope = kwargs.get('slope', None)
        if slope is None:
            with topo(self).slope() as slope:
                sl = slope.array
                slnd = slope.nodata
        else:
            slope = raster(slope)
            sl = slope.array
            slnd = slope.nodata
        streams = kwargs.get('streams', None)
        if streams is None:
            with self.stream_slope(
                min_contrib_area=min_contrib_area
            ) as strslo:
                sa = strslo.array
                sand = strslo.nodata
        else:
            with self.stream_slope(streams=streams) as strslo:
                sa = strslo.array
                sand = strslo.nodata
        elev = self.array

        # First phase
        sa_m = sa != sand
        sl_m = sl != slnd
        alluv = sa_m.astype('uint8')
        slope_m = sl_m & (sl >= min_dep) & (sl <= max_dep)
        labels, num = ndimage.label(slope_m, numpy.ones(shape=(3, 3),
                                                        dtype='bool'))
        # Find mean stream slopes within regions
        str_labels = numpy.copy(labels)
        str_labels[~sa_m] = 0
        un_lbl = numpy.unique(str_labels[str_labels != 0])
        min_slopes = ndimage.minimum(sa, str_labels, un_lbl)
        max_slopes = ndimage.maximum(sa, str_labels, un_lbl)
        # Find max elevation of streams within regions
        max_elev = ndimage.maximum(elev, str_labels, un_lbl)

        # Iterate stream labels and assign regions based on elevation and slope
        for i, reg in enumerate(un_lbl):
            if reg == 0:
                continue
            # Modify region to reflect stream slope variance and max elevation
            m = ((labels == reg) & (sl >= min_slopes[i]) &
                 (sl <= max_slopes[i]) & (elev < max_elev[i]))
            alluv[m] = 1

        # Remove regions not attached to a stream
        labels, num = ndimage.label(alluv, numpy.ones(shape=(3, 3),
                                                      dtype='bool'))
        alluv = numpy.zeros(shape=alluv.shape, dtype='uint8')
        for reg in numpy.unique(labels[sa_m]):
            alluv[labels == reg] = 1

        alluv_out = self.empty().astype('uint8')
        alluv_out.nodataValues = [0]
        alluv_out[:] = alluv
        return watershed(alluv_out, tempdir=self.tempdir)


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
                    self.hru_attributes[id][name] = 'No Data'
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
        raise Exception('Sinuosity sampling distance must be greater than 0')

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


class Groundwater2D(object):
    '''
    Groundwater flow calculation domain.
    k is a hydraulic conductivity surface, b is a thickness surface
    '''
    def __init__(self, k, b, head_fill=0, length_units='m', time_units='s'):
        # Create a head surface
        self.nodata = -99999
        if k.shape != b.shape:
            raise ValueError('k and b have different shapes.')
        self.shape = b.shape
        self.head = numpy.full(self.shape, head_fill, dtype='float32')

        print "Computing transmissivity"
        knodata = k.nodata
        bnodata = b.nodata
        self.template = raster(k)
        k = numpy.pad(k.array, 1, 'constant', constant_values=0)
        b = numpy.pad(b.array, 1, 'constant', constant_values=0)
        self.domain = (k != knodata) & (b != bnodata) & (k != 0) & (b != 0)
        # Ensure no-flow boundary created where K is zero or nodata
        self.head[~self.domain[1:-1, 1:-1]] = self.nodata
        # Compute transmissivity in all directions
        k_0, b_0 = k[1:-1, 1:-1], b[1:-1, 1:-1]
        # +y
        slicemask = self.domain[1:-1, 1:-1] & self.domain[:-2, 1:-1]
        k_1, b_1 = k[:-2, 1:-1], b[:-2, 1:-1]
        self.tup = ne.evaluate(
            'where(slicemask,(2/((1/b_1)+(1/b_0)))*(2/((1/k_1)+(1/k_0))),0)'
        )
        # -y
        slicemask = self.domain[1:-1, 1:-1] & self.domain[2:, 1:-1]
        k_1, b_1 = k[2:, 1:-1], b[2:, 1:-1]
        self.tdown = ne.evaluate(
            'where(slicemask,(2/((1/b_1)+(1/b_0)))*(2/((1/k_1)+(1/k_0))),0)'
        )
        # -x
        slicemask = self.domain[1:-1, 1:-1] & self.domain[1:-1, :-2]
        k_1, b_1 = k[1:-1, :-2], b[1:-1, :-2]
        self.tleft = ne.evaluate(
            'where(slicemask,(2/((1/b_1)+(1/b_0)))*(2/((1/k_1)+(1/k_0))),0)'
        )
        # +x
        slicemask = self.domain[1:-1, 1:-1] & self.domain[1:-1, 2:]
        k_1, b_1 = k[1:-1, 2:], b[1:-1, 2:]
        self.tright = ne.evaluate(
            'where(slicemask,(2/((1/b_1)+(1/b_0)))*(2/((1/k_1)+(1/k_0))),0)'
        )

    def addBoundary(self, indices, type='no-flow', values=None):
        '''
        Add a boundary condition to a set of cell indices.
        Available types are "no-flow", "constant", "head-dependent", "flux".
        If any of "constant, "head-dependent", or "flux", the corresponding
        "values" must be specified (and match the indices size).
        '''
        if type == 'no-flow':
            self.head[indices] = self.nodata # Need to fix this
        elif type == 'constant':
            intcon = numpy.full(self.shape, self.nodata, dtype='float32')
            if hasattr(self, 'constantmask'):
                intcon[self.constantmask] = self.constantvalues
            intcon[indices] = values
            intcon[~self.domain[1:-1, 1:-1]] = self.nodata
            self.constantmask = intcon != self.nodata
            self.constantvalues = intcon[self.constantmask]
        elif type == 'flux':
            if not hasattr(self, 'fluxmask'):
                self.fluxmask = numpy.zeros(shape=self.shape, dtype='bool')
                self.fluxmask[indices] = 1
                self.fluxvalues = values
            else:
                vals = numpy.zeros(shape=self.shape, dtype='float32')
                vals[self.fluxmask] = self.fluxvalues
                vals[indices] = values
                self.fluxmask = vals != 0
                self.fluxvalues = vals[self.fluxmask]
        elif type == 'head-dependent':
            pass

    def solve(self, max_iter=100, tolerance=1E-6, show_convergence=False):
        '''Calculate a head surface using current parameter set'''
        # Apply constant head
        if hasattr(self, 'constantmask'):
            self.head[self.constantmask] = self.constantvalues

        # Create a copy of head for calculation
        head = numpy.pad(self.head, 1, 'constant', constant_values=self.nodata)
        # Create iterator for neighbourhood calculations
        nbrs = [('up', slice(0, -2), slice(1, -1)),
                ('down', slice(2, None), slice(1, -1)),
                ('left', slice(1, -1), slice(0, -2)),
                ('right', slice(1, -1), slice(2, None))]
        m = head != self.nodata
        # Iterate and implicitly compute head using derivation:
        # http://inside.mines.edu/~epoeter/583/06/discussion/fdspreadsheet.htm
        resid = tolerance + 0.1
        iters = 0
        convergence = []
        # Create calculation set
        flux = numpy.zeros(shape=self.shape, dtype='float32')
        hdd = numpy.zeros(shape=self.shape, dtype='float32')
        if hasattr(self, 'fluxmask'):
            flux[self.fluxmask] = self.fluxvalues
        if hasattr(self, 'hddpmask'):
            hdd[self.hddpmask] = self.hddpvalues
        calc_set = {'f': flux, 'tup': self.tup, 'tdown': self.tdown,
                    'tleft': self.tleft, 'tright': self.tright,
                    'hdd': hdd}
        den = ne.evaluate('tup+tdown+tleft+tright', local_dict=calc_set)
        calc_set.update({'den': den})
        while resid > tolerance:
            iters += 1
            # Copy head from previous iteration
            h = numpy.copy(head)

            # Add head views to calculation set
            for nbr in nbrs:
                name, i_slice, j_slice = nbr
                calc_set['h%s' % name] = head[i_slice, j_slice]

            # Calculate head at output
            h[1:-1, 1:-1] = ne.evaluate(
                'where(den>0,((hup*tup)+(hdown*tdown)+(hleft*tleft)+'
                '(hright*tright)+f+hdd)/den,0)',
                local_dict=calc_set
            )

            # Replace constant head values
            if hasattr(self, 'constantmask'):
                h[1:-1, 1:-1][self.constantmask] = self.constantvalues

            # Compute residual
            resid = numpy.max(numpy.abs(head - h))
            convergence.append(resid)
            if iters == max_iter:
                print "No convergence, with a residual of %s" % (resid)
                break

            # Update heads
            head = h
        self.head = head[1:-1, 1:-1]

        print ("%i iterations completed with a residual of %s" %
               (iters, resid))

        if show_convergence:
            plt.plot(convergence)
            plt.ylabel('Residual')
            plt.xlabel('Iteration')
            plt.show()

    def calculate_Q(self):
        '''Use the head surface to calculate the steady-state flux'''
        nbrs = [(self.t_y,
                 (slice(0, -1), slice(None, None)),
                 (slice(1, None), slice(None, None))),
                (self.t_y,
                 (slice(1, None), slice(None, None)),
                 (slice(0, -1), slice(None, None))),
                (self.t_x,
                 (slice(None, None), slice(0, -1)),
                 (slice(None, None), slice(1, None))),
                (self.t_x,
                 (slice(None, None), slice(1, None)),
                 (slice(None, None), slice(0, -1)))]
        m = self.head != self.nodata
        for nbr in nbrs:
            t, take, place = nbr
            mask = m[take[0], take[1]] & m[place[0], place[1]]
            t = t[mask]
            calcset = den[place[0], place[1]][mask]
            den[place[0], place[1]][mask] = ne.evaluate('calcset+t')
            headset = self.head[take[0], take[1]][mask]
            calcset = num[place[0], place[1]][mask]
            num[place[0],
                place[1]][mask] = ne.evaluate('calcset+(t*headset)')
        self.q = q

    def view(self, attr='head'):
        '''View the current head surface'''
        fig, ax = plt.subplots()
        a = numpy.copy(self.__dict__[attr])
        a[a == self.nodata] = numpy.nan
        im = ax.imshow(a, cmap='terrain')
        fig.colorbar(im)
        plt.show()

    def saveAsRaster(self, outpath, attr='head', nodata=-99999):
        '''Save to a raster file'''
        self.template.nodataValues = [nodata]
        self.template.save_gdal_raster(outpath)
        temp = raster(outpath, mode='r+')
        if attr == 'Q':
            # Calculate Q
            self.calculate_Q()
            temp[:] = self.q
        else:
            a = self.__dict__[attr]
            a[~self.domain[1:-1, 1:-1]] = nodata
            temp[:] = a


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


def riparian_delineation(dem, stream_order, flow_accumulation):
    """
    Define zones of riparian connectivity.  Assumes all arrays match
    """
    # Calculate network of costs and normalize result
    print "Creating cost surface"
    cost = inverse(normalize(cost_surface(stream_order, topo(dem).slope())))

    # Calculate indexed sinuosity/stream slope and extrapolate outwards
    print "Calculating stream slope"
    stream_slope = gaussian(extrapolate_buffer(inverse(normalize(watershed(dem).stream_slope(stream_order))), 150), 15)

    print "Calculating sinuosity"
    sinu = gaussian(extrapolate_buffer(normalize(channel_density(stream_order)), 150), 15)

    # Reclassify flow accumulation and extrapolate outwards
    print "Reclassifying flow accumulation"
    flow_accumulation = raster(flow_accumulation)
    stream_order = raster(stream_order)

    output = raster(flow_accumulation).astype('float32')
    fa = flow_accumulation.array.astype('float32')
    fa[stream_order.array == stream_order.nodata] = output.nodata
    output[:] = fa
    flow_accumulation = gaussian(extrapolate_buffer(normalize(output), 150), 15)

    # Reclassify and aggregate data
    print "Reclassifying cost"
    percentile = 92
    a = cost.array
    m = a != cost.nodata
    thresh = numpy.percentile(a[m], percentile)
    thresh_m = a < thresh
    a[thresh_m] = cost.nodata
    _min = a[thresh_m].min()
    a[thresh_m] = (a[thresh_m] - _min) / (a[thresh_m].max() - _min)

    print "Aggregating output"
    strslo = stream_slope.array
    sinua = sinu.array
    faa = flow_accumulation.array
    m = (a != cost.nodata) & (strslo != stream_slope.nodata) & (sinua != sinu.nodata) & (faa != flow_accumulation.nodata)
    a[~m] = cost.nodata
    a[m] *= 10
    a[m] += strslo[m] + sinua[m] + faa[m]
    a[m] /= 13

    cost[:] = a
    return cost


def bank_slope(dem, slope_threshold=15, streams=None, slope=None, min_contrib_area=None):
    """
    Delineate steep regions contributing to a stream
    :param dem:
    :param streams:
    :return:
    """
    if streams is None:
        if min_contrib_area is None:
            raise RasterError('No minimum contributing area specified to delineate streams')
        print "Delineating streams"
        with watershed(dem).stream_order(min_contrib_area) as streamOrder:
            streams = streamOrder.array != streamOrder.nodata
        demRast = topo(dem)
    else:
        streamData = raster(streams)
        # Match DEM to streams
        demRast = topo(topo(dem).match_raster(streamData))
        streams = streamData.array != streamData.nodata

    # Calculate slope
    if slope is None:
        with demRast.slope() as slope:
            sla = slope.array
            slnd = slope.nodata
    else:
        with topo(slope).match_raster(demRast) as slope:
            sla = slope.array
            slnd = slope.nodata

    # Label regions above the slope threshold
    highSlope = (sla != slnd) & (sla > slope_threshold)
    labels, num = ndimage.label(highSlope, numpy.ones(shape=(3, 3)))
    labels = ndimage.binary_dilation(highSlope, numpy.ones(shape=(3, 3)))



def cumulative_effectiveness(stream_raster, tree_height=50):
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


def wetness(dem, minimum_area):
    """
    Calculate a wetness index using streams of a minimum contributing area
    :param dem: dem (raster)
    :param minimum_area: area in units^2
    :return: raster instance
    """
    return normalize(inverse(cost_surface(bluegrass.stream_order(dem, minimum_area), topo(dem).slope())))


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


def non_contributing_regions(dem, streams, distance_weight=0.6):
    # Make sure DEM aligns
    dem = raster(dem)
    dem.interpolationMethod = 'bilinear'
    dem = dem.match_raster(streams)

    # Delineate the riparian (to exclude)
    slope = topo(dem).slope()
    cost = cost_surface(streams, slope)
    a = cost.array
    m = a != cost.nodata
    thresh = numpy.percentile(a[m], 5)
    riparian = m & (a < thresh)
    del a, m

    # Calculate slope
    print "Calculating height above streams"
    # Generate mask from streams
    streams = raster(streams)
    stream_nodata = streams.array == streams.nodata

    # Generate stream elevation raster
    print "Adding elevation to streams"
    streams = dem.copy()
    streamA = streams.array
    streamA[stream_nodata] = streams.nodata
    streams[:] = streamA
    del streamA

    #  Interpolate no data values around streams
    print "Interpolating regions around streams"
    stream_surface = interpolate_nodata(streams)

    # Subtract interpolated elevations from dem
    print "Subtracting stream elevations from DEM"
    demA = dem.array
    m = demA != dem.nodata
    demA[m] -= stream_surface.array[m]

    # Create index where closest value to 0 is highest
    print "Creating index from elevation difference"
    over = m & (demA > 0) & ~riparian
    over_set = demA[over]
    over_set = (over_set - over_set.min()) / (over_set.max() - over_set.min())
    demA[over] = 1. - over_set
    del over_set
    under = m & (demA < 0) & ~riparian
    under_set = demA[under]
    under_set = (under_set - under_set.min()) / (under_set.max() - under_set.min())
    demA[under] = under_set
    del under_set
    demA[demA == 0] = 1.

    # Calculate distance array to streams and normalize
    print "Creating final connectivity index"
    d = distance(streams).array
    m = m & ~riparian
    data = d[m]
    dataMin = data.min()
    demA[m] += (data - dataMin) / (data.max() - dataMin) * distance_weight
    demA[m] /= 1 + distance_weight
    demA[riparian] = 0

    out = dem.empty()
    out[:] = demA
    return out
