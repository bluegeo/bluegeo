'''
Hydrologic analysis library

Blue Geosimulation, 2017
'''

from raster import *
import util
import math
from scipy import ndimage, interpolate
from numba.decorators import jit
from terrain import *
from rastfilter import *
from skimage.measure import label
try:
    from bluegrass import GrassSession
except ImportError:
    print "Warning: Grass functions not available"


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

    def stream_slope(self, iterations=1, streams=None, min_contrib_area=None,
                     fa=None, units='degrees'):
        '''
        Compute the slope from cell to cell in streams with a minimum
        contributing area.  If streams are specified, they will not be
        computed.
        '''
        if streams is not None:
            with self.match_raster(streams) as dem:
                elev = dem.array
            strms = raster(streams)
            m = strms.array != strms.nodata
        else:
            if min_contrib_area is None:
                raise WatershedError('min_contrib_area must be specified if no'
                                     ' stream layer is provided')
            if fa is not None:
                with self.stream_reclass(fa, min_contrib_area) as ws:
                    m = ws.array != ws.nodata
            else:
                with self.stream_order(min_contrib_area) as ws:
                    m = ws.array != ws.nodata
            elev = self.array
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
            rise = numpy.abs(base - elev[s][m[s]])
            run_ = run[m[s]]
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
        for _iter in range(iterations):
            slopefill = [compute(inds[0][i], inds[1][i])
                         for i in range(inds[0].shape[0])]
            a[m] = slopefill
        output[:] = a
        return output

    def alluvium(self, slope_thresh=6, stream_slope_thresh=5, **kwargs):
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
        streams = kwargs.get('streams', None)
        min_contrib_area = kwargs.get('min_contrib_area', 1E04)
        slope = kwargs.get('slope', None)
        fa = kwargs.get('fa', None)
        if streams is None:
            strslo = self.stream_slope(2, min_contrib_area=min_contrib_area,
                                       fa=fa)
        else:
            strslo = self.stream_slope(2, streams)
        seeds = set(zip(*numpy.where(strslo.array != strslo.nodata)))
        if slope is None:
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


class hru(raster):
    """
    Create a model domain instance that is a child of the raster class
    """
    def __init__(self, dem, output_srid=4269):
        # Open and change to float if not already
        # TODO: Add outlet argument to turn the input into a watershed
        if isinstance(dem, raster):
            self.__dict__.update(dem.__dict__)
        else:
            super(hru, self).__init__(dem)
        if 'float' not in self.dtype:
            selfcopy = self.astype('float32')
            self.__dict__.update(selfcopy.__dict__)
            selfcopy.garbage = []
            del selfcopy
        self.srid = 4269
        # Change interpolation method unless otherwise specified
        self.interpolationMethod = 'bilinear'
        self.wkdir = os.path.dirname(self.path)
        self.spatialData = {}
        self.zonalData = {}
        # TODO: Use watershed to create mask
        self.mask = self.array != self.nodata

    def add_spatial_data(self, dataset, name, summary_method='mean', interval=0, number=0, bins=[]):
        """
        Split spatial HRU's using a dataset and zones
        If the bins argument is used, it will override the other interval argumants.
        Similarly, if the number argument is not 0 it will override the interval argument.
        :param dataset: Instance of the raster class
        :param name: Name to be used for output HRU's
        :param summary_method: Method used to summarize original data within bins
        :param interval: float: Interval in units to divide into HRU's
        :param number: Number of regions to split the dataset into
        :param bins: Manual bin edges used to split the dataset into regions
        :return: None
        """
        # Check arguments
        if all([interval <= 0, number <= 0, len(bins) == 0]):
            raise hruError('One of interval, number, or bins must be specified to create spatial regions')
        if isinstance(dataset, raster):
            raise hruError('The input dataset must be a raster class instance'
                           '(so the correct interpolation method is inherent)')

        # HRU's must be reset
        del self.hrus

        # Align with hru domain and mask over watershed
        ds = dataset.match_raster(self.path)
        a = ds.array
        a[self.mask] = ds.nodata

        # Create output and gather data
        bands = numpy.zeros(shape=a.shape, dtype='uint64')
        m = a != ds.nodata
        a = a[m]

        # Digitize
        if len(bins) != 0:
            pass
        elif number > 0:
            bins = numpy.linspace(a.min(), a.max(), number + 1)
        else:
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
        bands[m] = numpy.digitize(a, bins) + 1

        # Update spatial HRU datasets with labeled data and original data
        out = ds.astype('uint64')
        out.nodataValues = [0]
        out[:] = self.relabel(bands)

        # Add to spatial datasets and add original to zonal datasets
        if name in self.spatialData.keys():
            print "Warning: Existing spatial dataset {} will be overwritten".format(name)
        self.spatialData[name] = out
        if name in self.zonalData.keys():
            print "Warning: Existing zonal dataset {} will be overwritten".format(name)
        self.zonalData[name] = (ds, summary_method)

    def add_zonal_data(self, dataset, name, summary_method='mean'):
        """
        Prepare a dataset for zonal statistics while creating HRUs
        :param dataset: Instance of the raster class
        :param name: Name of the dataset to be used in the HRU set
        :param summary_method: Statistical method to be applied
        :return: None
        """
        if isinstance(dataset, raster):
            raise hruError('The input dataset must be a raster class instance'
                           '(so the correct interpolation method is inherent)')

        if name in ['Area', 'centroid']:
            raise hruError("Name cannot be 'Area' or 'centroid'")

        # Align with domain
        ds = dataset.match_raster(self.path)

        # Add to spatial datasets
        if name in self.zonalData.keys():
            print "Warning: Existing zonal dataset {} will be overwritten".format(name)
        self.zonalData[name] = (ds,  summary_method)

    def create(self):
        """
        Create HRU set using spatial data
        :return: None
        """
        if len(self.spatialData) == 0:
            raise hruError('No spatial datasets available to create HRUs')

        # Create output raster
        self.hrus = self.astype('uint64')
        self.hrus.nodataValues = [0]

        # Iterate spatial datasets and create HRUs
        names = self.spatialData.keys()
        hrua = numpy.zeros(shape=self.hrus.shape, dtype='uint64')
        for name in names[:-1]:
            ds = self.spatialData[name]
            print "Adding {}".format(name)
            hrua = self.relabel(hrua + ds.array + hrua.max())

        # Add last dataset separately in order to create map
        name = names[-1]
        ds = self.spatialData[name]
        print "Adding {}".format(name)
        self.hrus[:], self.hruMap = self.relabel(hrua + ds.array + hrua.max(), return_map=True)

    def compute_zonal_stats(self):
        """
        Use domain.zonalData to produce zonal summary statistics for output.
        Centroids and areas are also added implicitly
        :return: dict of hru id's and the value of each column
        """
        if not hasattr(self, 'hrus'):
            self.create()

        methods = {'mean': numpy.mean,
                   'mode': util.mode,
                   'min': numpy.min,
                   'max': numpy.max,
                   'std': numpy.std}  # Add more as needed...

        hrus = {id: {'centroid': self.compute_centroid(id), 'Area': self.compute_area(id)}
                for id in self.hruMap.keys()}
        for name, zoneData in self.zonalData.iteritems():
            rast, method = zoneData
            a = rast.array
            nd = rast.nodata
            method = methods[method]
            for id in self.hruMap.keys():
                data = a[self.hruMap[id]]
                data = data[data != nd]
                if data.size == 0:
                    hrus[id][name] = 'No Data'
                hrus[id][name] = method(data)
        return hrus

    def write_raven_rvh(self, template_file):
        """
        Write an .rvh file to be used in the Raven Hydrological Modal
        :param template_file: Path to a file to use as a template to write an output .rvh file
        :return: None
        """
        # Create HRUs
        hrus = self.compute_zonal_stats()

        # Read template
        with open(template_file, 'r') as f:
            lines = f.readlines()
        self.computeCentroids()
        self.computeArea()
        out = open(output_name, 'w')
        w = False
        for line in lines:
            if ':HRUs' in line:
                out.write(line)
                w = True
                continue
            if w:
                keys = self.hrus[self.hrus.keys()[0]].keys()
                write = ['  :Attributes', 'AREA', 'ELEVATION', 'LATITUDE',
                         'LONGITUDE']
                for key in keys:
                    if key not in write:
                        write.append('%s' % key)
                out.write(','.join(write) + '\n')
                out.write('  :Units,km2,m,deg,deg, <----- Check these units,'
                          ' and manually enter remainder of units\n')
                spatial = ['AREA', 'ELEVATION', 'LATITUDE', 'LONGITUDE']
                for i in range(1, len(self.hrus) + 1):
                    hru, namedict = i, self.hrus[i]
                    write = ['%i' % (hru)]
                    for s in spatial:
                        write.append('%s' % (namedict[s]))
                    for name in keys:
                        if name in spatial:
                            continue
                        write.append('%s' % (namedict[name]))
                    out.write(','.join(write) + '\n')
                out.write(':EndHRUs')
                break
            else:
                out.write(line)
        print "Complete."

    def elevation_band(self, interval=100, number=0, bins=[]):
        """
        Add elevation bands to the spatial HRU set
        :param interval: see add_spatial_data
        :param number: see add_spatial_data
        :param bins: see add_spatial_data
        :return: None
        """
        # Create elevation bands
        self.add_spatial_data(raster(self.path), 'Elevation', reclass_method='mean',
                              interval=interval, number=number, bins=bins)

    def add_aspect(self, interval=0, number=4, bins=[]):
        """
        Compute aspect and add to spatial HRU set
        :param interval: see add_spatial_data
        :param number: see add_spatial_data
        :param bins: see add_spatial_data
        :return: None
        """
        # Compute aspect and add to HRU set
        with topo(self.data).aspect() as aspect:
            self.add_spatial_data(raster(aspect.path), 'Aspect', reclass_method='mode',
                                  interval=interval, number=number, bins=bins)

    def add_slope(self, interval=0, number=4, bins=[]):
        """
        Compute slope and add to spatial HRU set
        :param interval: see add_spatial_data
        :param number: see add_spatial_data
        :param bins: see add_spatial_data
        :return: None
        """
        # Compute aspect and add to HRU set
        with topo(self.data).slope() as slope:
            self.add_spatial_data(raster(slope.path), 'Slope', reclass_method='mean',
                                  interval=interval, number=number, bins=bins)

    def simplify(self, iterations):
        """
        Remove small segments of HRU's.  Applies an iterative mode filter.
        :param iterations: Number of iterations to smooth dataset
        :return: None
        """
        if not hasattr(self, 'hrus'):
            self.create()

        for i in range(iterations):
            print "Performing filter {} of {}".format(i + 1, iterations)
            self.hrus = raster(rastfilter(self.hrus.path).most_common().path)

        self.hrus[:], self.hruMap = self.relabel(self.hrus.array, return_map=True)

    def compute_centroid(self, id):
        """
        Compute the centre of mass centroid of a specific HRU
        :param id: hru id
        :return: centroid as (x, y)
        """
        try:
            inds = self.hruMap[id]
        except KeyError:
            raise hruError('Could not compute the cenroid for HRU {}, as it does not exist'.format(id))

        # Centre of mass in spatial reference of dem
        y = self.top - ((numpy.mean(inds[0]) + 0.5) * self.csy)
        x = self.left + ((numpy.mean(inds[1]) + 0.5) * self.csx)

        # Change to output srid
        insr = osr.SpatialReference()
        insr.ImportFromWkt(self.projection)
        outsr = osr.SpatialReference()
        outsr.ImportFromEPSG(self.srid)
        coordTransform = osr.CoordinateTransformation(insr, outsr)
        x, y, _ = coordTransform.TransformPoint(x, y)
        return x, y

    def compute_area(self, id):
        """
        Compute area in the units of the dem spatial reference
        :param id: HRU index
        :return: float of area
        """
        try:
            inds = self.hruMap[id]
        except KeyError:
            raise hruError('Could not compute the cenroid for HRU {}, as it does not exist'.format(id))

        return inds[0].size * self.csx * self.csy

    def save_hru_raster(self, output_name):
        """
        Save the current HRU set as a raster
        :param output_name: name of the output raster
        :return: None
        """
        if not hasattr(self, 'hrus'):
            self.create()

        if output_name.split('.')[-1].lower() != 'tif':
            output_name += '.tif'
        self.hrus.save_gdal_raster(output_name)

    def get_specs(self):
        """
        Return a dictionary of the specifications of the current HRU set
        :return: dict
        """
        if not hasattr(self, 'hrus'):
            self.create()

        return {'Number': numpy.max(self.hruMap.keys()),
                'Spatial Datasets': self.spatialData.keys(),
                'Zonal Datasets': self.zonalData.keys()}


def sinuosity(**kwargs):
    """
    Calculate sinuosity from a dem or streams
    :param kwargs: dem=path to dem _or_ stream_order=path to strahler stream order raster
        distance=search distance to calculate sinuosity ratio
    :return: sinuosity as a ratio
    """
    # Collect as raster of streams
    dem = kwargs.get('dem', None)
    stream_order = kwargs.get('stream_order')
    distance = kwargs.get('sample_distance', 100)  # Default measurement distance is 100m
    radius = distance / 2.
    if distance <= 0:
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
        raise Exception('Sinuosity needs requires either a dem or streams dataset')

    # Remove connecting regions to avoid over-counting
    m = rastfilter(stream_order).min_filter().array != rastfilter(stream_order).max_filter().array
    streamCopy = stream_order.copy()
    a = streamCopy.array
    a[m] = stream_order.nodata
    streamCopy[:] = a
    del a

    # Label and map stream order
    stream_order, stream_map = region_label(streamCopy, True)
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

    sinuosity_raster = stream_order.copy().astype('float32')
    outa = sinuosity_raster.array
    sinuNodata = outa == sinuosity_raster.nodata
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
    outa[sinuNodata] = sinuosity_raster.nodata
    outaVals = outa[~sinuNodata]
    outaMin = outaVals.min()
    outa[~sinuNodata] = (outaVals - outaMin) / (outaVals.max() - outaMin)
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


def riparianDelineation(dem, streams=None, slope=None, cost_rasters=[], min_contrib_area=None, reclass_percentile=10):
    """
    Define zones of riparian connectivity.

    min_cut_slope is the lowest slope of cut point slopes.
    max_bench_slope is the maximum slope a bench can be.
    number_benches is the number of sequential benches to delineate as part of
    the riparian zone.
    """
    # Delineate streams
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

    print "Creating cost surface"
    # Slope and distance to streams are always used to create a cost surface
    if slope is None:
        print "Calculating Slope"
        with demRast.slope() as slope:
            sla = slope.array
            slnd = slope.nodata
    else:
        with topo(slope).match_raster(demRast) as slope:
            sla = slope.array
            slnd = slope.nodata

    costMask = sla != slnd
    cost = numpy.zeros(shape=sla.shape, dtype='float32')
    if len(cost_rasters) > 0:
        slopeFactor = len(cost_rasters)
    else:
        slopeFactor = 1
    cost[costMask] = (sla[costMask] / sla[costMask].max()) * slopeFactor
    del sla

    # Add distance to streams to cost
    # streamDist = ndimage.distance_transform_edt(~streams, (demRast.csx, demRast.csy))
    # streamDist /= streamDist.max()
    # cost[costMask] += streamDist[costMask]

    # Apply additional cost surfaces
    for surface in cost_rasters:
        with topo(surface).match_raster(demRast) as cr:
            a = cr.array
            costMask = costMask & (a != cr.nodata)
            cost[costMask] += a[costMask]
            del a

    cost[~costMask] = -1
    cost[streams] = 0
    cost[cost != -1] /= cost.max()

    # Calculate least cost path surface from outlet
    print "Calculating network"
    streamElev = demRast.array
    streamElev[~streams] = demRast.nodata
    i, j = numpy.where(streamElev == numpy.min(streamElev[streamElev != demRast.nodata]))
    i, j = i[0], j[0]
    mcp = MCP_Geometric(cost, sampling=(demRast.csy, demRast.csx))
    cost, traceback = mcp.find_costs([(i, j)])

    m = numpy.isnan(cost) | numpy.isinf(cost)

    cost[m] = demRast.nodata
    # cost[cost > numpy.percentile(cost[~m], reclass_percentile)] = demRast.nodata
    costData = cost != demRast.nodata
    aVals = cost[costData]
    aMin = aVals.min()
    cost[costData] = 1 - ((aVals - aMin) / (aVals.max() - aMin))
    # Prepare and send output
    outrast = demRast.copy()
    outrast[:] = cost
    return outrast


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



def cumulativeEffectiveness(stream_raster, fa_dilation,
                            slope_dilation, tree_height=50):
    """
    Define zones of riparian cumulative effectiveness.

    """
    # Read input
    util.parseInput(stream_raster)
    util.parseInput(fa_dilation)
    util.parseInput(slope_dilation)
    # Compute distance
    dist = streamDistance(stream_raster)
    dist.changeDataType('float32')
    # Buffer width is associated with contributing area
    conta = fa_dilation.load('r+')
    dista = dist.load('r+')
    # Scale contributing area
    m = conta != fa_dilation.nodata[0]
    conta[m] = numpy.sqrt(conta[m]) / numpy.sqrt(numpy.max(conta[m][dista[m] <= tree_height]))
    # Scale Distance
    m = dista != dist.nodata[0]
    dista[m] /= tree_height
    # Create similarity index
    m = m & (conta != 0) & (conta != fa_dilation.nodata[0])
    dista[m] = dista[m] / conta[m]
    dista[dista > 1] = dist.nodata[0]
    dista[~m] = dist.nodata[0]
    del conta
    del dista
    # Index slope dilation
    sl = slope_dilation.load('r+')
    m = sl != slope_dilation.nodata[0]
    sl[m] /= numpy.percentile(sl[m], 85)
    sl[(sl > 1) & m] = 1
    sl[m] = 1 - sl[m]
    # Shade is equal to dist
    shade = raster(dist)
    a = shade.load('r+')
    m = a != shade.nodata[0]
    a[m] = 1 - a[m]
    del a
    # Litter
    litter = raster(dist)
    la = litter.load('r+')
    la[m][la[m] > 0.6] = litter.nodata[0]
    m = la != litter.nodata[0]
    la[m] /= 0.6
    la[m] = (1 - la[m]) * sl[m]
    del la
    # Coarse
    coarse = raster(dist)
    co = coarse.load('r+')
    m = co != coarse.nodata[0]
    co[m] = (1 - co[m]) * sl[m]
    del co
    # Root
    root = raster(dist)
    a = root.load('r+')
    m = a != root.nodata[0]
    a[a < 0.25] = root.nodata[0]
    a[a >= 0.25] = 1
    m = a != root.nodata[0]
    a[m] *= sl[m]
    del a
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
