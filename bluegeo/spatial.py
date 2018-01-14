"""
Custom vector analysis library
"""
import os
import shutil
import numbers
from contextlib import contextmanager
import numpy
from osgeo import gdal, ogr, osr, gdalconst
import h5py
from skimage.measure import label as sklabel
from skimage.measure import regionprops
try:
    import Image
    import ImageDraw
except ImportError:
    from PIL import Image, ImageDraw
try:
    # Numexpr is used as the default solver
    import numexpr as ne
    NUMEXPR = True
except ImportError:
    NUMEXPR = False
try:
    from shapely import wkb as shpwkb
    from shapely import geometry, ops
except ImportError:
    print "Warning: Shapely is not installed and some operations will not be possible."

from util import parse_projection, generate_name, coords_to_indices, indices_to_coords, transform_points

gdal.PushErrorHandler('CPLQuietErrorHandler')

"""Custom Exceptions"""
class RasterError(Exception):
    pass


class VectorError(Exception):
    pass


class ExtentError(Exception):
    pass


class extent(object):

    def __init__(self, data):
        """
        Extent to be used to control geometries
        :param data: iterable of (top, bottom, left, right) or vector class instance, or instance of raster class
        """
        if any(isinstance(data, o) for o in [tuple, list, numpy.ndarray]):
            self.bounds = data
            self.geo = None
        elif isinstance(data, vector):
            self.bounds = (data.top, data.bottom, data.left, data.right)
            self.geo = data
        elif isinstance(data, raster):
            self.bounds = (data.top, data.bottom, data.left, data.right)
            self.geo = data
        elif isinstance(data, extent):
            self.__dict__.update(data.__dict__)
        else:
            raise ExtentError('Unsupported extent argument of type {}'.format(type(data).__name__))

        try:
            if self.bounds[0] <= self.bounds[1] or self.bounds[2] >= self.bounds[3]:
                assert False
        except:
            raise ExtentError("Invalid or null extent")

    def within(self, other):
        """
        Check if this extent is within another
        :param other: other data that can be instantiated using the extent class
        :return: boolean
        """
        try:
            other = extent(other).transform(self.geo.projection)
        except AttributeError:
            other = extent(other)
        top, bottom, left, right = self.bounds
        _top, _bottom, _left, _right = extent(other).bounds
        if all([top <= _top, bottom >= _bottom, left >= _left, right <= _right]):
            return True
        else:
            return False

    def intersects(self, other):
        """
        Check if this extent intersects another
        :param other: other data that can be instantiated using the extent class
        :return: boolean
        """
        try:
            other = extent(other).transform(self.geo.projection)
        except AttributeError:
            other = extent(other)
        top, bottom, left, right = self.bounds
        _top, _bottom, _left, _right = extent(other).bounds
        if any([top <= _bottom, bottom >= _top, left >= _right, right <= _left]):
            return False
        else:
            return True

    def contains(self, other):
        return extent(other).within(self)

    def transform(self, projection, precision=1E-09):
        """
        Transform the current extent to the defined projection.
        Note, the geo attribute will be disconnected for safety.
        :param projection: input projection argument
        :return: new extent instance
        """
        if self.geo is None:
            # Cannot transform, as the coordinate system is unknown
            return extent(self.bounds)

        # Gather the spatial references
        wkt = parse_projection(projection)
        insr = osr.SpatialReference()
        insr.ImportFromWkt(self.geo.projection)
        outsr = osr.SpatialReference()
        outsr.ImportFromWkt(wkt)
        if insr.IsSame(outsr):
            # Nothing needs to be done
            return extent(self.bounds)

        def optimize_extent(ct, constant_bound, start, stop, accum=True, direction='y'):
            space = numpy.linspace(start, stop, 4)
            residual = precision + 1.
            while residual > precision:
                if direction == 'y':
                    coords = [ct.TransformPoint(x, constant_bound)[1] for x in space]
                else:
                    coords = [ct.TransformPoint(constant_bound, y)[0] for y in space]
                if accum:
                    next_index = numpy.argmax(coords)
                else:
                    next_index = numpy.argmin(coords)
                track = coords[next_index]
                if next_index == 0 or next_index == 3:
                    break
                delta = space[1] - space[0]
                space = numpy.linspace(space[next_index] - delta, space[next_index] + delta, 4)
                try:
                    residual = abs(track - _track)
                except:
                    pass
                _track = track
            return track

        # Generate coordinate transform instance
        coordTransform = osr.CoordinateTransformation(insr, outsr)

        top = optimize_extent(coordTransform, self.bounds[0], self.bounds[2], self.bounds[3])

        bottom = optimize_extent(coordTransform, self.bounds[1], self.bounds[2], self.bounds[3], False)

        left = optimize_extent(coordTransform, self.bounds[2], self.bounds[1], self.bounds[0], False, 'x')

        right = optimize_extent(coordTransform, self.bounds[3], self.bounds[1], self.bounds[0], True, 'x')

        return extent((top, bottom, left, right))

    @property
    def corners(self):
        """Get corners of extent as coordinates [(x1, y1), ...(xn, yn)]"""
        top, bottom, left, right = self.bounds
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    def __eq__(self, other):
        check = extent(other)
        if not self.geo is None:
            check = check.transform(self.geo.projection)
        return numpy.all(numpy.isclose(self.bounds, check.bounds))


class raster(object):
    """
    Main raster data interfacing class

    Data can be:
        1.  Path to a GDAL supported raster
        2.  Similarly, a gdal raster instance
        3.  Path to an HDF5 dataset
        4.  An h5py dataset instance
        5.  Another instance of this raster class,
            which creates a copy of the input.

    Data are first read virtually, whereby all
    attributes may be accessed, but no underlying
    grid data are loaded into memory.

    Data from the raster may be loaded into memory
    using standard __getitem__ syntax, by accessing
    the .array attribute (property), or by iterating
    all chunks (i.e. blocks, or tiles).

    Modes for modifying raster datasets are 'r',
    'r+', and 'w'.  Using 'w' will create an empty
    raster, 'r+' will allow direct modification of the
    input dataset, and 'r' will automatically
    create a copy of the raster for writing if
    a function that requires modification is called.
    """
    def __init__(self, input_data, mode='r', **kwargs):
        # Record mode
        if mode in ['r', 'r+', 'w']:
            self.mode = mode
        else:
            raise RasterError('Unrecognized file mode "%s"' % mode)

        # Check if input_data is a string
        if isinstance(input_data, basestring):
            # If in 'w' mode, write a new file
            if self.mode == 'w':
                self.build_new_raster(input_data, **kwargs)
            # Check if input_data is a valid file
            elif not os.path.isfile(input_data) and not os.path.isdir(input_data):
                raise RasterError('%s is not a raster file' % input_data)
            else:
                # Try an HDF5 input_data source
                try:
                    with h5py.File(input_data, libver='latest') as ds:
                        self.load_from_hdf5(ds)
                        self.format = 'HDF5'
                except Exception as e:
                    # Try for a gdal dataset
                    ds = gdal.Open(input_data)
                    try:
                        gt = ds.GetGeoTransform()
                        # assert gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                        self.load_from_gdal(ds)
                        ds = None
                        self.format = 'gdal'
                    except Exception as e:
                        raise RasterError('Unable to open dataset %s because:\n%s' %
                                          (input_data, e))
        # If not a string, maybe an osgeo dataset?
        elif isinstance(input_data, gdal.Dataset):
            self.load_from_gdal(input_data)
            self.format = 'gdal'
        # ...or an h5py dataset
        elif isinstance(input_data, h5py.File):
            self.load_from_hdf5(input_data)
            self.format = 'HDF5'
        # ...or a raster instance
        elif isinstance(input_data, raster):
            # Create pointer to other instance (same object)
            self.__dict__ = input_data.__dict__
            # Garbage is only collected if one instance exists
            if hasattr(self, 'garbage'):
                self.garbage['num'] += 1
        else:
            raise RasterError("Unknown input data of type {}\n"
                              "Data may be one of:\n"
                              "path\ngdal.Dataset\n"
                              "h5py.File\n"
                              "raster instance\n"
                              "kwargs to build new raster".format(type(input_data).__name__))

        # Populate other attributes
        self.activeBand = 1
        # Set default interpolation- this should be changed manually to
        #   configure all interpolations of this raster.  Use one of:
        #     'bilinear',
        #     'nearest',
        #     'cubic',
        #     'cubic spline',
        #     'lanczos',
        #     'mean',
        #     'mode'
        self.interpolationMethod = 'nearest'
        self.useChunks = True
        if NUMEXPR:
            self.aMethod = 'ne'
        else:
            self.aMethod = 'np'

    def load_from_gdal(self, ds):
        '''Load attributes from a gdal raster'''
        self.projection = ds.GetProjectionRef()
        gt = ds.GetGeoTransform()
        self.left = float(gt[0])
        self.csx = float(gt[1])
        self.top = float(gt[3])
        self.csy = float(abs(gt[5]))
        self.shape = (ds.RasterYSize, ds.RasterXSize)
        self.bottom = self.top - (self.csy * self.shape[0])
        self.right = self.left + (self.csx * self.shape[1])
        self.bandCount = ds.RasterCount
        band = ds.GetRasterBand(1)
        dtype = gdal.GetDataTypeName(band.DataType)
        if dtype.lower() == 'byte':
            self.dtype = 'uint8'
        else:
            self.dtype = dtype.lower()
        self.nodataValues = []
        for i in range(1, self.bandCount + 1):
            band = ds.GetRasterBand(i)
            nd = band.GetNoDataValue()
            if nd is None:
                self.nodataValues.append(numpy.nan)
            else:
                self.nodataValues.append(nd)
        self.path = ds.GetFileList()[0]
        self.ndchecked = False

    def load_from_hdf5(self, ds):
        """Load attributes from an HDF5 file"""
        self.__dict__.update({key: (None if not isinstance(val, numpy.ndarray) and val == 'None' else val)
                              for key, val in dict(ds.attrs).iteritems()})
        self.shape = tuple(self.shape)
        self.path = str(ds.filename)

    def new_hdf5_raster(self, output_path, compress):
        """
        Create a new HDF5 data source for raster.  Factory for 'save' and 'empty'.
        :param output_path: path for raster
        :return: None
        """
        # Do not overwrite self
        if os.path.normpath(output_path).lower() == os.path.normpath(self.path).lower() and self.mode == 'r':
            raise RasterError('Cannot overwrite the source dataset because it is open as read-only')

        # Create new file
        with h5py.File(output_path, mode='w', libver='latest') as newfile:
            # Copy data from data source to new file
            prvb = self.activeBand
            # File may not exist yet if being built from new raster
            if hasattr(self, 'path'):
                chunks = self.chunks
            else:
                chunks = (256, 256)
            # Add attributes to new file
            newfile.attrs.update({key: ('None' if not isinstance(val, numpy.ndarray) and val == None else val)
                                  for key, val in self.__dict__.iteritems()
                                  if key not in ['garbage', 'mode', 'path']})
            newfile.attrs.update({'format': 'HDF5', 'bottom': self.top - (self.csy * self.shape[0]),
                                  'right': self.left + (self.csx * self.shape[1])})

            for band in self.bands:
                if compress:
                    compression = 'lzf'
                else:
                    compression = None
                ds = newfile.create_dataset(str(band), self.shape,
                                            dtype=self.dtype,
                                            compression=compression,
                                            chunks=chunks)

    def save(self, output_path, compress=True):
        """
        Save to a new file, and use with self
        Note, using data overrides the empty arg
        """
        # Path extension dictates output type
        extension = output_path.split('.')[-1].lower()

        # If HDF5:
        if extension == 'h5':
            #  Create new HDF5 file
            self.new_hdf5_raster(output_path, compress=compress)

            # Transfer data
            out_rast = raster(output_path, mode='r+')
            for _ in self.bands:
                out_rast.activeBand = self.activeBand
                if self.useChunks:
                    for a, s in self.iterchunks():
                        out_rast[s] = a
                else:
                    out_rast[:] = self.array

        # If GDAL:
        elif extension in self.gdal_drivers.keys():
            # Create data source
            outraster = self.new_gdal_raster(output_path, self.shape,
                                             self.bandCount, self.dtype,
                                             self.left, self.top, self.csx,
                                             self.csy, self.projection,
                                             self.chunks, compress)

            # Add data and nodata attributes
            for band in self.bands:
                outraster.activeBand = band
                with outraster.dataset as ds:
                    try:
                        ds.GetRasterBand(outraster.band).SetNoDataValue(self.nodata)
                    except TypeError:
                        ds.GetRasterBand(outraster.band).SetNoDataValue(float(self.nodata))
                ds = None
                if self.useChunks:
                    for a, s in self.iterchunks():
                        outraster[s] = a
                else:
                    outraster[:] = self.array
            del outraster

        # Unsure...
        else:
            raise RasterError(
                'Unknown file type, or file type not implemented yet: {}'.format(output_path.split('.')[-1])
            )

    def build_new_raster(self, path, **kwargs):
        """
        Build a new raster from a set of keyword args
        """
        # Force to HDF5 TODO: Provide support for GDAL types
        if path.split('.')[-1].lower() != 'h5':
            path += '.h5'
        # Get kwargs- for building the new raster
        self.projection = parse_projection(
            kwargs.get('projection', None))
        self.csy = float(kwargs.get('csy', 1))
        self.csx = float(kwargs.get('csx', 1))
        self.dtype = kwargs.get('dtype', None)
        self.left = float(kwargs.get('left', 0))
        self.shape = kwargs.get('shape', None)
        self.interpolationMethod = kwargs.get('interpolationMethod',
                                              'nearest')
        compress = kwargs.get('compress', True)
        self.useChunks = kwargs.get('useChunks', True)
        data = kwargs.get('data', None)
        if data is None and self.shape is None:
            raise RasterError('Either "data" or "shape" must be specified'
                              ' when building a new raster.')
        if NUMEXPR:
            self.aMethod = 'ne'
        else:
            self.aMethod = 'np'

        # Build it
        if data is not None:
            # Get specs from data
            data = numpy.array(data)
            if data.ndim == 1:
                data = data.reshape((1, data.shape[0]))
            self.shape = (data.shape[0], data.shape[1])
            if self.dtype is None:
                self.dtype = data.dtype.name
            else:
                data = data.astype(self.dtype)
            # If array is 3-D, third dimension (axis 2) are the bands
            if data.ndim == 3:
                self.bandCount = data.shape[2]
            else:
                self.bandCount = 1
            self.dtype = data.dtype.name
        else:
            # Load specs from shape
            # If the shape is 3-D, third dimension (axis 2) are the bands
            if len(self.shape) == 3:
                self.bandCount = self.shape[2]
            else:
                # If only one row and 1D, change to 2D
                if len(self.shape) == 1:
                    self.shape = (1, self.shape[0])
                self.bandCount = 1
            self.shape = (self.shape[0], self.shape[1])
            if self.dtype is None:
                self.dtype = 'float32'
        # Use shape to build arbitrary top if not specified
        self.top = float(kwargs.get('top', self.shape[0] * self.csy))
        self.nodataValues = kwargs.get('nodata', None)
        if self.nodataValues is None:
            self.nodataValues = [0 for i in self.bands]
        self.nodataValues = numpy.array(self.nodataValues, dtype=self.dtype).reshape(self.bandCount)

        self.activeBand = 1
        self.activeBand = 1
        self.format = 'HDF5'
        self.mode = 'r+'
        self.path = path

        if data is None:
            # Data will be broadcast to nodata if it is not specified
            new_rast = self.full(self.nodataValues[0], path, compress=compress)
        else:
            # TODO: Change to support bands
            new_rast = self.empty(path, compress=compress)
            new_rast[:] = data
        self.__dict__.update(new_rast.__dict__)

    def copy(self, file_suffix='copy', format='h5'):
        """
        Create a copy of the underlying dataset to use for writing
        temporarily.  Used when mode is 'r' and processing is required,
        or self is instantiated using another raster instance.
        Defaults to HDF5 format.
        """
        path = generate_name(self.path, file_suffix, format)
        # Write new file and return raster
        self.save(path)
        new_rast = raster(path)
        # new_rast is temporary, so prepare garbage
        new_rast.garbage = {'path': path, 'num': 1}
        new_rast.mode = 'r+'
        return new_rast

    def empty(self, path=None, compress=True):
        """
        Return an empty copy of the raster- fast for empty raster instantiation
        :param path: output path if desired
        :param compress: Compress the output
        :return: raster instance
        """
        if path is None:
            out_path = generate_name(self.path, 'copy', 'h5')
        else:
            out_path = path
        #  Create new HDF5 file
        self.new_hdf5_raster(out_path, compress=compress)

        # Add to garbage if temporary, and make writeable
        outrast = raster(out_path)
        if path is None:
            outrast.garbage = {'path': out_path, 'num': 1}
        outrast.mode = 'r+'
        return outrast

    def full(self, data, path=None, compress=True):
        """
        Return a copy of the raster filled with the input data
        :param data: Array or scalar to be used to fill the raster
        :param path: output path if not temporary
        :param compress: Compress the output or not
        :return: raster instance
        """
        outrast = self.empty(path=path, compress=compress)
        # h5py is very slow to broadcast and write- broadcast in advance
        data = numpy.broadcast_to(data, self.shape)
        outrast[:] = data
        return outrast

    @property
    def size(self):
        return (float(self.itemsize) * self.shape[0] * self.shape[1] *
                self.bandCount / 1E9)

    @property
    def nodata(self):
        return self.nodataValues[self.band - 1]

    @property
    @contextmanager
    def dataset(self):
        if self.mode not in ['r', 'r+']:
            raise RasterError('Unrecognized file mode "%s"' % self.mode)
        if self.format == 'gdal':
            if self.mode == 'r':
                mode = gdalconst.GA_ReadOnly
            else:
                mode = gdalconst.GA_Update
            ds = gdal.Open(self.path, mode)
            if ds is None:
                raise RasterError('Oops...the raster %s is now missing.' %
                                  self.path)
            yield ds
            # Somehow need to find a way to safely close a gdal dataset using
            #   the context manager.  You will note "ds = None" after with
            #   statements everywhere the self.format == 'gdal'

            # Just makes the local ds None!
            #   |
            #   V
            ds = None
        elif self.format == 'HDF5':
            ds = h5py.File(self.path, mode=self.mode, libver='latest')
            yield ds
            ds.close()

    @property
    def band(self):
        try:
            b = int(self.activeBand)
            assert b <= self.bandCount and b > 0
        except:
            raise RasterError('Active band "%s" cannot be accessed' %
                              self.activeBand)
        return b

    @property
    def bands(self):
        for i in range(1, self.bandCount + 1):
            self.activeBand = i
            yield i

    @property
    def array(self):
        """
        Load underlying data into a numpy array
        """
        if self.format == 'gdal':
            with self.dataset as ds:
                a = ds.GetRasterBand(self.band).ReadAsArray()
            ds = None
        else:
            with self.dataset as ds:
                a = ds[str(self.band)][:]
        return a

    @property
    def chunks(self):
        """Chunk shape of self"""
        if hasattr(self, 'chunk_override'):
            chunks = self.chunk_override
        elif self.format == 'gdal':
            with self.dataset as ds:
                chunks = ds.GetRasterBand(self.band).GetBlockSize()
                # If no blocks exist, a single scanline will result
                if chunks[0] == self.shape[1] and chunks[1] == 1:
                    lines = int((256 * 256 * self.itemsize) /
                                (self.shape[0] * self.itemsize))
                    chunks = (lines, self.shape[1])
                else:
                    # Reverse to match numpy index notation
                    chunks = (chunks[1], chunks[0])
            ds = None
        else:
            try:
                with self.dataset as ds:
                    chunks = ds[str(self.band)].chunks
            except:
                chunks = (256, 256)
        if chunks[0] > self.shape[0]:
            chunks = (int(self.shape[0]), chunks[1])
        if chunks[1] > self.shape[1]:
            chunks = (chunks[0], int(self.shape[1]))
        return chunks

    @property
    def itemsize(self):
        """Return number of bytes/element for a specific dtype"""
        return numpy.dtype(self.dtype).itemsize

    @property
    def mgrid(self):
        """Return arrays of the coordinates of all raster grid cells"""
        top_c = self.top - (self.csy * 0.5)
        left_c = self.left + (self.csx * 0.5)
        ishape, jshape = self.shape
        return numpy.mgrid[top_c:top_c - (self.csy * (ishape - 1)):ishape * 1j,
                           left_c:left_c + (self.csx *
                                            (jshape - 1)):jshape * 1j]

    @property
    def interpolation(self):
        interp_methods = {
            'bilinear': gdalconst.GRA_Bilinear,
            'nearest': gdalconst.GRA_NearestNeighbour,
            'cubic': gdalconst.GRA_Cubic,
            'cubic spline': gdalconst.GRA_CubicSpline,
            'lanczos': gdalconst.GRA_Lanczos,
            'mean': gdalconst.GRA_Average,
            'mode': gdalconst.GRA_Mode
        }
        try:
            return interp_methods[self.interpolationMethod]
        except KeyError:
            raise RasterError('Unrecognized interpolation method %s' %
                              self.interpolationMethod)

    def use_chunks(self, memory):
        '''
        Determine whether chunks need to be used, given the supplied memory arg
        '''
        if self.size > memory:
            return True
        else:
            return False

    def iterchunks(self, custom_chunks=None, fill_cache=True, expand=0):
        '''
        Generate an array and slice over the dataset.
            custom_chunks: tuple, (height, width)
            fill_cache: Return as many tiles as can fit in GDAL cache
            expand: Number of extra rows/cols on all sides
        '''
        if custom_chunks is None:
            # Regenerate chunks
            chunks = self.chunks
        else:
            try:
                chunks = map(int, custom_chunks)
                assert len(custom_chunks) == 2
            except:
                raise RasterError('Custom chunks must be a tuple or list of'
                                  ' length 2 containing integers')

        # Parse expand arg (for chunk overlap)
        if type(expand) in [int, float, numpy.ndarray]:
            i = j = int(expand)
        elif type(expand) == tuple or type(expand) == list:
            i, j = map(int, expand)
        else:
            raise RasterError('Unable to interpret the expand argument "%s" of'
                              ' type %s' % (expand, type(expand).__name__))
        # Change expand to an edge
        if i != 0:
            ifr = int(numpy.ceil((i - 1) / 2.))
            ito = int((i - 1) / 2.)
        else:
            ifr = 0
            ito = 0
        if j != 0:
            jfr = int(numpy.ceil((j - 1) / 2.))
            jto = int((j - 1) / 2.)
        else:
            jfr = 0
            jto = 0

        # Expand chunks to fill cache
        if fill_cache:
            cache_chunks = gdal.GetCacheMax() / (chunks[0] * chunks[1] *
                                                 self.itemsize)
            # Propagate number of cache chunks in the x-dimension first
            x_chunks = int(numpy.ceil(float(self.shape[1]) / chunks[1]))
            # Compute the residual cache chunks overflowing the x-dimension
            resid = cache_chunks - x_chunks
            # Check whether chunks may propagate the x-dimension more than one
            #   more time, and occupy some of the y-dimension
            if resid > 0:
                chunks_j = self.shape[1]
                chunks_i = chunks[0] * ((resid / x_chunks) + 1)
            else:
                chunks_j = chunks[1] * cache_chunks
                chunks_i = chunks[0]
        else:
            chunks_i, chunks_j = chunks

        ychunks = range(0, self.shape[0], chunks_i) + [self.shape[0]]
        xchunks = range(0, self.shape[1], chunks_j) + [self.shape[1]]
        ychunks = zip(numpy.array(ychunks[:-1]) - ifr,
                      numpy.array(ychunks[1:]) + ito)
        ychunks[0] = (0, ychunks[0][1])
        ychunks[-1] = (ychunks[-1][0], self.shape[0])
        xchunks = zip(numpy.array(xchunks[:-1]) - jfr,
                      numpy.array(xchunks[1:]) + jto)
        xchunks[0] = (0, xchunks[0][1])
        xchunks[-1] = (xchunks[-1][0], self.shape[1])

        # Create a generator out of slices
        for ych in ychunks:
            for xch in xchunks:
                s = (slice(ych[0], ych[1]), slice(xch[0], xch[1]))
                yield self[s[0], s[1]], s

    @staticmethod
    def get_gdal_dtype(dtype):
        """Return a gdal data type from a numpy counterpart"""
        datatypes = {
            'int8': 'Int16',
            'bool': 'Byte',
            'uint8': 'Byte',
            'int16': 'Int16',
            'int32': 'Int32',
            'uint16': 'UInt16',
            'uint32': 'UInt32',
            'uint64': 'UInt32',
            'float32': 'Float32',
            'float64': 'Float64',
            'int64': 'Int32'
        }
        try:
            return gdal.GetDataTypeByName(datatypes[dtype])
        except KeyError:
            raise RasterError('Unrecognized data type "%s" encountered while'
                              ' trying to save a GDAL raster' % dtype)

    def astype(self, dtype):
        '''Change the data type of self and return copy'''
        # Check the input
        dtype = str(dtype)
        try:
            dtype = dtype.lower()
            assert dtype in ['bool', 'int8', 'uint8', 'int16', 'uint16',
                             'int32', 'uint32', 'int64', 'uint64', 'float32',
                             'float64']
        except:
            raise RasterError('Unrecognizable data type "%s"' % dtype)
        # Ensure they are already not the same
        if dtype == self.dtype:
            return self.copy(dtype)

        # Create a copy with new data type
        prvdtype = self.dtype
        self.dtype = dtype
        out = self.empty()
        self.dtype = prvdtype

        # Complete casting
        for band in self.bands:
            out.activeBand = band
            for a, s in self.iterchunks():
                out[s] = a.astype(dtype)
            out.nodataValues[band - 1] =\
                numpy.asscalar(numpy.array(out.nodata).astype(dtype))

        # Return casted raster
        return out

    def define_projection(self, projection):
        '''Define the raster projection'''
        if (self.projection != '' and
                self.projection is not None) and self.mode == 'r':
            raise RasterError('The current raster already has a spatial'
                              ' reference, use mode="r+" to replace.')
        self.projection = parse_projection(projection)
        # Write projection to output files
        if self.format == 'gdal':
            with self.dataset as ds:
                ds.SetProjection(parse_projection(projection))
            ds = None
            del ds
        else:
            # Add as attribute
            with self.dataset as ds:
                ds.attrs['projection'] = projection

    def match_raster(self, input_raster, tolerance=1E-05):
        """
        Align extent and cells with another raster
        """
        def isclose(input, values):
            values = [(val - tolerance, val + tolerance) for val in values]
            if any([lower < input < upper for lower, upper in values]):
                return True
            else:
                return False

        inrast = raster(input_raster)

        # Check if spatial references align
        insrs = osr.SpatialReference()
        insrs.ImportFromWkt(self.projection)
        outsrs = osr.SpatialReference()
        outsrs.ImportFromWkt(inrast.projection)
        samesrs = insrs.IsSame(outsrs)
        inrast_bbox = extent(inrast).bounds

        # Check if cells align
        if all([isclose(self.csx, [inrast.csx]),
                isclose(self.csy, [inrast.csy]),
                isclose((self.top - inrast.top) % self.csy, [0, self.csy]),
                isclose((self.left - inrast.left) % self.csx, [0, self.csx]),
                samesrs]):
            # Simple slicing is sufficient
            return self.clip(inrast_bbox)
        else:
            # Transform required
            print "Transforming to match rasters..."
            if samesrs:
                return self.transform(csx=inrast.csx, csy=inrast.csy,
                                      extent=inrast_bbox,
                                      interpolation=self.interpolation)
            else:
                return self.transform(csx=inrast.csx, csy=inrast.csy,
                                      projection=inrast.projection, extent=inrast_bbox,
                                      interpolation=self.interpolation)

    def slice_from_bbox(self, bbox):
        """
        Compute slice objects to using a bbox.

        Returns slicer for self, another for an array the size of bbox, and
        the shape of the output array from bbox.
        """
        top, bottom, left, right = map(float, bbox)
        if any([top < bottom,
                bottom > top,
                right < left,
                left > right]):
                raise RasterError('Bounding box is invalid, check that the'
                                  ' coordinate positions are (top, bottom,'
                                  ' left, right)')
        # Snap to the nearest cell boundary
        resid = (self.top - top) / self.csy
        top += (resid - int(round(resid))) * self.csy
        if top > self.top:
            top = self.top
        resid = (self.top - bottom) / self.csy
        bottom += (resid - int(round(resid))) * self.csy
        if bottom < self.bottom:
            bottom = self.bottom
        resid = (self.left - left) / self.csx
        left += (resid - int(round(resid))) * self.csx
        if left < self.left:
            left = self.left
        resid = (self.left - right) / self.csx
        right += (resid - int(round(resid))) * self.csx
        if right > self.right:
            right = self.right

        # Compute shape and slices
        shape = (int(round((top - bottom) / self.csy)),
                 int(round((right - left) / self.csx)))
        i = int(round((self.top - top) / self.csy))
        if i < 0:
            i_ = abs(i)
            i = 0
        else:
            i_ = 0
        j = int(round((left - self.left) / self.csx))
        if j < 0:
            j_ = abs(j)
            j = 0
        else:
            j_ = 0
        _i = int(round((self.top - bottom) / self.csy))
        if _i > self.shape[0]:
            _i_ = shape[0] - (_i - self.shape[0])
            _i = self.shape[0]
        else:
            _i_ = shape[0]
        _j = int(round((right - self.left) / self.csx))
        if _j > self.shape[1]:
            _j_ = shape[1] - (_j - self.shape[1])
            _j = self.shape[1]
        else:
            _j_ = shape[1]

        return ((slice(i, _i), slice(j, _j)),
                (slice(i_, _i_), slice(j_, _j_)),
                shape,
                (top, bottom, left, right))

    def clip(self, bbox_or_dataset):
        """
        Slice self using bounding box coordinates.

        Note: the bounding box may not be honoured exactly.  To accomplish this
        use a transform.
        :param bbox_or_dataset: raster, vector, or bbox (top, bottom, left, right)
        """
        # Get input
        clipper = assert_type(bbox_or_dataset)(bbox_or_dataset)
        bbox = extent(clipper).transform(self.projection).bounds
        if isinstance(clipper, vector):
            vector_mask = clipper.transform(self.projection)
        else:
            vector_mask = None

        # Check that bbox is not inverted
        if any([bbox[0] <= bbox[1], bbox[2] >= bbox[3]]):
            raise RasterError('Input bounding box appears to be inverted'
                              ' or has null dimensions')

        # Get slices
        self_slice, insert_slice, shape, bbox = self.slice_from_bbox(bbox)

        # Check if no change
        if all([self.top == bbox[0], self.bottom == bbox[1],
                self.left == bbox[2], self.right == bbox[3],
               vector_mask is None]):
            return self.copy('clip')

        # Create output dataset with new extent
        path = generate_name(self.path, 'clip', 'h5')
        kwargs = {
            'projection': self.projection,
            'csx': self.csx,
            'csy': self.csy,
            'shape': shape + (self.bandCount,),
            'dtype': self.dtype,
            'top': bbox[0],
            'left': bbox[2],
            'nodata': self.nodata,
            'interpolationMethod': self.interpolationMethod,
            'useChunks': self.useChunks
        }
        outds = raster(path, mode='w', **kwargs)
        # If a cutline is specified, use it to create a mask
        if vector_mask is not None:
            cutline = vector_mask.rasterize(outds.path).array.astype('bool')

        # Add output to garbage
        outds.garbage = {'path': path, 'num': 1}
        for _ in outds.bands:
            insert_array = self[self_slice]
            if vector_mask is not None:
                insert_array[~cutline] = outds.nodata
            outds[insert_slice] = insert_array
        return outds

    def clip_to_data(self):
        """
        Change raster extent to include the minimum bounds where data exist
        :return: raster instance
        """
        i, j = numpy.where(self.array != self.nodata)
        y, x = indices_to_coords(([i.min(), i.max()], [j.min(), j.max()]),
                                 self.top, self.left, self.csx, self.csy)
        return self.clip((y[0] + self.csy / 2, y[1] - self.csy / 2,
                          x[0] - self.csx / 2, x[1] + self.csx / 2))

    def transform(self, **kwargs):
        """
        Change cell size, projection, or extent.
        ------------------------
        In Args

        "projection": output projection argument
            (wkt, epsg, osr.SpatialReference, raster instance)

        "csx": output cell size in the x-direction

        "cxy": output cell size in the y-direction

        "extent": (extent instance) bounding box for output
            Note- if extent does not have a defined coordinate system,
            it is assumed to be in the output spatial reference

        "template": other raster instance, overrides all other arguments and projects into the raster
        """
        template = kwargs.get('template', None)
        if template is None:
            # Get keyword args to determine what's done
            csx = kwargs.get('csx', None)
            if csx is not None:
                csx = float(csx)
            csy = kwargs.get('csy', None)
            if csy is not None:
                csy = float(csy)
            projection = parse_projection(kwargs.get('projection', None))
            output_extent = kwargs.get('extent', None)
            if output_extent is None:
                bbox = extent(self)
            else:
                bbox = extent(output_extent)

            # Check for spatial reference change
            if projection != '':
                insrs = osr.SpatialReference()
                insrs.ImportFromWkt(self.projection)
                outsrs = osr.SpatialReference()
                outsrs.ImportFromWkt(projection)
                if insrs.IsSame(outsrs):
                    insrs, outsrs = None, None
            else:
                insrs, outsrs = None, None
            if all([i is None for i in [insrs, outsrs, csx, csy]] + [bbox == extent(self)]):
                print ("Warning: No transformation operation was necessary")
                return self.copy('transform')

            # Recalculate the extent and calculate potential new cell sizes if a coordinate system change is necessary
            if insrs is not None:
                # Get the corners and transform the extent
                bbox = bbox.transform(outsrs)

                # Calculate the new cell size
                points = [(self.left + self.csx, self.top), (self.left, self.top)]
                points = transform_points(points, self.projection, projection)[:2]
                ncsx = points[0][0] - points[1][0]
                points = [(self.right, self.bottom), (self.right - self.csx, self.bottom)]
                points = transform_points(points, self.projection, projection)[:2]
                ncsx = min([points[0][0] - points[1][0], ncsx])
                points = [(self.left, self.top), (self.left, self.top - self.csy)]
                points = transform_points(points, self.projection, projection)[:2]
                ncsy = points[0][1] - points[1][1]
                points = [(self.right, self.bottom + self.csy), (self.right, self.bottom)]
                points = transform_points(points, self.projection, projection)[:2]
                ncsy = min([points[0][1] - points[1][1], ncsy])
            else:
                ncsx, ncsy = self.csx, self.csy

            top, bottom, left, right = bbox.bounds

            # Snap the potential new cell sizes to the extent
            ncsx = (right - left) / int(round((right - left) / ncsx))
            ncsy = (top - bottom) / int(round((top - bottom) / ncsy))

            # One of extent or cell sizes must be updated to match depending on args
            if output_extent is not None:
                # Check that cell sizes are compatible if they are inputs
                if csx is not None:
                    xresid = round((bbox.bounds[3] - bbox.bounds[2]) % csx, 5)
                    if xresid != round(csx, 5) and xresid != 0:
                        raise RasterError('Transform cannot be completed due to an'
                                          ' incompatible extent %s and cell size (%s) in'
                                          ' the x-direction' % ((bbox.bounds[3], bbox.bounds[2]), csx))
                else:
                    # Use ncsx
                    csx = ncsx
                if csy is not None:
                    yresid = round((bbox.bounds[0] - bbox.bounds[1]) % csy, 5)
                    if yresid != round(csy, 5) and yresid != 0:
                        raise RasterError('Transform cannot be completed due to an'
                                          ' incompatible extent %s and cell size (%s) in'
                                          ' the y-direction' % ((bbox.bounds[0], bbox.bounds[1]), csy))
                else:
                    # Use ncsy
                    csy = ncsy
            else:
                # Use the cell size to modify the output extent
                if csx is None:
                    csx = ncsx
                if csy is None:
                    csy = ncsy

                # Compute the shape using the existing extent and input cell sizes
                shape = (int(round((top - bottom) / csy)), int(round(right - left) / csx))

                # Expand extent to fit cell size
                resid = (right - left) - (csx * shape[1])
                if resid < 0:
                    resid /= -2.
                elif resid > 0:
                    resid = resid / (2. * csx)
                    resid = (numpy.ceil(resid) - resid) * csx
                left -= resid
                right += resid

                # Expand extent to fit cell size
                resid = (top - bottom) - (csy * shape[0])
                if resid < 0:
                    resid /= -2.
                elif resid > 0:
                    resid = resid / (2. * csy)
                    resid = (numpy.ceil(resid) - resid) * csy
                bottom -= resid
                top += resid
                bbox = extent((top, bottom, left, right))

            # Compute new shape
            shape = (int(round((bbox.bounds[0] - bbox.bounds[1]) / csy)),
                     int(round((bbox.bounds[3] - bbox.bounds[2]) / csx)))

            # Create output raster dataset
            if insrs is not None:
                insrs = insrs.ExportToWkt()
            if outsrs is not None:
                outsrs = outsrs.ExportToWkt()
                output_srs = outsrs
            else:
                output_srs = self.projection

        else:
            t = raster(template)
            insrs = self.projection
            outsrs = t.projection
            output_srs = t.projection
            shape, bbox, csx, csy = t.shape, extent(t), t.csx, t.csy

        # Cast to floating point if the interpolation method is not nearest
        if 'int' in self.dtype.lower() and self.interpolationMethod != 'nearest':
            dtype = 'float32'
        else:
            dtype = self.dtype

        path = generate_name(self.path, 'transform', 'tif')
        out_raster = self.new_gdal_raster(path, shape, self.bandCount,
                                          dtype, bbox.bounds[2], bbox.bounds[0], csx,
                                          csy, output_srs, (256, 256))

        # Set/fill nodata value
        for band in out_raster.bands:
            with out_raster.dataset as ds:
                ds.GetRasterBand(out_raster.band).SetNoDataValue(self.nodata)
            ds = None
            for a, s in out_raster.iterchunks():
                shape = self.gdal_args_from_slice(s, self.shape)
                shape = (shape[3], shape[2])
                out_raster[s] = numpy.full(shape, self.nodata, dtype)
        with out_raster.dataset as outds:
            # The context manager does not work with ogr datasets
            pass

        # Direct to input raster dataset
        if self.format != 'gdal':
            with self.copy('copy', 'tif') as input_raster:
                with input_raster.dataset as inds:
                    # The context manager does not work with ogr datasets
                    pass
                    gdal.ReprojectImage(inds, outds, insrs, outsrs, self.interpolation)
        else:
            with self.dataset as inds:
                # The context manager does not work with ogr datasets
                pass
                gdal.ReprojectImage(inds, outds, insrs, outsrs, self.interpolation)
        inds, outds = None, None

        # Return new raster
        outrast = raster(path)
        # This is temporary as an output
        outrast.garbage = {'path': path, 'num': 1}
        outrast.mode = 'r+'
        return outrast

    def fix_nodata(self):
        """Fix no data values to be identifiable if rounding errors exist"""
        for band in self.bands:
            if self.useChunks:
                for a, s in self.iterchunks():
                    close = numpy.isclose(a, self.nodata)
                    if numpy.any(close) and numpy.all(a != self.nodata):
                        self.nodataValues[band - 1] =\
                            numpy.asscalar(a[close][0])
                        break
            else:
                a = self.array
                close = numpy.isclose(a, self.nodata)
                if numpy.any(close) and numpy.all(a != self.nodata):
                    self.nodataValues[band - 1] = numpy.asscalar(a[close][0])

    def new_gdal_raster(self, output_path, shape, bands, dtype, left, top, csx, csy,
                        projection, chunks, compress=True):
        """Generate a new gdal raster dataset"""
        extension = output_path.split('.')[-1].lower()
        if extension not in self.gdal_drivers.keys():
            raise RasterError(
                'Unknown file type, or file type not implemented yet: {}'.format(output_path.split('.')[-1])
            )
        driver = gdal.GetDriverByName(self.gdal_drivers[extension])
        # Only tif currently set up, so these are the hard-coded compression and writing parameters
        if compress:
            comp = 'COMPRESS=LZW'
        else:
            comp = 'COMPRESS=NONE'
        # When the chunk sizes are close to the overall shape the TIFF driver
        #   goes bonkers, so do not tile in this case.
        if chunks[1] * 2 > shape[1] or chunks[0] * 2 > shape[0]:
            blockxsize, blockysize, tiled = ('BLOCKXSIZE=0', 'BLOCKYSIZE=0',
                                             'TILED=NO')
        else:
            blockxsize = 'BLOCKXSIZE=%s' % chunks[1]
            blockysize = 'BLOCKYSIZE=%s' % chunks[0]
            tiled = 'TILED=YES'
        size = (shape[0] * shape[1] * float(numpy.dtype(dtype).itemsize)) / 1E09
        if size > 4.:
            bigtiff='BIGTIFF=YES'
        else:
            bigtiff = 'BIGTIFF=NO'
        parszOptions = [tiled, blockysize, blockxsize, comp, bigtiff]
        ds = driver.Create(output_path, int(shape[1]), int(shape[0]),
                           bands, raster.get_gdal_dtype(dtype),
                           parszOptions)
        if ds is None:
            raise RasterError('GDAL error trying to create new raster.')
        ds.SetGeoTransform((left, float(csx), 0, top, 0, csy * -1.))
        projection = parse_projection(projection)
        ds.SetProjection(projection)
        ds = None
        outraster = raster(output_path, mode='r+')
        return outraster

    def polygonize(self):
        """
        Factory for raster.vectorize to wrap the gdal.Polygonize method
        :return: vector instance
        """
        # Create a .tif if necessary
        raster_path, garbage = force_gdal(self)
        input_raster = raster(raster_path)

        # Allocate an output vector
        vector_path = generate_name(self.path, 'polygonize', 'shp')
        outvect = vector(vector_path, mode='w', geotype='Polygon', projection=self.projection)
        outvect.add_fields('raster_val', self.dtype)

        with input_raster.dataset as ds:
            band = ds.GetRasterBand(self.band)
            maskband = band.GetMaskBand()
            with outvect.layer() as out_layer:
                gdal.Polygonize(band, maskband, out_layer, 0, ['8CONNECTED=8'])

        # Clean up
        ds = None
        band = None

        if garbage:
            try:
                os.remove(raster_path)
            except:
                pass

        outvect.__dict__.update(vector(vector_path).__dict__)
        outvect.garbage = {'path': vector_path, 'num': 1}
        return outvect

    def vectorize(self, geotype='Polygon', **kwargs):
        """
        Create a polygon, line, or point vector from the raster
        :param geotype: The geometry type of the output vector.  Choose from 'Polygon', 'LineString', and 'Point'
        :param kwargs:
            centroid=False Use the centroid of the raster regions when creating points
        :return: vector instance
        """
        if geotype == 'Polygon':
            # Apply gdal.Polygonize method
            return self.polygonize()

        elif geotype == 'LineString':
            raise NotImplementedError('This is still in development')

        elif geotype == 'Point':
            a = self.array

            # If centroid is specified, label the raster and grab the respective points
            if kwargs.get('centroid', False):
                labels, number = sklabel(a, return_num=True, background=self.nodata)
                properties = regionprops(labels)
                values = [a[tuple(properties[i].coords[0])] for i in range(number)]
                coords = numpy.array([properties[i].centroid for i in range(number)])
                coords[:, 0] = (self.top - (self.csy / 2)) - (coords[:, 0] * self.csy)
                coords[:, 1] = (self.left + (self.csx / 2)) + (coords[:, 1] * self.csx)
                return vector([shpwkb.dumps(pnt) for pnt in geometry.MultiPoint(numpy.fliplr(coords))],
                              fields=numpy.array(values, dtype=[('raster_val', self.dtype)]),
                              projection=self.projection)

            # Grab the coordinates and make a vector using shapely wkb string dumps
            m = a != self.nodata
            y, x = indices_to_coords(numpy.where(m), self.top, self.left, self.csx, self.csy)

            return vector([shpwkb.dumps(pnt) for pnt in geometry.MultiPoint(zip(x, y)).geoms],
                          fields=numpy.array(a[m], dtype=[('raster_val', self.dtype)]),
                          projection=self.projection)

    def extent_to_vector(self, as_mask=False):
        """
        Write the current raster extent to a shapefile
        :param as_mask: Return a vector of values with data in the raster.  Otherwise the image bounds are used.
        :return: vector instance
        """
        if as_mask:
            # Polygonize mask
            return self.mask.vectorize('Polygon')
        else:
            # Create a wkb from the boundary of the raster
            geo = geometry.Polygon(extent(self).corners)
            _wkb = shpwkb.dumps(geo)

            # Create an output shapefile from geometry
            return vector([_wkb], projection=self.projection)

    @property
    def mask(self):
        """Return a raster instance of the data mask (boolean)"""
        return self != self.nodata

    @staticmethod
    def gdal_args_from_slice(s, shape):
        """Factory for __getitem__ and __setitem__"""
        if type(s) == int:
            xoff = 0
            yoff = s
            win_xsize = shape[1]
            win_ysize = 1
        elif type(s) == tuple:
            # Convert numpy objects to integers
            s = [int(o) if 'numpy' in str(type(o)) else o for o in s]
            if type(s[0]) == int:
                yoff = s[0]
                win_ysize = 1
            elif s[0] is None:
                yoff = 0
                win_ysize = shape[0]
            else:
                yoff = s[0].start
                start = yoff
                if start is None:
                    start = 0
                    yoff = 0
                stop = s[0].stop
                if stop is None:
                    stop = shape[0]
                win_ysize = stop - start
            if type(s[1]) == int:
                xoff = s[1]
                win_xsize = 1
            elif s[1] is None:
                xoff = 0
                win_xsize = shape[1]
            else:
                xoff = s[1].start
                start = xoff
                if start is None:
                    start = 0
                    xoff = 0
                stop = s[1].stop
                if stop is None:
                    stop = shape[1]
                win_xsize = stop - start
        elif type(s) == slice:
            xoff = 0
            win_xsize = shape[1]
            if s.start is None:
                yoff = 0
            else:
                yoff = s.start
            if s.stop is None:
                stop = shape[0]
            else:
                stop = s.stop
            win_ysize = stop - yoff
        return xoff, yoff, win_xsize, win_ysize

    def max(self):
        """
        Collect the maximum
        :return: The maximum value
        """
        if self.useChunks:
            stack = []
            for a, s in self.iterchunks():
                stack.append(numpy.max(a[a != self.nodata]))
            return max(stack)
        else:
            return numpy.max(self.array[self.array != self.nodata])

    def min(self):
        """
        Collect the maximum
        :return: Ummm...the minimum value
        """
        if self.useChunks:
            stack = []
            for a, s in self.iterchunks():
                stack.append(numpy.min(a[a != self.nodata]))
            return min(stack)
        else:
            return numpy.min(self.array[self.array != self.nodata])

    def __getitem__(self, s):
        """
        Slice a raster using numpy-like syntax.
        :param s: item for slicing, may be a slice object, integer, instance of the extent class
        :return:
        """
        # TODO: add boolean, fancy, and extent slicing
        if self.format == 'gdal':
            # Do a simple check for necessary no data value fixes using topleft
            if not self.ndchecked:
                ab = self.activeBand
                for band in self.bands:
                    with self.dataset as ds:
                        tlc = ds.GetRasterBand(self.band).ReadAsArray(
                            xoff=0, yoff=0, win_xsize=1, win_ysize=1)
                    ds = None
                    if numpy.isclose(tlc, self.nodata) and tlc != self.nodata:
                        print "Warning: No data values must be fixed."
                        self.ndchecked = True
                        self.fix_nodata()
                self.activeBand = ab
            # Tease gdal band args from s
            try:
                xoff, yoff, win_xsize, win_ysize =\
                    self.gdal_args_from_slice(s, self.shape)
            except:
                raise RasterError('Boolean and fancy indexing currently'
                                  ' unsupported for GDAL raster data sources.'
                                  ' Convert to HDF5 to use this'
                                  ' functionality.')
            with self.dataset as ds:
                a = ds.GetRasterBand(self.band).ReadAsArray(
                    xoff=xoff, yoff=yoff, win_xsize=win_xsize,
                    win_ysize=win_ysize
                )
            ds = None
        else:
            with self.dataset as ds:
                a = ds[str(self.band)][s]
        return a

    def __setitem__(self, s, a):
        if self.mode == 'r':
            raise RasterError('Dataset open as read-only.')
        a = numpy.asarray(a)
        if self.format == 'gdal':
            xoff, yoff, win_xsize, win_ysize =\
                self.gdal_args_from_slice(s, self.shape)
            if (a.size > 1 and
                    (win_ysize != a.shape[0] or win_xsize != a.shape[1])):
                raise RasterError('Raster data of the shape %s cannot be'
                                  ' replaced with array of shape %s' %
                                  ((win_ysize, win_xsize), a.shape))
            # Broadcast for scalar (future add axis-based broadcasting)
            if a.size == 1:
                a = numpy.full((win_ysize, win_xsize), a, a.dtype)
            with self.dataset as ds:
                ds.GetRasterBand(self.band).WriteArray(a, xoff=xoff, yoff=yoff)
            ds = None
        else:
            # try:
            with self.dataset as ds:
                ds[str(self.band)][s] = a
            # except Exception as e:
            #     raise RasterError('Error writing raster data. Check that mode'
            #                       ' is "r+" and that the arrays match.\n\nMore'
            #                       ' info:\n%s' % e)

    def perform_operation(self, r, op):
        """
        Factory for all operand functions
        :param r: value
        :param op: operand
        :return: raster instance
        """
        if isinstance(r, numbers.Number) or isinstance(r, numpy.ndarray):
            out = self.full(r)
        else:
            try:
                r = raster(r)
            except:
                raise RasterError('Expected a number, numpy array, or valid raster while'
                                  ' using the "%s" operator' % op)
            out = r.match_raster(self)

        outnd = out.nodata
        nd = self.nodata

        if self.useChunks:
            # Compute over chunks
            for a, s in self.iterchunks():
                b = out[s]
                if self.aMethod == 'ne':
                    out[s] = ne.evaluate('where((a!=nd)&(b!=outnd),a%sb,'
                                         'outnd)' % op)
                elif self.aMethod == 'np':
                    m = (a != nd) & (b != outnd)
                    b = b[m]
                    c = a[m]
                    a[m] = eval('c%sb' % op)
                    a[~m] = outnd
                    out[s] = a
        else:
            # Load into memory, then compute
            a = self.array
            b = out.array
            if self.aMethod == 'ne':
                out[:] = ne.evaluate('where((a!=nd)&(b!=outnd),a%sb,'
                                     'outnd)' % op)
            elif self.aMethod == 'np':
                m = (a != nd) & (b != outnd)
                b = b[m]
                c = a[m]
                a[m] = eval('c%sb' % op)
                a[~m] = outnd
                out[:] = a
        return out

    def perform_i_operation(self, r, op):
        """
        Factory for all i-operand functions
        :param r: number, array, or raster
        :param op: operand
        :return: raster instance
        """
        if self.mode == 'r':
            raise RasterError('%s open as read-only.' %
                              os.path.basename(self.path))
        if isinstance(r, numbers.Number) or isinstance(r, numpy.ndarray):
            r = self.full(r)
        else:
            try:
                r = raster(r)
            except:
                raise RasterError('Expected a number, numpy array, or valid raster while'
                                  ' using the "%s=" operator' % op)
            r.match_raster(self)

        rnd = r.nodata
        nd = self.nodata
        if self.useChunks:
            for a, s in self.iterchunks():
                b = r[s]
                if self.aMethod == 'ne':
                    self[s] = ne.evaluate('where((a!=nd)&(b!=rnd),a%sb,nd)'
                                          % op)
                elif self.aMethod == 'np':
                    m = (a != nd) & (b != rnd)
                    b = b[m]
                    c = a[m]
                    a[m] = eval('c%sb' % op)
                    self[s] = a
        else:
            a = self.array
            b = r.array
            if self.aMethod == 'ne':
                self[:] = ne.evaluate('where((a!=nd)&(b!=rnd),a%sb,nd)'
                                      % op)
            elif self.aMethod == 'np':
                m = (a != nd) & (b != rnd)
                b = b[m]
                c = a[m]
                a[m] = eval('c%sb' % op)
                self[:] = a

    def perform_cond_operation(self, r, op):
        """
        Wrap the numpy conditional operators using underlying raster data
        :param r:
        :param op:
        :return:
        """
        is_array = False
        if any([isinstance(r, t) for t in [numpy.ndarray, int, float]]):
            is_array = True
            r = numpy.broadcast_to(r, self.shape)
        else:
            r = assert_type(r)(r)
            if isinstance(r, vector):
                r = r.rasterize(self)
            elif isinstance(r, raster):
                r = r.match_raster(self)

        # Create a mask raster where rasters match
        mask = self.astype('bool').full(0)
        mask.nodataValues = [0 for i in range(self.bandCount)]

        if self.useChunks:
            for band in self.bands:
                mask.activeBand = band
                if not is_array:
                    r.activeBand = band
                for a, s in self.iterchunks():
                    self_data, comp_data = self[s], r[s]
                    if is_array:
                        m = (self_data != self.nodata) & (getattr(self_data, op)(comp_data))
                    else:
                        m = (self_data != self.nodata) & (comp_data != r.nodata) & (getattr(self_data, op)(comp_data))
                    mask[s] = m
        else:
            for band in self.bands:
                mask.activeBand = band
                if not is_array:
                    r.activeBand = band
                    comp_data = r.array
                else:
                    comp_data = r
                self_data = self.array
                if is_array:
                    m = (self_data != self.nodata) & (getattr(self_data, op)(comp_data))
                else:
                    m = (self_data != self.nodata) & (comp_data != r.nodata) & (getattr(self_data, op)(comp_data))
                mask[:] = m

        return mask

    def __add__(self, r):
        return self.perform_operation(r, '+')

    def __iadd__(self, r):
        self.perform_i_operation(r, '+')
        return self

    def __sub__(self, r):
        return self.perform_operation(r, '-')

    def __isub__(self, r):
        self.perform_i_operation(r, '-')
        return self

    def __div__(self, r):
        return self.perform_operation(r, '/')

    def __idiv__(self, r):
        self.perform_i_operation(r, '/')
        return self

    def __mul__(self, r):
        return self.perform_operation(r, '*')

    def __imul__(self, r):
        self.perform_i_operation(r, '*')
        return self

    def __pow__(self, r):
        return self.perform_operation(r, '**')

    def __ipow__(self, r):
        self.perform_i_operation(r, '**')
        return self

    def __mod__(self, r):
        return self.perform_operation(r, '%')

    def __eq__(self, r):
        return self.perform_cond_operation(r, '__eq__')

    def __ne__(self, r):
        return self.perform_cond_operation(r, '__ne__')

    def __lt__(self, r):
        return self.perform_cond_operation(r, '__lt__')

    def __gt__(self, r):
        return self.perform_cond_operation(r, '__gt__')

    def __le__(self, r):
        return self.perform_cond_operation(r, '__le__')

    def __ge__(self, r):
        return self.perform_cond_operation(r, '__ge__')

    def __repr__(self):
        insr = osr.SpatialReference(wkt=self.projection)
        projcs = insr.GetAttrValue('projcs')
        datum = insr.GetAttrValue('datum')
        if projcs is None:
            projcs = 'None'
        else:
            projcs = projcs.replace('_', ' ')
        if datum is None:
            datum = 'None'
        else:
            datum = datum.replace('_', ' ')
        methods = {
            'ne': 'numexpr',
            'np': 'numpy'
        }
        if os.path.isfile(self.path):
            prestr = ('A happy raster named %s of house %s\n' %
                      (os.path.basename(self.path), self.format.upper()))
        else:
            prestr = ('An orphaned raster %s of house %s\n' %
                      (os.path.basename(self.path), self.format.upper()))
        return (prestr +
                '    Bands               : %s\n'
                '    Shape               : %s rows, %s columns\n'
                '    Cell Size           : %s (x), %s (y)\n'
                '    Extent              : %s\n'
                '    Data Type           : %s\n'
                '    Uncompressed Size   : %s GB\n'
                '    Projection          : %s\n'
                '    Datum               : %s\n'
                '    Active Band         : %s\n'
                '    Interpolation Method: %s\n'
                '    Using Chunks        : %s\n'
                '    Calculation Method  : %s'
                '' % (self.bandCount, self.shape[0], self.shape[1], self.csx,
                      self.csy, (self.top, self.bottom, self.left, self.right),
                      self.dtype, round(self.size, 3),
                      projcs, datum,
                      self.activeBand, self.interpolationMethod,
                      self.useChunks, methods[self.aMethod]))

    @property
    def gdal_drivers(self):
        known_drivers = {'tif': 'GTiff'}
        return known_drivers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.clean_garbage()

    def __del__(self):
        """Oh well"""
        self.clean_garbage()

    def clean_garbage(self):
        """Remove all temporary files"""
        if hasattr(self, 'garbage'):
            if self.garbage['num'] > 1:
                self.garbage['num'] -= 1
            else:
                try:
                    os.remove(self.garbage['path'])
                except Exception as e:
                    print "Unable to remove temporary file {} because:\n{}".format(
                        self.garbage['path'], e)


class mosaic(raster):
    """Handle a mosaic of rasters"""
    def __init__(self, raster_list):
        self.rasterList = []
        self.extents = []
        self.rasterDtypes = []
        self.cellSizes = []
        projection = None
        for inputRaster in raster_list:
            rast = raster(inputRaster)
            self.rasterList.append(inputRaster)
            self.extents.append((rast.top, rast.bottom, rast.left, rast.right))
            self.cellSizes.append((rast.csx, rast.csy))
            self.rasterDtypes.append(rast.dtype)
            # Take the first occurrence of projection
            if rast.projection is not None and projection is None:
                projection = rast.projection
        self.mergeOrder = numpy.arange(len(self.rasterList))
        self.fullyMerged = False

        # Dimensions of all
        top, bottom, left, right = zip(*self.extents)
        top, bottom, left, right = max(top), min(bottom), min(left), max(right)
        csx, csy = zip(*self.cellSizes)
        csx, csy = min(csx), min(csy)

        # Calculate shape
        shape = (int(numpy.ceil((top - bottom) / csy)),
                 int(numpy.ceil((right - left) / csx)))

        # Collect the most precise data type
        precision = numpy.argmax([numpy.dtype(dtype).itemsize for dtype in self.rasterDtypes])
        dtype = self.rasterDtypes[precision]

        # Build a new raster using the combined specs
        path = generate_name(self.path, 'rastermosaic', 'h5')
        # TODO: Add this path to the garbage of the output
        super(mosaic, self).__init__(path, mode='w', csx=csx, csy=csy, top=top, left=left,
                                     shape=shape, projection=projection, dtype=dtype)
        # Create a mask to track where data are merged
        with h5py.File(self.path, libver='latest') as f:
            f.create_dataset('mosaic', data=numpy.zeros(shape=self.shape, dtype='bool'),
                             compression='lzf')

    def __getitem__(self, s):
        """
        Merge where data are retrieved
        :param s:
        :return:
        """
        # If all rasters are merged, using parent
        if self.fullyMerged:
            super(mosaic, self).__getitem__(s)
        else:
            with h5py.File(self.path, libver='latest') as f:
                ds = f['mosaic']

    @property
    def array(self):
        """
        Merge all rasters and return
        :return:
        """
        pass

    def merge(self, raster_list, s):
        """
        Merge a region
        :param raster_list:
        :return:
        """
        pass

    def interpolation_method(self, method):
        """
        Update interpolation methods for each raster
        :param method: A single method, or a list of methods that matches the number of rasters
        :return: None
        """
        if hasattr(method, '__iter__'):
            if len(method) != len(self.rasterList):
                raise RasterError('Number of methods must match the number of rasters')
            for i, m in enumerate(method):
                self.rasterList[i].interpolationMethod = m
        else:
            for rast in self.rasterList:
                rast.interpolationMethod = method


# Numpy-like methods
def rastround(input_raster, decimal_places):
    """
    Round a raster to a defined number of decimal places
    :param input_raster: raster to be rounded
    :param decimal_places: (int) number of decimal places
    :return: raster instance
    """
    # Open everything
    dp = int(decimal_places)
    r = raster(input_raster)
    out = r.empty()

    # Perform rounding
    if r.useChunks:
        for a, s in r.iterchunks():
            m = a != r.nodata
            a[m] = numpy.round(a[m], dp)
            out[s] = a
    else:
        out[:] = numpy.round(r.array, dp)

    return out


def copy(input_raster):
    """Copy a raster dataset"""
    return raster(input_raster).copy('copy')


def empty(input_raster):
    return raster(input_raster).empty()


def full(input_raster):
    return raster(input_raster).full()


def unique(input_raster):
    """Compute unique values"""
    r = raster(input_raster)
    if r.useChunks:
        uniques = []
        for a, s in r.iterchunks():
            uniques.append(numpy.unique(a[a != r.nodata]))
        return numpy.unique(numpy.concatenate(uniques))
    else:
        return numpy.unique(r.array[r.array != r.nodata])

def rastmin(input_raster):
    """
    Calculate the minimum value
    :param input_raster: A raster-compatible object
    :return: Minimum value
    """
    r = raster(input_raster)
    if r.useChunks:
        stack = []
        for a, s in r.iterchunks():
            try:
                stack.append(numpy.min(a[a != r.nodata]))
            except ValueError:
                pass
        if len(stack) > 0:
            return min(stack)
        else:
            raise ValueError('Cannot collect minimum: No data in raster')
    else:
        try:
            return numpy.min(r.array[r.array != r.nodata])
        except ValueError:
            raise ValueError('Cannot collect minimum: No data in raster')

def rastmax(input_raster):
    """
    Calculate the maximum value
    :param input_raster: A raster-compatible object
    :return: Maximum value
    """
    r = raster(input_raster)
    if r.useChunks:
        stack = []
        for a, s in r.iterchunks():
            try:
                stack.append(numpy.max(a[a != r.nodata]))
            except ValueError:
                pass
        if len(stack) > 0:
            return max(stack)
        else:
            raise ValueError('Cannot collect maximum: No data in raster')
    else:
        try:
            return numpy.max(r.array[r.array != r.nodata])
        except ValueError:
            raise ValueError('Cannot collect maximum: No data in raster')



class vector(object):

    def __init__(self, data=None, mode='r', fields=None, projection=None, geotype=None):
        """
        Vector interfacing class
        :param data: Input data- may be one of:
            1. Path (string)
            2. ogr vector instance
            3. Table
            4. wkt geometry
            5. wkb geometry
            6. A list of any of the above
        """
        self.mode = mode.lower()
        if self.mode not in ['r', 'w', 'r+']:
            raise VectorError('Unsupported file mode {}'.format(mode))
        populate_from_ogr = False
        if isinstance(data, basestring):
            if os.path.isfile(data) and mode != 'w':
                # Open a vector file or a table
                self.driver = self.get_driver_by_path(data)
                if self.driver == 'table':
                    # TODO: Implement table driver
                    self.path = None
                else:
                    # Open dataset
                    driver = ogr.GetDriverByName(self.driver)
                    _data = driver.Open(data)
                    if _data is None:
                        raise VectorError('Unable to open the dataset {} as a vector'.format(data))
                    populate_from_ogr = True
                    self.path = data
            elif mode == 'w':
                # Create an empty data source using an input geometry
                geotypes = ['Polygon', 'LineString', 'Point']
                # Check input geometry type to make sure it works
                if geotype not in ['Polygon', 'LineString', 'Point']:
                    raise VectorError(
                        'Geometry type {} not understood while creating new vector data source.  '
                        'Use one of: {}'.format(geotype, ', '.join(geotypes))
                    )

                # Create some attributes so vector.empty can be called
                self.geometryType = geotype
                self.projection = parse_projection(projection)
                if fields is not None:
                    self.featureCount = len(fields)
                self.fieldWidth, self.fieldPrecision = {}, {}

                outVect = self.empty(spatial_reference=projection, out_path=data)

                self.__dict__.update(outVect.__dict__)
            else:
                raise Exception('Check that {} is a valid file'.format(data))

        elif isinstance(data, ogr.DataSource):
            # Instantiate as ogr instance
            _data = data
            self.path = None
            populate_from_ogr = True

        elif isinstance(data, vector):
            # Update to new vector instance
            self.__dict__.update(data.__dict__)
            # Garbage is only collected if one instance exists
            if hasattr(self, 'garbage'):
                self.garbage['num'] += 1
            # Re-populate from source file
            populate_from_ogr = True
            driver = ogr.GetDriverByName(data.driver)
            _data = driver.Open(data.path)

        elif any([isinstance(data, t) for t in [tuple, list, numpy.ndarray]]):
            # Try for an iterable of wkt's or wkb's
            geos_and_types = [self.geo_from_wellknown(d) for d in data]

            # Ensure the geometry types are all the same
            if not all([geos_and_types[0][1] == g[1] for g in geos_and_types]):
                raise VectorError('Input well-known geometries have multiple geometry types.')

            # Make sure the same number of fields were specified
            if fields is not None and len(fields) != len(geos_and_types):
                raise VectorError('The number of input fields and well-known geometries do not match')

            # Create some attributes so vector.empty can be called
            self.geometryType = geos_and_types[0][1]
            self.path = generate_name(None, '', 'shp')
            self.projection = parse_projection(projection)
            self.featureCount = len(geos_and_types)
            self.fieldWidth, self.fieldPrecision = {}, {}

            if fields is None:
                field_def = []
            else:
                self.fieldTypes = zip(fields.dtype.names, (fields.dtype[i].name for i in range(len(fields.dtype))))
                field_def = self.fieldTypes
            outVect = self.empty(spatial_reference=projection, fields=field_def)
            outVect.garbage['num'] += 1  # Delay garbage by one delete

            # Insert geometries into features
            with outVect.layer() as outlyr:
                # Output layer definition
                outLyrDefn = outlyr.GetLayerDefn()
                # Write fields to output
                if len(field_def) > 0:
                    newFields = self.check_fields(self.fieldTypes)
                    fields = {newField[0]: self.field_to_pyobj(fields[oldField[0]], None, None)
                              for oldField, newField in zip(self.fieldTypes, newFields)}
                    self.fieldTypes = newFields

                # Iterate geometries and populate output with transformed geo's
                for i in range(self.featureCount):
                    # Write geometries as features
                    outFeat = ogr.Feature(outLyrDefn)
                    if len(field_def) > 0:
                        for name, dtype in newFields:
                            outFeat.SetField(name, fields[name][i])
                    geo = ogr.CreateGeometryFromWkb(geos_and_types[i][0])
                    outFeat.SetGeometry(geo)

                    # Add to output layer and clean up
                    outlyr.CreateFeature(outFeat)
                    outFeat.Destroy()

            self.__dict__.update(vector(outVect.path).__dict__)

        else:
            raise VectorError('Cannot read the input data of type {}'.format(type(data).__name__))

        if populate_from_ogr:
            # Collect meta
            try:
                layer = _data.GetLayer()
            except:
                raise VectorError('Could not open the dataset "{}"'.format(data))
            layerDefn = layer.GetLayerDefn()

            # Spatial Reference
            insr = layer.GetSpatialRef()
            if insr is not None:
                self.projection = str(insr.ExportToWkt())
            else:
                self.projection = ''

            # Geometry type
            self.geometryType = str(self.geom_wkb_to_name(layer.GetGeomType()))

            # Feature count
            self.featureCount = layer.GetFeatureCount()

            # Fields
            self.fieldCount = layerDefn.GetFieldCount()
            self.fieldTypes = []
            self.fieldWidth, self.fieldPrecision = {}, {}
            for i in range(self.fieldCount):
                fieldDefn = layerDefn.GetFieldDefn(i)
                name = fieldDefn.GetName()
                dtype = self.ogr_dtype_to_numpy(fieldDefn.GetFieldTypeName(fieldDefn.GetType()),
                                                name, fieldDefn.GetWidth())
                self.fieldTypes.append((name, dtype))
                self.fieldWidth[name] = fieldDefn.GetWidth()
                self.fieldPrecision[name] = fieldDefn.GetPrecision()
            self.fieldNames = [meta[0] for meta in self.fieldTypes]

    @property
    def top(self):
        with self.layer() as inlyr:
            return inlyr.GetExtent()[3]

    @property
    def bottom(self):
        with self.layer() as inlyr:
            return inlyr.GetExtent()[2]

    @property
    def left(self):
        with self.layer() as inlyr:
            return inlyr.GetExtent()[0]

    @property
    def right(self):
        with self.layer() as inlyr:
            return inlyr.GetExtent()[1]

    def geo_from_wellknown(self, wk_):
        """Try and load strings as either a wkb or wkt"""
        try:
            geo = ogr.CreateGeometryFromWkb(wk_)
            if geo is None:
                geo = ogr.CreateGeometryFromWkt(wk_)
                if geo is None:
                    assert False
        except:
            raise VectorError('Unable to load the geometry {}'.format(wk_))
        type = self.geom_wkb_to_name(geo.GetGeometryType())
        return geo.ExportToWkb(), type

    @contextmanager
    def layer(self, i=0):
        """
        Open the source vector file for writing
        :return: ogr vector layer instance
        """
        driver = ogr.GetDriverByName(self.driver)
        if self.mode in ['r+', 'w']:
            writeAccess = True
        else:
            writeAccess = False
        ds = driver.Open(self.path, writeAccess)
        if ds is None:
            raise VectorError('The data source {} can no longer be accessed'.format(self.path))
        layer = ds.GetLayer(i)
        yield layer
        del layer
        ds.Destroy()

    def save(self, path):
        """
        Save the vector instance
        :param path: Output save path
        :return: vector instance where saved
        """
        # TODO: Allow NoneType path and save as in-memory vector
        outDriver = self.get_driver_by_path(path)
        if outDriver == self.driver:
            # Simply copy files
            for f in self.filenames:
                if f.split('.')[-1].lower() != path.split('.')[-1].lower():
                    np = '.'.join(path.split('.')[:-1]) + '.{}'.format(f.split('.')[-1])
                else:
                    np = path
                shutil.copy(f, np)
        else:
            # Save in the prescribed format
            newFields = self.fieldTypes
            # Check that fields are less than 10 chars if shp
            if outDriver == 'ESRI Shapefile':
                newFields = self.check_fields(self.fieldTypes)

            # Generate file and layer
            driver = ogr.GetDriverByName(outDriver)
            ds = driver.CreateDataSource(path)
            outsr = osr.SpatialReference()
            outsr.ImportFromWkt(self.projection)
            outlyr = ds.CreateLayer('bluegeo_vector', outsr, self.geometry_type)

            # Add fields
            for name, dtype in newFields:
                # Create field definition
                fieldDefn = ogr.FieldDefn(name, self.numpy_dtype_to_ogr(dtype))
                fieldDefn.SetWidth(self.fieldWidth[name])
                fieldDefn.SetPrecision(self.fieldPrecision[name])
                # Create field
                outlyr.CreateField(fieldDefn)

            with self.layer() as inlyr:
                # Output layer definition
                outLyrDefn = outlyr.GetLayerDefn()
                fields = {newField[0]: self.field_to_pyobj(self[oldField[0]], self.fieldWidth[oldField[0]],
                                                           self.fieldPrecision[oldField[0]])
                          for oldField, newField in zip(self.fieldTypes, newFields)}

                # Iterate and population geo's
                for i in range(self.featureCount):
                    # Gather and transform geometry
                    inFeat = inlyr.GetFeature(i)
                    geo = inFeat.GetGeometryRef()

                    # Write output fields and geometries
                    outFeat = ogr.Feature(outLyrDefn)
                    for name, dtype in newFields:
                        outFeat.SetField(name, fields[name][i])
                    outFeat.SetGeometry(geo)

                    # Add to output layer and clean up
                    outlyr.CreateFeature(outFeat)
                    outFeat.Destroy()

            ds.Destroy()

    def define_projection(self, spatial_reference):
        """
        Define a projection for the shapefile
        :param spatial_reference: EPSG, wkt, or osr object
        :return: None
        """
        # Don't overwrite if it's open in read-only mode
        if self.projection != '' and self.mode == 'r':
            raise VectorError('Vector source file has a defined projection, and is open in read-only mode')

        # Parse projection argument into wkt
        wkt = parse_projection(spatial_reference)

        # If the source is a .shp create path for output .prj file
        if self.driver == 'ESRI Shapefile':
            prj_path = '.'.join(self.path.split('.')[:-1]) + '.prj'
            with open(prj_path, 'w') as f:
                f.write(wkt)
        else:
            # Need to write it out
            # TODO: this
            raise VectorError('Not implemented yet')

    def empty(self, spatial_reference=None, fields=[], prestr='copy', out_path=None):
        """
        Create a copy of self as an output shp without features
        :return: Fresh vector instance
        """
        add_to_garbage = False
        if out_path is None:
            # Create an output file path
            add_to_garbage = True
            out_path = generate_name(self.path, prestr, 'shp')
        elif out_path.split('.')[-1].lower() != 'shp':
            out_path += '.shp'

        # Check that fields are less than 10 chars
        fields = self.check_fields(fields)

        # Generate output projection
        outsr = osr.SpatialReference()
        if spatial_reference is not None:
            outsr.ImportFromWkt(parse_projection(spatial_reference))
        else:
            outsr.ImportFromWkt(self.projection)

        # Generate file and layer
        driver = ogr.GetDriverByName(self.get_driver_by_path(out_path))
        ds = driver.CreateDataSource(out_path)
        layer = ds.CreateLayer('bluegeo vector', outsr, self.geometry_type)

        # Add fields
        for name, dtype in fields:
            # Create field definition
            fieldDefn = ogr.FieldDefn(name, self.numpy_dtype_to_ogr(dtype))
            try:
                fieldDefn.SetWidth(self.fieldWidth[name])
                fieldDefn.SetPrecision(self.fieldPrecision[name])
            except KeyError:
                pass  # Field is new and does not have a width or precision assigned
            # Create field
            layer.CreateField(fieldDefn)

        # Clean up and return vector instance
        del layer
        ds.Destroy()

        return_vect = vector(out_path, mode='r+')
        if add_to_garbage:
            return_vect.garbage = {'path': out_path, 'num': 1}
        return return_vect

    def add_fields(self, name, dtype, data=None):
        """
        Create a new vector with the output field(s)
        :param name: single or list of names to create
        :param dtype: single or list of data types for each field
        :param data: list of lists to write to each new field
        :return: vector instance
        """
        if isinstance(name, basestring):
            name, dtype = [name], [dtype]  # Need to be iterable (and not strings)

        if self.mode in ['r+', 'w']:
            # Iterate features and insert data
            with self.layer() as inlyr:
                outLyrDefn = inlyr.GetLayerDefn()
                for _name, _dtype in zip(name, dtype):
                    # Modify the current vector with the field
                    fieldDefn = ogr.FieldDefn(_name, self.numpy_dtype_to_ogr(_dtype))
                    # Create field
                    inlyr.CreateField(fieldDefn)
                    self.fieldTypes.append((_name, _dtype))
                    self.fieldPrecision[_name] = fieldDefn.GetPrecision()
                    self.fieldWidth[_name] = fieldDefn.GetWidth()
                    self.fieldCount += 1
                if data is not None:
                    # Broadcast data to the necessary shape
                    shape = 1 if isinstance(name, basestring) else len(name)
                    shape = (shape, self.featureCount)
                    try:
                        data = numpy.broadcast_to(data, shape)
                    except:
                        raise VectorError("Unable to fit the data to the number of fields/features")
                    data = [self.field_to_pyobj(a, 19, 11) for a in data]
                    # Iterate features
                    for _name, _data in zip(name, data):
                        for i in range(self.featureCount):
                            inFeat = inlyr.GetFeature(i)
                            if inFeat is None:
                                inFeat = ogr.Feature(outLyrDefn)
                                # Write output field
                                inFeat.SetField(_name, _data[i])
                                inlyr.SetFeature(inFeat)
                                # Add to output layer and clean up
                                inlyr.CreateFeature(inFeat)
                                inFeat.Destroy()
                            else:
                                # Write output field
                                inFeat.SetField(_name, _data[i])
                                inlyr.SetFeature(inFeat)

        else:
            # Create a new data source
            new_fieldtypes = self.check_fields(self.fieldTypes + zip(name, dtype))

            # Gather data to write into fields
            fields = {field[0]: self.field_to_pyobj(self[field[0]], self.fieldWidth[field[0]],
                                                    self.fieldPrecision[field[0]]) for field in self.fieldTypes}

            fields.update({name_and_dtype[0]: data[i]
                           for i, name_and_dtype in enumerate(zip(name, dtype))})

            # Create empty output
            outVect = self.empty(self.projection, new_fieldtypes, prestr='add_field')

            # Populate output with new field(s)
            with self.layer() as inlyr:
                with outVect.layer() as outlyr:
                    # Output layer definition
                    outLyrDefn = outlyr.GetLayerDefn()
                    # Iterate features
                    for i in range(self.featureCount):
                        inFeat = inlyr.GetFeature(i)
                        if inFeat is not None:
                            geo = inFeat.GetGeometryRef()

                        # Write output fields and geometries
                        outFeat = ogr.Feature(outLyrDefn)
                        for _name, _dtype in new_fieldtypes:
                            outFeat.SetField(_name, fields[_name][i])
                        if inFeat is not None:
                            outFeat.SetGeometry(geo)

                        # Add to output layer and clean up
                        outlyr.CreateFeature(outFeat)
                        outFeat.Destroy()

            return vector(outVect)  # Re-read vector to ensure meta up to date

    def transform(self, sr):
        """
        Project dataset into a new system
        :param sr: Input spatial reference
        :return: vector instance
        """
        # Create ogr coordinate transformation object
        insr = osr.SpatialReference()
        outsr = osr.SpatialReference()
        if self.projection == '':
            raise VectorError('Source vector has an unknown spatial reference, and cannot be transformed')
        insr.ImportFromWkt(self.projection)
        outsr.ImportFromWkt(parse_projection(sr))

        # Check if they're the same, and return a copy if so
        if insr.IsSame(outsr):
            return self

        coordTransform = osr.CoordinateTransformation(insr, outsr)

        # Create empty output
        outVect = self.empty(outsr, self.fieldTypes, prestr='transformed')

        # Transform geometries and populate output
        with self.layer() as inlyr:
            with outVect.layer() as outlyr:
                # Output layer definition
                outLyrDefn = outlyr.GetLayerDefn()
                # Gather field values to write to output
                newFields = self.check_fields(self.fieldTypes)
                fields = {newField[0]: self.field_to_pyobj(self[oldField[0]], self.fieldWidth[oldField[0]],
                                                           self.fieldPrecision[oldField[0]])
                          for oldField, newField in zip(self.fieldTypes, newFields)}

                # Iterate geometries and populate output with transformed geo's
                for i in range(self.featureCount):
                    # Gather and transform geometry
                    inFeat = inlyr.GetFeature(i)
                    geo = inFeat.GetGeometryRef()
                    geo.Transform(coordTransform)

                    # Write output fields and geometries
                    outFeat = ogr.Feature(outLyrDefn)
                    for name, dtype in newFields:
                        outFeat.SetField(name, fields[name][i])
                    outFeat.SetGeometry(geo)

                    # Add to output layer and clean up
                    outlyr.CreateFeature(outFeat)
                    outFeat.Destroy()
        return vector(outVect)  # Re-read vector to ensure meta up to date

    @staticmethod
    def check_fields(fields):
        """
        Ensure fields have names that are less than 10 characters
        :param fields: input field types argument (list of tuples)
        :return: new field list
        """
        outputFields = []
        cursor = []
        for name, dtype in fields:
            name = name[:10]
            i = 0
            while name in cursor:
                i += 1
                name = name[:10-len(str(r))] + str(i)
            cursor.append(name)
            outputFields.append((name, dtype))
        return outputFields

    @property
    def vertices(self):
        """
        Gather x, y, (z) coordinates of all vertices
        :return: 3d array with each node on axis 0
        """
        def get_next(geo, vertices):
            count = geo.GetGeometryCount()
            if count > 0:
                for i in range(count):
                    get_next(geo.GetGeometryRef(i), vertices)
            else:
                vertices += [[geo.GetX(i),geo.GetY(i), geo.GetZ(i)] for i in range(geo.GetPointCount())]

        vertices = []
        for wkb in self[:]:
            try:
                get_next(ogr.CreateGeometryFromWkb(wkb), vertices)
            except:
                pass

        return numpy.array(vertices)

    def __getitem__(self, item):
        """
        __getitem__ functionality of vector class
        :param item: If an index or slice is used, geometry wkb's are returned.
        If a string is used, it will return a numpy array of the field values.
        If an instance of the extent class is used, the output will be a clipped vector instance
        :return: List of wkb's or numpy array of field
        """
        with self.layer() as lyr:
            if any([isinstance(item, obj) for obj in [slice, int, numpy.ndarray, list, tuple]]):
                data = []
                iter = numpy.arange(self.featureCount)[item]
                if not hasattr(iter, '__iter__'):
                    iter = [iter]
                for i in iter:
                    feature = lyr.GetFeature(i)
                    geo = feature.GetGeometryRef()
                    try:
                        data.append(geo.ExportToWkb())
                    except:
                        # Null geometry
                        data.append('')
                    feature.Destroy()

            elif isinstance(item, basestring):
                # Check that the field exists
                try:
                    _, dtype = [ft for ft in self.fieldTypes if ft[0] == str(item)][0]
                except IndexError:
                    raise VectorError('No field named {} in the file {}'.format(item, self.path))

                data = numpy.array([lyr.GetFeature(i).GetField(item) for i in numpy.arange(self.featureCount)],
                                   dtype=dtype)

            # TODO: Finish this- need to figure out how ogr.Layer.Clip works. Maybe through an in-memory ds
            elif isinstance(item, extent):
                # Clip and return a vector instance
                data = self.empty(self.projection, self.fieldTypes, prestr='clip')
                # Perform a clip
                with item.layer as cliplyr:
                    with data.layer as outlyr:
                        if item.geo is not None:
                            # With geometry
                            cliplyr.Clip(cliplyr, outlyr)
                        else:
                            # With extent
                            cliplyr.Clip(extlyr, outlyr)

        return data

    def __setitem__(self, item, val):
        """Set a field or geometry if the vector is open for writing"""
        if self.mode not in ['r+', 'w']:
            raise VectorError('Current vector open as read-only')

        # Check the type of input
        if isinstance(item, basestring):
            # If the instance has no features, make them
            if self.featureCount == 0:
                if hasattr(val, '__iter__'):
                    self.featureCount = len(val)
                else:
                    self.featureCount = 1

            # Need to overwrite field data or create a new field
            try:
                write_data = numpy.broadcast_to(val, self.featureCount).copy()
            except ValueError:
                raise VectorError('Input array cannot be broadcast '
                                  'into the number of features ({})'.format(self.featureCount))
            try:
                dtype = [dt for name, dt in self.fieldTypes if name == item][0]
            except IndexError:
                # New field required
                self.add_fields(item, write_data.dtype.name, write_data)
            else:
                write_data = self.field_to_pyobj(write_data.astype(dtype), self.fieldWidth[item],
                                                 self.fieldPrecision[item])
                # Iterate features and insert data
                with self.layer() as inlyr:
                    # Iterate features
                    for i in range(self.featureCount):
                        inFeat = inlyr.GetFeature(i)
                        # Write output field
                        inFeat.SetField(item, write_data[i])
                        inlyr.SetFeature(inFeat)
            return
        elif isinstance(item, slice):
            start, stop = slice.start, slice.stop
            # Make sure start and stop are not slice member descriptors
            if not isinstance(start, int):
                start = 0
            if not isinstance(stop, int):
                stop = self.featureCount
        elif isinstance(item, int):
            start, stop = item, item + 1

        # Make input iterable
        val = val if hasattr(val, '__iter__') else [val]

        # Create new features if the count is 0
        if self.featureCount == 0:
            with self.layer() as inlyr:
                LyrDefn = inlyr.GetLayerDefn()
                for i in range(stop):
                    Feat = ogr.Feature(LyrDefn)

                    # Add to output layer and clean up
                    inlyr.CreateFeature(Feat)
                    Feat.Destroy()
            self.featureCount = stop

        # Check for bounds
        if start > self.featureCount or stop > self.featureCount:
            raise VectorError('Input feature slice outside of current feature bounds')
        # Check for wkbs
        if not isinstance(val[0], basestring):
            raise VectorError('Item to set must be a wkb geometry')

        # Update feature geometries using the val
        with self.layer() as inlyr:
            for geo_index, i in enumerate(range(start, stop)):
                # Gather and insert geometry
                inFeat = inlyr.GetFeature(i)
                try:
                    geo = ogr.CreateGeometryFromWkb(val[geo_index])
                except:
                    raise VectorError('Unable to load geometry from index {}'.format(geo_index))
                inFeat.SetGeometry(geo)
                # Write feature
                inlyr.SetFeature(inFeat)

    @property
    def filenames(self):
        """
        Return list of files associated with self
        :return: list of file paths
        """
        if self.driver == 'ESRI Shapefile':
            prestr = '.'.join(os.path.basename(self.path).split('.')[:-1])
            d = os.path.dirname(self.path)
            if d == '':
                _d = '.'
            else:
                _d = d
            return [os.path.join(d, f) for f in os.listdir(_d)
                    if '.'.join(f.split('.')[:-1]) == prestr and
                    f.split('.')[-1].lower() in ['shp', 'shx', 'dbf', 'prj']]
        else:
            if os.path.isfile(self.path):
                return [self.path]
            else:
                return []

    @property
    def size(self):
        """
        Get size of file in KB
        :return: float
        """
        files = self.filenames
        if len(files) == 0:
            return 0
        else:
            return sum([os.path.getsize(f) for f in files]) / 1E3

    def __repr__(self):
        insr = osr.SpatialReference(wkt=self.projection)
        projcs = insr.GetAttrValue('projcs')
        datum = insr.GetAttrValue('datum')
        if projcs is None:
            projcs = 'None'
        else:
            projcs = projcs.replace('_', ' ')
        if datum is None:
            datum = 'None'
        else:
            datum = datum.replace('_', ' ')
        size = self.size
        if size > 0:
            prestr = ('A happy vector named %s of house %s\n' %
                      (os.path.basename(self.path), self.driver))
        else:
            prestr = ('An orphaned vector %s of house %s\n' %
                      (os.path.basename(self.path), self.driver))
        return (prestr +
                '    Features   : %s\n'
                '    Fields     : %s\n'
                '    Extent     : %s\n'
                '    Projection : %s\n'
                '    Datum      : %s\n'
                '    File Size  : %s KB\n' %
                (self.featureCount, self.fieldCount, (self.top, self.bottom, self.left, self.right),
                 projcs, datum, size))

    @staticmethod
    def guess_nodata(dtype):
        """"""
        _dtype = dtype.lower()
        if 'float' in _dtype:
            return float(numpy.nan)
        elif 'int' in _dtype:
            return 0
        elif 's' in _dtype:
            return " "
        elif 'datetime' in _dtype:
            return 0
        else:
            raise VectorError('Cannot determine default no data values with the data type {}'.format(dtype))

    @staticmethod
    def field_to_pyobj(a, width, prec):
        """
        Convert numpy array to a python object to be used by ogr
        :param a: input numpy array
        :return: list
        """
        _types = {'uint8': int, 'int8': int,
                  'uint16': int, 'int16': int,
                  'uint32': int, 'int32': int,
                  'uint64': int, 'int64': int,
                  'float32': float, 'float64': float,
                  's': str,
                  'object': str
        }

        dtype = a.dtype.name
        if dtype[0].lower() == 's' or width is None or prec is None:
            dtype = 's'
            fmt = '{}'
        elif 'float' in dtype.lower():
            fmt = '{' + ':{}.{}f'.format(width, prec) + '}'
        else:
            fmt = '{:d}'

        # return [None if obj == 'None' else obj for obj in map(_types[dtype], a)]
        return map(fmt.format, a)

    @staticmethod
    def drivers():
        """
        Dict of drivers
        :return: dict of drivers
        """
        return {'shp': 'ESRI Shapefile',
                'kml': 'KML',
                'kmz': 'KML',
                'geojson': 'GeoJSON',
                'csv': 'table',
                'xls': 'table',
                'xlsx': 'table'
                }

    @staticmethod
    def get_driver_by_path(path):
        """
        Return a supported ogr (or internal) driver using a file extension
        :param path: File path
        :return: driver name
        """
        ext = path.split('.')[-1].lower()
        if len(ext) == 0:
            raise VectorError('File path does not have an extension')

        try:
            return vector.drivers()[ext]
        except KeyError:
            raise VectorError('Unsupported file format: "{}"'.format(ext))

    @staticmethod
    def numpy_dtype_to_ogr(dtype):
        """
        Convert a numpy data type string to ogr data type object
        :param dtype: numpy data type
        :return: ogr field data type
        """
        _types = {'float64': ogr.OFTReal,
                  'float32': ogr.OFTReal,
                  'int32': ogr.OFTInteger,
                  'uint32': ogr.OFTInteger,
                  'int64': ogr.OFTInteger64,
                  'uint64': ogr.OFTInteger64,
                  'int8': ogr.OFTInteger,
                  'uint8': ogr.OFTInteger,
                  'bool': ogr.OFTInteger,
                  'DateTime64': ogr.OFTDate,
                  's': ogr.OFTString
                  }

        if dtype[0].lower() == 's':
            dtype = 's'

        return _types[dtype]

    @staticmethod
    def ogr_dtype_to_numpy(field_type, field_name, width=None):
        """
        Convert an ogr field type name to a numpy dtype equivalent
        :param field_type: type string
        :return: numpy datatype string
        """
        fieldTypes = {'Real': 'float32',
                      'Integer': 'int32',
                      'Integer64': 'int64',
                      'String': 'S{}'.format(width),
                      'Date': 'datetime64'}

        try:
            return fieldTypes[field_type]
        except KeyError:
            raise VectorError('Field {} has an unrecognized data type "{}"'.format(field_name, field_type))

    @property
    def geometry_type(self):
        """
        ogr geometry types from string representations
        :return: dict of ogr geometry objects
        """
        _types  = {'Unknown': ogr.wkbUnknown,
                   'Point': ogr.wkbPoint,
                   'LineString': ogr.wkbLineString,
                   'Polygon': ogr.wkbPolygon,
                   'MultiPoint': ogr.wkbMultiPoint,
                   'MultiLineString': ogr.wkbMultiLineString,
                   'MultiPolygon': ogr.wkbMultiPolygon,
                   'GeometryCollection': ogr.wkbGeometryCollection,
                   'None': ogr.wkbNone,
                   'LinearRing': ogr.wkbLinearRing,
                   'PointZ': ogr.wkbPointZM,
                   'Point25D': ogr.wkb25DBit,
                   'LineString25D': ogr.wkbLineString25D,
                   'Polygon25D': ogr.wkbPolygon25D,
                   'MultiPoint25D': ogr.wkbMultiPoint25D,
                   'MultiLineString25D': ogr.wkbMultiLineString25D,
                   'MultiPolygon25D': ogr.wkbMultiPolygon25D,
                   'GeometryCollection25D': ogr.wkbGeometryCollection25D
                   }

        return _types[self.geometryType]

    @staticmethod
    def geom_wkb_to_name(code):
        """
        Return the name representation of a OGRwkbGeometryType
        :param code: OGRwkbGeometryType
        :return: String name of geometry
        """
        # Collected from
        # https://kite.com/docs/python/django.contrib.gis.admin.options.OGRGeomType
        wkb25bit = -2147483648
        _types = {0: 'Unknown',
                  1: 'Point',
                  2: 'LineString',
                  3: 'Polygon',
                  4: 'MultiPoint',
                  5: 'MultiLineString',
                  6: 'MultiPolygon',
                  7: 'GeometryCollection',
                  100: 'None',
                  101: 'LinearRing',
                  1 + wkb25bit: 'Point25D',
                  2 + wkb25bit: 'LineString25D',
                  3 + wkb25bit: 'Polygon25D',
                  4 + wkb25bit: 'MultiPoint25D',
                  5 + wkb25bit: 'MultiLineString25D',
                  6 + wkb25bit: 'MultiPolygon25D',
                  7 + wkb25bit: 'GeometryCollection25D',
                  }

        try:
            return _types[code]
        except KeyError:
            raise VectorError('Unrecognized Geometry type OGRwkbGeometryType: {}'.format(code))


    def rasterize(self, template_raster, attribute_field=None):
        """
        Create a raster from the current instance
        :param template_raster: raster to use specs from
        :param attribute_field: attribute field to use for raster values (returns a mask if None)
        :return: raster instance
        """
        # Grab the raster specs
        r = raster(template_raster)
        top, left, nrows, ncols, csx, csy = r.top, r.left, r.shape[0], r.shape[1], r.csx, r.csy

        # Transform self if necessary
        vector = self.transform(r.projection)

        # Grab the data type from the input field
        if attribute_field is not None:
            try:
                dtype = [field[1] for field in vector.fieldTypes if field[0] == attribute_field][0]
            except IndexError:
                raise VectorError('Cannot find the field {} during rasterize'.format(attribute_field))
            # If dtype is a string, try to cast the field into a float, else error
            write_data = vector[attribute_field]
            if 's' in dtype.lower():
                try:
                    write_data = numpy.float32(write_data)
                except:
                    raise ValueError('Cannot cast the field {} into a numeric type'.format(attribute_field))
            nodata = numpy.array(r.nodata).astype(dtype)
        else:
            nodata = 0
            dtype = 'bool'
            write_data = numpy.ones(shape=vector.featureCount)

        # Allocate output array and raster for writing raster values
        outarray = numpy.full(r.shape, nodata, dtype)

        outrast = r.astype(dtype)
        outrast.nodataValues = [nodata]

        def get_next(geo, vertices, hole, geom_type):
            """Recursive function to return lists of vertices in geometries"""
            count = geo.GetGeometryCount()
            if count > 0:
                for i in range(count):
                    _geo = geo.GetGeometryRef(i)
                    next_is_zero = _geo.GetGeometryCount() == 0
                    if i > 0 and 'Polygon' in geom_type and next_is_zero:
                        hole = True
                    get_next(_geo, vertices, hole, geom_type)
            else:
                vertices.append(([(geo.GetX(i), geo.GetY(i)) for i in range(geo.GetPointCount())],
                                 hole))  # Tuple in the form ([(x1, y1),...(xn, yn)], hole or not)

        def _rasterize(vertices, geom_type):
            """Rasterize a list of vertices within the containing envelope"""
            pixels = coords_to_indices(zip(*vertices), top, left, csx, csy, r.shape)

            if pixels[0].size == 0:
                # Nothing overlaps the output window
                return numpy.zeros(shape=(1, 1), dtype='bool'), 0, 0

            # Remove duplicates in sequence if the geometry is not a point
            if 'Point' not in geom_type:
                duplicates = numpy.zeros(shape=pixels[0].shape, dtype='bool')
                duplicates[1:] = (pixels[0][1:] == pixels[0][:-1]) & (pixels[1][1:] == pixels[1][:-1])
                row_inds, col_inds = pixels[0][~duplicates], pixels[1][~duplicates]
            else:
                row_inds, col_inds = pixels[0], pixels[1]
            del pixels

            # Local window shape and pixels
            i_insert = row_inds.min()
            j_insert = col_inds.min()
            nrows = row_inds.max() - i_insert + 1
            ncols = col_inds.max() - j_insert + 1
            row_inds -= i_insert
            col_inds -= j_insert

            if 'Point' in geom_type or len(row_inds) == 1:
                # Simply apply the points to the output
                out_array = numpy.zeros(shape=(nrows, ncols), dtype='bool')
                out_array[(row_inds, col_inds)] = 1
                return out_array, i_insert, j_insert
            window = Image.new("1", (ncols, nrows), 0)  # Create 1-bit image
            window_image = ImageDraw.Draw(window)  # Create imagedraw instance
            if 'Polygon' in geom_type or 'LinearRing' in geom_type:
                # Create a polygon mask (1) with the outline filled
                window_image.polygon(zip(col_inds, row_inds), outline=1, fill=1)
            elif 'Line' in geom_type:
                # Draw a line mask (1)
                window_image.line(zip(col_inds, row_inds), 1)

            # Return output image as array (which needs to be reshaped) and the insertion location
            return numpy.array(window, dtype='bool').reshape(nrows, ncols), i_insert, j_insert

        # Need to track where data have been inserted to ensure holes don't overwrite data from other features
        data_track = numpy.zeros(shape=outarray.shape, dtype='bool')
        for feature_index, wkb in enumerate(vector[:]):
            vertices = []
            geo = ogr.CreateGeometryFromWkb(wkb)
            get_next(geo, vertices, False, self.geom_wkb_to_name(geo.GetGeometryType()))

            # Track slices from this feature
            mask_update = []

            for points_and_hole in vertices:
                point_set, is_hole = points_and_hole
                window, i_insert, j_insert = _rasterize(point_set, vector.geometryType)
                i_end = i_insert + window.shape[0]
                j_end = j_insert + window.shape[1]

                window = window & ~(data_track[i_insert:i_end, j_insert:j_end])

                # Use window as mask to insert data into output
                if is_hole:
                    write_value = nodata
                else:
                    write_value = write_data[feature_index]
                outarray[i_insert:i_end, j_insert:j_end][window] = write_value

                mask_update.append((i_insert, i_end, j_insert, j_end))

            # Update data track
            for i_, _i, j_, _j in mask_update:
                data_track[i_:_i, j_:_j][outarray[i_:_i, j_:_j] != nodata] = 1

        outrast[:] = outarray

        return outrast

    def __del__(self):
        """Oh Well"""
        self.clean_garbage()

    def clean_garbage(self):
        """Remove all temporary files"""
        extensions = ['shp', 'shx', 'dbf', 'prj', 'xml', 'sbn', 'sbx', 'cpg']
        if hasattr(self, 'garbage'):
            if self.garbage['num'] > 1:
                self.garbage['num'] -= 1
            else:
                for ext in extensions:
                    p = self.garbage['path']
                    p = os.path.join(os.path.dirname(p),
                                     '.'.join(os.path.basename(p).split('.')[:-1]) + '.{}'.format(ext))
                    if os.path.isfile(p):
                        try:
                            os.remove(p)
                        except:
                            pass


def force_gdal(input_raster):
    r = raster(input_raster)
    if r.format == 'HDF5':
        path = generate_name(r.path, 'copy', 'tif')
        r.save(path)
        tmp = True
    else:
        path = r.path
        tmp = False
    return path, tmp


def assert_type(data):
    """
    Check and return the type of data
    :param data: data to parse
    :return: object to instantiate (raster, vector, or extent)
    """
    def raise_type_error():
        raise TypeError('Unable to provide a means to open {}'.format(data))

    if isinstance(data, vector):
        return vector
    if isinstance(data, raster):
        return raster

    if any([isinstance(data, t) for t in [list, tuple, numpy.ndarray]]) and len(data) == 4:
        return extent

    if isinstance(data, basestring):
        # Check if gdal raster
        ds = gdal.Open(data)
        if ds is not None:
            return raster

        # Check if a bluegeo raster
        if data.split('.')[-1].lower() == 'h5':
            with h5py.File(data, libver='latest') as ds:
                # Just check for the format attribute
                if 'format' in dict(ds.attrs).keys():
                    return raster

        # Check if a vector
        try:
            driver = vector.get_driver_by_path(data)
        except:
            raise_type_error()
        if driver == 'table':
            return vector
        else:
            # Try to open dataset
            driver = ogr.GetDriverByName(driver)
            _data = driver.Open(data)
            if _data is None:
                raise_type_error()
            else:
                return vector
