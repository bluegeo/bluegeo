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
from rtree import index
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
    print("Warning: Shapely is not installed and some operations will not be possible.")

from .util import (parse_projection, generate_name, coords_to_indices, indices_to_coords, transform_points, isclose,
                   compare_projections)

gdal.PushErrorHandler('CPLQuietErrorHandler')

"""Custom Exceptions"""


class RasterError(Exception):
    pass


class VectorError(Exception):
    pass


class ExtentError(Exception):
    pass


class Extent(object):

    def __init__(self, data):
        """
        Extent to be used to control geometries
        :param data: iterable of (top, bottom, left, right) or Vector class instance, or instance of Raster class
        """
        if any(isinstance(data, o) for o in [tuple, list, numpy.ndarray]):
            self.bounds = data
            self.geo = None
        elif isinstance(data, Vector):
            self.bounds = (data.top, data.bottom, data.left, data.right)
            self.geo = data
        elif isinstance(data, Raster):
            self.bounds = (data.top, data.bottom, data.left, data.right)
            self.geo = data
        elif isinstance(data, Extent):
            self.__dict__.update(data.__dict__)
        else:
            raise ExtentError('Unsupported Extent argument of type {}'.format(type(data).__name__))

        try:
            if self.bounds[0] <= self.bounds[1] or self.bounds[2] >= self.bounds[3]:
                assert False
        except:
            raise ExtentError("Invalid or null Extent")

    def within(self, other):
        """
        Check if this Extent is within another
        :param other: other data that can be instantiated using the Extent class
        :return: boolean
        """
        try:
            other = Extent(other).transform(self.geo.projection)
        except AttributeError:
            other = Extent(other)
        top, bottom, left, right = self.bounds
        _top, _bottom, _left, _right = Extent(other).bounds
        if all([top <= _top, bottom >= _bottom, left >= _left, right <= _right]):
            return True
        else:
            return False

    def intersects(self, other):
        """
        Check if this Extent intersects another
        :param other: other data that can be instantiated using the Extent class
        :return: boolean
        """
        try:
            other = Extent(other).transform(self.geo.projection)
        except AttributeError:
            other = Extent(other)
        top, bottom, left, right = self.bounds
        _top, _bottom, _left, _right = Extent(other).bounds
        if any([top <= _bottom, bottom >= _top, left >= _right, right <= _left]):
            return False
        else:
            return True

    def contains(self, other):
        return Extent(other).within(self)

    def transform(self, projection, precision=1E-09):
        """
        Transform the current Extent to the defined projection.
        Note, the geo attribute will be disconnected for safety.
        :param projection: input projection argument
        :return: new Extent instance
        """
        if self.geo is None or self.geo.projection == '':
            # Cannot transform, as the coordinate system is unknown
            return Extent(self.bounds)

        # Gather the spatial references
        wkt = parse_projection(projection)
        insr = osr.SpatialReference()
        insr.ImportFromWkt(self.geo.projection)
        outsr = osr.SpatialReference()
        outsr.ImportFromWkt(wkt)
        if insr.IsSame(outsr):
            # Nothing needs to be done
            return Extent(self.bounds)

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

        return Extent((top, bottom, left, right))

    @property
    def corners(self):
        """Get corners of Extent as coordinates [(x1, y1), ...(xn, yn)]"""
        top, bottom, left, right = self.bounds
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    def __eq__(self, other):
        check = Extent(other)
        if not self.geo is None:
            check = check.transform(self.geo.projection)
        return numpy.all(numpy.isclose(self.bounds, check.bounds))


class Raster(object):
    """
    Main Raster data interfacing class

    Data can be:
        1.  Path to a GDAL supported Raster
        2.  Similarly, a gdal Raster instance
        3.  Path to an HDF5 dataset
        4.  An h5py dataset instance
        5.  Another instance of this Raster class,
            which creates a copy of the input.

    Data are first read virtually, whereby all
    attributes may be accessed, but no underlying
    grid data are loaded into memory.

    Data from the Raster may be loaded into memory
    using standard __getitem__ syntax, by accessing
    the .array attribute (property), or by iterating
    all chunks (i.e. blocks, or tiles).

    Modes for modifying Raster datasets are 'r',
    'r+', and 'w'.  Using 'w' will create an empty
    Raster, 'r+' will allow direct modification of the
    input dataset, and 'r' will automatically
    create a copy of the Raster for writing if
    a function that requires modification is called.
    """

    def __init__(self, input_data, mode='r', **kwargs):
        # Record mode
        if mode in ['r', 'r+', 'w']:
            self.mode = mode
        else:
            raise RasterError('Unrecognized file mode "%s"' % mode)

        # Check if input_data is a string
        if isinstance(input_data, str):
            # If in 'w' mode, write a new file
            if self.mode == 'w':
                self.build_new_raster(input_data, **kwargs)
            # Check if input_data is a valid file
            elif not os.path.isfile(input_data) and not os.path.isdir(input_data):
                raise RasterError('%s is not a Raster file' % input_data)
            else:
                # Try an HDF5 input_data source
                try:
                    with h5py.File(input_data, libver='latest', mode='r') as ds:
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
        # ...or a Raster instance
        elif isinstance(input_data, Raster):
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
                              "Raster instance\n"
                              "kwargs to build new Raster".format(type(input_data).__name__))

        # Populate other attributes
        self.activeBand = 1
        # Set default interpolation- this should be changed manually to
        #   configure all interpolations of this Raster.  Use one of:
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
        '''Load attributes from a gdal Raster'''
        self.projection = ds.GetProjectionRef()
        gt = ds.GetGeoTransform()
        self.left = float(gt[0])
        self.csx = float(gt[1])
        self.top = float(gt[3])
        self.csy = float(abs(gt[5]))
        self.shape = (ds.RasterYSize, ds.RasterXSize)
        self.bottom = self.top - (self.csy * self.shape[0])
        self.right = self.left + (self.csx * self.shape[1])
        self.bandCount = int(ds.RasterCount)
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
                              for key, val in dict(ds.attrs).items()})
        self.nodataValues = [getattr(numpy, self.dtype)(d) for d in self.nodataValues]
        self.shape = tuple(self.shape)
        self.path = str(ds.filename)

    def new_hdf5_raster(self, output_path, compress):
        """
        Create a new HDF5 data source for Raster.  Factory for 'save' and 'empty'.
        :param output_path: path for Raster
        :return: None
        """
        # Do not overwrite self
        if os.path.normpath(output_path).lower() == os.path.normpath(self.path).lower() and self.mode == 'r':
            raise RasterError('Cannot overwrite the source dataset because it is open as read-only')

        # Create new file
        with h5py.File(output_path, mode='w', libver='latest') as newfile:
            # Copy data from data source to new file
            prvb = self.activeBand
            # File may not exist yet if being built from new Raster
            if hasattr(self, 'path'):
                chunks = self.chunks
            else:
                chunks = (256, 256)
            # Add attributes to new file
            newfile.attrs.update({key: ('None' if not isinstance(val, numpy.ndarray) and val == None else val)
                                  for key, val in self.__dict__.items()
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
            out_rast = Raster(output_path, mode='r+')
            for _ in self.bands:
                out_rast.activeBand = self.activeBand
                if self.useChunks:
                    for a, s in self.iterchunks():
                        out_rast[s] = a
                else:
                    out_rast[:] = self.array

        # If GDAL:
        elif extension in list(self.gdal_drivers.keys()):
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
        Build a new Raster from a set of keyword args
        """
        # Force to HDF5 TODO: Provide support for GDAL types
        if path.split('.')[-1].lower() != 'h5':
            path += '.h5'
        # Get kwargs- for building the new Raster
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
                              ' when building a new Raster.')
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
        or self is instantiated using another Raster instance.
        Defaults to HDF5 format.
        """
        path = generate_name(self.path, file_suffix, format)
        # Write new file and return Raster
        self.save(path)
        new_rast = Raster(path)
        # new_rast is temporary, so prepare garbage
        new_rast.garbage = {'path': path, 'num': 1}
        new_rast.mode = 'r+'
        return new_rast

    def empty(self, path=None, compress=True):
        """
        Return an empty copy of the Raster- fast for empty Raster instantiation
        :param path: output path if desired
        :param compress: Compress the output
        :return: Raster instance
        """
        if path is None:
            out_path = generate_name(self.path, 'copy', 'h5')
        else:
            out_path = path
        #  Create new HDF5 file
        self.new_hdf5_raster(out_path, compress=compress)

        # Add to garbage if temporary, and make writeable
        outrast = Raster(out_path)
        if path is None:
            outrast.garbage = {'path': out_path, 'num': 1}
        outrast.mode = 'r+'
        return outrast

    def full(self, data, path=None, compress=True):
        """
        Return a copy of the Raster filled with the input data
        :param data: Array or scalar to be used to fill the Raster
        :param path: output path if not temporary
        :param compress: Compress the output or not
        :return: Raster instance
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
                raise RasterError('Oops...the Raster %s is now missing.' %
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
        """Return arrays of the coordinates of all Raster grid cells"""
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
                chunks = list(map(int, custom_chunks))
                assert len(custom_chunks) == 2
            except:
                raise RasterError('Custom chunks must be a tuple or list of'
                                  ' length 2 containing integers')

        # Parse expand arg (for chunk overlap)
        if type(expand) in [int, float, numpy.ndarray]:
            i = j = int(expand)
        elif type(expand) == tuple or type(expand) == list:
            i, j = list(map(int, expand))
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
                chunks_i = int(chunks[0] * ((resid / x_chunks) + 1))
            else:
                chunks_j = int(chunks[1] * cache_chunks)
                chunks_i = chunks[0]
        else:
            chunks_i, chunks_j = chunks

        ychunks = list(range(0, self.shape[0], chunks_i)) + [self.shape[0]]
        xchunks = list(range(0, self.shape[1], chunks_j)) + [self.shape[1]]
        ychunks = list(zip(numpy.array(ychunks[:-1]) - ifr,
                           numpy.array(ychunks[1:]) + ito))
        ychunks[0] = (0, ychunks[0][1])
        ychunks[-1] = (ychunks[-1][0], self.shape[0])
        xchunks = list(zip(numpy.array(xchunks[:-1]) - jfr,
                           numpy.array(xchunks[1:]) + jto))
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
                              ' trying to save a GDAL Raster' % dtype)

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

        # Return casted Raster
        return out

    def define_projection(self, projection):
        '''Define the Raster projection'''
        if (self.projection != '' and
                self.projection is not None) and self.mode == 'r':
            raise RasterError('The current Raster already has a spatial'
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
        Align Extent and cells with another Raster
        """
        inrast = Raster(input_raster)
        inrast_bbox = Extent(inrast).bounds

        samesrs = compare_projections(self.projection, inrast.projection)

        # Check if cells align
        if all([isclose(self.csx, [inrast.csx], tolerance),
                isclose(self.csy, [inrast.csy], tolerance),
                isclose((self.top - inrast.top) % self.csy, [0, self.csy], tolerance),
                isclose((self.left - inrast.left) % self.csx, [0, self.csx], tolerance),
                samesrs]):
            # Simple slicing is sufficient
            return self.clip(inrast_bbox)
        else:
            # Transform required
            print("Transforming to match rasters...")
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
        top, bottom, left, right = list(map(float, bbox))
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

    def aligns(self, other_raster, tolerance=1E-06):
        """
        Check if the raster instance is aligned spatially with another raster
        :param other_raster: Raster to test
        :returns bool:
        """
        inrast = Raster(other_raster)

        # Check spatial references

        return all([isclose(self.csx, [inrast.csx], tolerance),
                    isclose(self.csy, [inrast.csy], tolerance),
                    isclose(self.top, [inrast.top], tolerance),
                    isclose(self.left, [inrast.left], tolerance),
                    self.shape == inrast.shape,
                    compare_projections(self.projection, inrast.projection)])

    def clip(self, bbox_or_dataset):
        """
        Slice self using bounding box coordinates or, using Vector or Raster data coverage.

        Note: the bounding box may not be honoured exactly, as the Extent needs to be snapped to the cell size.
        To yield an extact bounding box Extent use transform.
        :param bbox_or_dataset: Raster, Vector, Extent, or bbox (top, bottom, left, right)
        """
        # Get input
        clipper = assert_type(bbox_or_dataset)(bbox_or_dataset)
        bbox = Extent(clipper).transform(self.projection).bounds

        # Check that bbox is not inverted
        if any([bbox[0] <= bbox[1], bbox[2] >= bbox[3]]):
            raise RasterError('Input bounding box appears to be inverted'
                              ' or has null dimensions')

        # Get slices
        self_slice, insert_slice, shape, bbox = self.slice_from_bbox(bbox)

        # Check if no change
        if all([self.top == bbox[0], self.bottom == bbox[1],
                self.left == bbox[2], self.right == bbox[3],
                isinstance(clipper, Extent)]):
            return self.copy('clip')

        # Create output dataset with new Extent
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
        outds = Raster(path, mode='w', **kwargs)
        # If a Vector or Raster is specified, use it to create a mask
        if isinstance(clipper, Vector):
            spatial_mask = clipper.transform(self.projection).rasterize(outds.path).array
        elif isinstance(clipper, Raster):
            spatial_mask = clipper.match_raster(self).mask.array
        else:
            spatial_mask = None

        # Add output to garbage
        outds.garbage = {'path': path, 'num': 1}
        for _ in outds.bands:
            insert_array = self[self_slice]
            if spatial_mask is not None:
                insert_array[~spatial_mask] = outds.nodata
            outds[insert_slice] = insert_array
        return outds

    def clip_to_data(self):
        """
        Change Raster Extent to include the minimum bounds where data exist
        :return: Raster instance
        """
        i, j = numpy.where(self.array != self.nodata)
        y, x = indices_to_coords(([i.min(), i.max()], [j.min(), j.max()]),
                                 self.top, self.left, self.csx, self.csy)
        return self.clip((y[0] + self.csy / 2, y[1] - self.csy / 2,
                          x[0] - self.csx / 2, x[1] + self.csx / 2))

    def transform(self, **kwargs):
        """
        Change cell size, projection, or Extent.
        ------------------------
        In Args

        "projection": output projection argument
            (wkt, epsg, osr.SpatialReference, Raster instance)

        "csx": output cell size in the x-direction

        "cxy": output cell size in the y-direction

        "Extent": (Extent instance) bounding box for output
            Note- if Extent does not have a defined coordinate system,
            it is assumed to be in the output spatial reference

        "template": other Raster instance, overrides all other arguments and projects into the Raster
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
            output_extent = kwargs.get('Extent', None)
            if output_extent is None:
                bbox = Extent(self)
            else:
                bbox = Extent(output_extent)

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
            if all([i is None for i in [insrs, outsrs, csx, csy]] + [bbox == Extent(self)]):
                print("Warning: No transformation operation was necessary")
                return self.copy('transform')

            # Recalculate the Extent and calculate potential new cell sizes if a coordinate system change is necessary
            if insrs is not None:
                # Get the corners and transform the Extent
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

            # Snap the potential new cell sizes to the Extent
            ncsx = (right - left) / int(round((right - left) / ncsx))
            ncsy = (top - bottom) / int(round((top - bottom) / ncsy))

            # One of Extent or cell sizes must be updated to match depending on args
            if output_extent is not None:
                # Check that cell sizes are compatible if they are inputs
                if csx is not None:
                    xresid = round((bbox.bounds[3] - bbox.bounds[2]) % csx, 5)
                    if xresid != round(csx, 5) and xresid != 0:
                        raise RasterError('Transform cannot be completed due to an'
                                          ' incompatible Extent %s and cell size (%s) in'
                                          ' the x-direction' % ((bbox.bounds[3], bbox.bounds[2]), csx))
                else:
                    # Use ncsx
                    csx = ncsx
                if csy is not None:
                    yresid = round((bbox.bounds[0] - bbox.bounds[1]) % csy, 5)
                    if yresid != round(csy, 5) and yresid != 0:
                        raise RasterError('Transform cannot be completed due to an'
                                          ' incompatible Extent %s and cell size (%s) in'
                                          ' the y-direction' % ((bbox.bounds[0], bbox.bounds[1]), csy))
                else:
                    # Use ncsy
                    csy = ncsy
            else:
                # Use the cell size to modify the output Extent
                if csx is None:
                    csx = ncsx
                if csy is None:
                    csy = ncsy

                # Compute the shape using the existing Extent and input cell sizes
                shape = (int(round((top - bottom) / csy)), int(round(right - left) / csx))

                # Expand Extent to fit cell size
                resid = (right - left) - (csx * shape[1])
                if resid < 0:
                    resid /= -2.
                elif resid > 0:
                    resid = resid / (2. * csx)
                    resid = (numpy.ceil(resid) - resid) * csx
                left -= resid
                right += resid

                # Expand Extent to fit cell size
                resid = (top - bottom) - (csy * shape[0])
                if resid < 0:
                    resid /= -2.
                elif resid > 0:
                    resid = resid / (2. * csy)
                    resid = (numpy.ceil(resid) - resid) * csy
                bottom -= resid
                top += resid
                bbox = Extent((top, bottom, left, right))

            # Compute new shape
            shape = (int(round((bbox.bounds[0] - bbox.bounds[1]) / csy)),
                     int(round((bbox.bounds[3] - bbox.bounds[2]) / csx)))

            # Create output Raster dataset
            if insrs is not None:
                insrs = insrs.ExportToWkt()
            if outsrs is not None:
                outsrs = outsrs.ExportToWkt()
                output_srs = outsrs
            else:
                output_srs = self.projection

        else:
            t = Raster(template)
            insrs = self.projection
            outsrs = t.projection
            output_srs = t.projection
            shape, bbox, csx, csy = t.shape, Extent(t), t.csx, t.csy

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

        # Direct to input Raster dataset
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

        # Return new Raster
        outrast = Raster(path)
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
        """Generate a new gdal Raster dataset"""
        extension = output_path.split('.')[-1].lower()
        if extension not in list(self.gdal_drivers.keys()):
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
            bigtiff = 'BIGTIFF=YES'
        else:
            bigtiff = 'BIGTIFF=NO'
        parszOptions = [tiled, blockysize, blockxsize, comp, bigtiff]
        ds = driver.Create(output_path, int(shape[1]), int(shape[0]),
                           int(bands), Raster.get_gdal_dtype(dtype),
                           parszOptions)
        if ds is None:
            raise RasterError('GDAL error trying to create new Raster.')
        ds.SetGeoTransform((left, float(csx), 0, top, 0, csy * -1.))
        projection = parse_projection(projection)
        ds.SetProjection(projection)
        ds = None
        outraster = Raster(output_path, mode='r+')
        return outraster

    def polygonize(self):
        """
        Factory for Raster.vectorize to wrap the gdal.Polygonize method
        :return: Vector instance
        """
        # Create a .tif if necessary
        raster_path, garbage = force_gdal(self)
        input_raster = Raster(raster_path)

        # Allocate an output Vector
        vector_path = generate_name(self.path, 'polygonize', 'shp')
        outvect = Vector(vector_path, mode='w', geotype='Polygon', projection=self.projection)
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

        outvect.__dict__.update(Vector(vector_path).__dict__)
        outvect.garbage = {'path': vector_path, 'num': 1}
        return outvect

    def vectorize(self, geotype='Polygon', **kwargs):
        """
        Create a polygon, line, or point Vector from the Raster
        :param geotype: The geometry type of the output Vector.  Choose from 'Polygon', 'LineString', and 'Point'
        :param kwargs:
            centroid=False Use the centroid of the Raster regions when creating points
        :return: Vector instance
        """
        if geotype == 'Polygon':
            # Apply gdal.Polygonize method
            return self.polygonize()

        elif geotype == 'LineString':
            raise NotImplementedError('This is still in development')

        elif geotype == 'Point':
            a = self.array

            # If centroid is specified, label the Raster and grab the respective points
            if kwargs.get('centroid', False):
                labels, number = sklabel(a, return_num=True, background=self.nodata)
                properties = regionprops(labels)
                values = [a[tuple(properties[i].coords[0])] for i in range(number)]
                coords = numpy.array([properties[i].centroid for i in range(number)])
                coords[:, 0] = (self.top - (self.csy / 2)) - (coords[:, 0] * self.csy)
                coords[:, 1] = (self.left + (self.csx / 2)) + (coords[:, 1] * self.csx)
                return Vector([shpwkb.dumps(pnt) for pnt in geometry.MultiPoint(numpy.fliplr(coords))],
                              fields=numpy.array(values, dtype=[('raster_val', self.dtype)]),
                              projection=self.projection)

            # Grab the coordinates and make a Vector using shapely wkb string dumps
            m = a != self.nodata
            y, x = indices_to_coords(numpy.where(m), self.top, self.left, self.csx, self.csy)

            return Vector([shpwkb.dumps(pnt) for pnt in geometry.MultiPoint(list(zip(x, y))).geoms],
                          fields=numpy.array(a[m], dtype=[('raster_val', self.dtype)]),
                          projection=self.projection)

    def extent_to_vector(self, as_mask=False):
        """
        Write the current Raster Extent to a shapefile
        :param as_mask: Return a Vector of values with data in the Raster.  Otherwise the image bounds are used.
        :return: Vector instance
        """
        if as_mask:
            # Polygonize mask
            return self.mask.vectorize('Polygon')
        else:
            # Create a wkb from the boundary of the Raster
            geo = geometry.Polygon(Extent(self).corners)
            _wkb = shpwkb.dumps(geo)

            # Create an output shapefile from geometry
            return Vector([_wkb], projection=self.projection)

    @property
    def mask(self):
        """Return a Raster instance of the data mask (boolean)"""
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
        return int(xoff), int(yoff), int(win_xsize), int(win_ysize)

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
        Slice a Raster using numpy-like syntax.
        :param s: item for slicing, may be a slice object, integer, instance of the Extent class
        :return:
        """
        # TODO: add boolean, fancy, and Extent slicing
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
                        print("Warning: No data values must be fixed.")
                        self.ndchecked = True
                        self.fix_nodata()
                self.activeBand = ab
            # Tease gdal band args from s
            try:
                xoff, yoff, win_xsize, win_ysize =\
                    self.gdal_args_from_slice(s, self.shape)
            except:
                raise RasterError('Boolean and fancy indexing currently'
                                  ' unsupported for GDAL Raster data sources.'
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
            #     raise RasterError('Error writing Raster data. Check that mode'
            #                       ' is "r+" and that the arrays match.\n\nMore'
            #                       ' info:\n%s' % e)

    def perform_operation(self, r, op):
        """
        Factory for all operand functions
        :param r: value
        :param op: operand
        :return: Raster instance
        """
        if isinstance(r, numbers.Number) or isinstance(r, numpy.ndarray):
            out = self.full(r)
        else:
            try:
                r = Raster(r)
            except:
                raise RasterError('Expected a number, numpy array, or valid Raster while'
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
        :param r: number, array, or Raster
        :param op: operand
        :return: Raster instance
        """
        if self.mode == 'r':
            raise RasterError('%s open as read-only.' %
                              os.path.basename(self.path))
        if isinstance(r, numbers.Number) or isinstance(r, numpy.ndarray):
            r = self.full(r)
        else:
            try:
                r = Raster(r)
            except:
                raise RasterError('Expected a number, numpy array, or valid Raster while'
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
        Wrap the numpy conditional operators using underlying Raster data
        :param r:
        :param op:
        :return:
        """
        is_array = False
        if any([isinstance(r, t) for t in [numpy.ndarray, int, float]] +
               [type(r).__module__ == numpy.__name__]):
            is_array = True
            r = numpy.broadcast_to(r, self.shape)
        else:
            r = assert_type(r)(r)
            if isinstance(r, Vector):
                r = r.rasterize(self)
            elif isinstance(r, Raster):
                r = r.match_raster(self)

        # Create a mask Raster where rasters match
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
            prestr = ('A happy Raster named %s of house %s\n' %
                      (os.path.basename(self.path), self.format.upper()))
        else:
            prestr = ('An orphaned Raster %s of house %s\n' %
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
                    print("Unable to remove temporary file {} because:\n{}".format(
                        self.garbage['path'], e))


class mosaic(Raster):
    """Handle a mosaic of rasters"""

    def __init__(self, raster_list):
        self.rasterList = []
        self.extents = []
        self.rasterDtypes = []
        self.cellSizes = []
        projection = None
        for inputRaster in raster_list:
            rast = Raster(inputRaster)
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
        top, bottom, left, right = list(zip(*self.extents))
        top, bottom, left, right = max(top), min(bottom), min(left), max(right)
        csx, csy = list(zip(*self.cellSizes))
        csx, csy = min(csx), min(csy)

        # Calculate shape
        shape = (int(numpy.ceil((top - bottom) / csy)),
                 int(numpy.ceil((right - left) / csx)))

        # Collect the most precise data type
        precision = numpy.argmax([numpy.dtype(dtype).itemsize for dtype in self.rasterDtypes])
        dtype = self.rasterDtypes[precision]

        # Build a new Raster using the combined specs
        path = generate_name(self.path, 'rastermosaic', 'h5')
        # TODO: Add this path to the garbage of the output
        super(mosaic, self).__init__(path, mode='w', csx=csx, csy=csy, top=top, left=left,
                                     shape=shape, projection=projection, dtype=dtype)
        # Create a mask to track where data are merged
        with h5py.File(self.path, libver='latest', mode='a') as f:
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
        Update interpolation methods for each Raster
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
    Round a Raster to a defined number of decimal places
    :param input_raster: Raster to be rounded
    :param decimal_places: (int) number of decimal places
    :return: Raster instance
    """
    # Open everything
    dp = int(decimal_places)
    r = Raster(input_raster)
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
    """Copy a Raster dataset"""
    return Raster(input_raster).copy('copy')


def empty(input_raster):
    return Raster(input_raster).empty()


def full(input_raster):
    return Raster(input_raster).full()


def unique(input_raster):
    """Compute unique values"""
    r = Raster(input_raster)
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
    :param input_raster: A Raster-compatible object
    :return: Minimum value
    """
    r = Raster(input_raster)
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
            raise ValueError('Cannot collect minimum: No data in Raster')
    else:
        try:
            return numpy.min(r.array[r.array != r.nodata])
        except ValueError:
            raise ValueError('Cannot collect minimum: No data in Raster')


def rastmax(input_raster):
    """
    Calculate the maximum value
    :param input_raster: A Raster-compatible object
    :return: Maximum value
    """
    r = Raster(input_raster)
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
            raise ValueError('Cannot collect maximum: No data in Raster')
    else:
        try:
            return numpy.max(r.array[r.array != r.nodata])
        except ValueError:
            raise ValueError('Cannot collect maximum: No data in Raster')


class Vector(object):
    """Vector data handler"""

    GEOMETRIES = {'Unknown': ogr.wkbUnknown,
                  'Point': ogr.wkbPoint,
                  'LineString': ogr.wkbLineString,
                  'Polygon': ogr.wkbPolygon,
                  'MultiPoint': ogr.wkbMultiPoint,
                  'MultiLineString': ogr.wkbMultiLineString,
                  'MultiPolygon': ogr.wkbMultiPolygon,
                  'GeometryCollection': ogr.wkbGeometryCollection,
                  'None': ogr.wkbNone,
                  'LinearRing': ogr.wkbLinearRing,
                  'Point25D': ogr.wkb25DBit,
                  'LineString25D': ogr.wkbLineString25D,
                  'Polygon25D': ogr.wkbPolygon25D,
                  'MultiPoint25D': ogr.wkbMultiPoint25D,
                  'MultiLineString25D': ogr.wkbMultiLineString25D,
                  'MultiPolygon25D': ogr.wkbMultiPolygon25D,
                  'GeometryCollection25D': ogr.wkbGeometryCollection25D
                  }

    def __init__(self, data=None, mode='r', fields=None, projection=None, geotype=None):
        """
        Vector interfacing class
        :param data: Input data- may be one of:
            1. Path (string)
            2. ogr Vector instance
            3. Table
            4. wkt geometry
            5. wkb geometry
            6. A list of wkt's or wkb's
        """
        self.mode = mode.lower()
        if self.mode not in ['r', 'w', 'r+']:
            raise VectorError('Unsupported file mode {}'.format(mode))
        populate_from_ogr = False
        if isinstance(data, str):
            if os.path.isfile(data) and mode != 'w':
                # Open a Vector file or a table
                self.driver = self.get_driver_by_path(data)
                if self.driver == 'table':
                    # TODO: Implement table driver
                    self.path = None
                else:
                    # Open dataset
                    driver = ogr.GetDriverByName(self.driver)
                    _data = driver.Open(data)
                    if _data is None:
                        raise VectorError('Unable to open the dataset {} as a Vector'.format(data))
                    populate_from_ogr = True
                    self.path = data
            elif mode == 'w':
                # Create an empty data source using an input geometry
                geotypes = ['Polygon', 'LineString', 'Point']
                # Check input geometry type to make sure it works
                if geotype not in ['Polygon', 'LineString', 'Point']:
                    raise VectorError(
                        'Geometry type {} not understood while creating new Vector data source.  '
                        'Use one of: {}'.format(geotype, ', '.join(geotypes))
                        )

                # Create some attributes so Vector.empty can be called
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

        elif isinstance(data, Vector):
            # Update to new Vector instance
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

            # Make sure the same number of fields were specified
            if fields is not None and len(fields) != len(geos_and_types):
                raise VectorError('The number of input fields and well-known geometries do not match')

            # Create some attributes so Vector.empty can be called
            self.geometryType = geos_and_types[0][1]
            self.path = generate_name(None, '', 'shp')
            self.projection = parse_projection(projection)
            self.featureCount = len(geos_and_types)
            self.fieldWidth, self.fieldPrecision = {}, {}

            if fields is None:
                field_def = []
            else:
                self.fieldTypes = list(
                    zip(fields.dtype.names, (fields.dtype[i].name for i in range(len(fields.dtype)))))
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

            self.__dict__.update(Vector(outVect.path, mode='r+').__dict__)

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

    @property
    def field_data(self):
        """
        Return all fields in a numpy structured array
        """
        fields = [(value,) for value in self[self.fieldTypes[0][0]]]

        for field in self.fieldTypes[1:]:
            for i, value in enumerate(self[field[0]]):
                fields[i] += (value,)

        return numpy.array(fields, dtype=self.fieldTypes)

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
        Open the source Vector file for writing
        :return: ogr Vector layer instance
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
        Save the Vector instance
        :param path: Output save path
        :return: Vector instance where saved
        """
        # TODO: Allow NoneType path and save as in-memory Vector
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

    def empty(self, spatial_reference=None, fields=[], prestr='copy', out_path=None, geom_type=None):
        """
        Create a copy of self as an output shp without features
        :return: Fresh Vector instance
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
        if geom_type is not None:
            out_geotype = self.GEOMETRIES[geom_type]
        else:
            out_geotype = self.geometry_type
        layer = ds.CreateLayer('bluegeo Vector', outsr, out_geotype)

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

        # Clean up and return Vector instance
        del layer
        ds.Destroy()

        return_vect = Vector(out_path, mode='r+')
        if add_to_garbage:
            return_vect.garbage = {'path': out_path, 'num': 1}
        return return_vect

    def add_fields(self, name, dtype, data=None):
        """
        Create a new Vector with the output field(s)
        :param name: single or list of names to create
        :param dtype: single or list of data types for each field
        :param data: list of lists to write to each new field
        :return: Vector instance
        """
        if isinstance(name, str):
            name, dtype = [name], [dtype]  # Need to be iterable (and not strings)

        if self.mode in ['r+', 'w']:
            # Iterate features and insert data
            with self.layer() as inlyr:
                outLyrDefn = inlyr.GetLayerDefn()
                for _i, (_name, _dtype) in enumerate(zip(name, dtype)):
                    # Modify the current Vector with the field
                    fieldDefn = ogr.FieldDefn(_name, self.numpy_dtype_to_ogr(_dtype))
                    # Create field
                    inlyr.CreateField(fieldDefn)
                    self.fieldTypes.append((_name, _dtype))
                    self.fieldPrecision[_name] = self._get_precision(_dtype)
                    self.fieldWidth[_name] = self._get_width(_dtype, data[_i] if data is not None else [' ' * 254])
                    self.fieldCount += 1
                if data is not None:
                    # Broadcast data to the necessary shape
                    shape = 1 if isinstance(name, str) else len(name)
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
            new_fieldtypes = self.check_fields(self.fieldTypes + list(zip(name, dtype)))

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

            return Vector(outVect)  # Re-read Vector to ensure meta up to date

    def transform(self, sr):
        """
        Project dataset into a new system
        :param sr: Input spatial reference
        :return: Vector instance
        """
        # Create ogr coordinate transformation object
        insr = osr.SpatialReference()
        outsr = osr.SpatialReference()
        if self.projection == '':
            raise VectorError('Source Vector has an unknown spatial reference, and cannot be transformed')
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
        return Vector(outVect)  # Re-read Vector to ensure meta up to date

    @staticmethod
    def _get_precision(dtype):
        if 'float' in dtype.lower():
            return 6
        else:
            return 0

    @staticmethod
    def _get_width(dtype, data):
        if 'float' in dtype.lower():
            return 15
        elif 'int' in dtype.lower():
            return 9
        else:
            return max([len(e) for e in data])

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
                vertices += [[geo.GetX(i), geo.GetY(i), geo.GetZ(i)] for i in range(geo.GetPointCount())]

        vertices = []
        for wkb in self[:]:
            try:
                get_next(ogr.CreateGeometryFromWkb(wkb), vertices)
            except:
                pass

        return numpy.array(vertices)

    def round(self, precision=2):
        """Round vertices of features to a specified precision

        Keyword Arguments:
            precision {int} -- Precision for rounding (default: {2})
        """
        precision = int(precision)

        output = self.empty()

        geos = []
        for geo in self[:]:
            geo = shpwkb.loads(geo)
            if 'Multi' not in geo.geom_type:
                geo = [geo]

            rounded = []
            for g in geo:
                try:
                    coords = g.coords
                except NotImplementedError:
                    coords = g.boundary.coords
                coords = [tuple(numpy.round(c, precision)) for c in coords]
                rounded.append(getattr(geometry, g.geom_type)(coords))

            if len(rounded) == 1:
                geos.append(shpwkb.dumps(rounded[0]))
            else:
                geos.append(shpwkb.dumps(getattr(geometry, 'Multi{}'.format(rounded[0].geom_type))(rounded)))

        output[:] = geos
        field_data = []
        for field, _ in self.fieldTypes:
            field_data.append(self[field])
        output.add_fields([f[0] for f in self.fieldTypes], [f[1] for f in self.fieldTypes], field_data)

        return output

    def spatially_unique(self, round=None):
        """
        Remove spatially-redundant features
        Attributes from the first feature will be preserved when redundancy occurs

        :param round: Round vertices to a tolerance when evaluating uniqueness
        """
        if self.geometryType != 'Point':
            raise VectorError('Only point geometries are supported for this operation')

        if round is not None:
            geos = self.round(round)[:]
        else:
            geos = self[:]
        un, ind = numpy.unique(geos, return_index=True)

        output = self.empty()
        output[:] = [geos[i] for i in ind]
        field_data = []
        for field, _ in self.fieldTypes:
            field_data.append(self[field][ind])
        output.add_fields([f[0] for f in self.fieldTypes], [f[1] for f in self.fieldTypes], field_data)

        return output

    def buffer(self, distance):
        """
        Perform a buffer operation on the vector
        :param float distance: Width of the buffer
        :return: Vector instance
        """
        distance = float(distance)

        # Create empty output
        out_vect = self.empty(self.projection, self.fieldTypes, prestr='buffer', geom_type='Polygon')

        # Transform geometries and populate output
        with self.layer() as inlyr:
            with out_vect.layer() as outlyr:
                # Output layer definition
                outLyrDefn = outlyr.GetLayerDefn()
                # Gather field values to write to output
                new_fields = self.check_fields(self.fieldTypes)
                fields = {newField[0]: self.field_to_pyobj(self[oldField[0]], self.fieldWidth[oldField[0]],
                                                           self.fieldPrecision[oldField[0]])
                          for oldField, newField in zip(self.fieldTypes, new_fields)}

                # Iterate geometries and populate output with buffered geo's
                for i in range(self.featureCount):
                    # Gather and transform geometry
                    inFeat = inlyr.GetFeature(i)
                    geo = inFeat.GetGeometryRef()
                    buffered_geo = geo.Buffer(distance)

                    # Write output fields and geometries
                    out_feature = ogr.Feature(outLyrDefn)
                    for name, dtype in new_fields:
                        out_feature.SetField(name, fields[name][i])
                    out_feature.SetGeometry(buffered_geo)

                    # Add to output layer and clean up
                    outlyr.CreateFeature(out_feature)
                    out_feature.Destroy()

        return Vector(out_vect)  # Re-read Vector to ensure meta up to date

    def dissolve(self, field=None, summary_method='first'):
        """
        Perform a dissolve operation
        :param str field: Field to use as dissolve filter
        :return: Vector instance
        """
        # Check if the field exists before commencing the algorithm
        if field is not None and not field in self.fieldNames:
            raise VectorError('The field {} does not exist'.format(field))

        # Create empty output
        out_vect = self.empty(self.projection, self.fieldTypes, prestr='dissolve')

        newFields = out_vect.fieldTypes
        fields = {(oldField[0], newField[0]): self.field_to_pyobj(self[oldField[0]], self.fieldWidth[oldField[0]],
                                                                  self.fieldPrecision[oldField[0]])
                  for oldField, newField in zip(self.fieldTypes, newFields)}

        # Collect the input features
        if field is not None:
            field_data = self[field]
            feat_values = numpy.unique(field_data)

            # A dissolve is redundant if all fields are unique
            if feat_values.size == self.featureCount:
                print("Warning, no dissolve took place because the input field has entirely unique values")
                return self
        else:
            feat_values = [None]

        # Open layers and populate with dissolved data
        with self.layer() as _:
            with out_vect.layer() as outlyr:
                # Output layer definition
                outLyrDefn = outlyr.GetLayerDefn()

                # Iterate the output features and dissolve using shapely.ops.cascaded_union
                for i, feat_value in enumerate(feat_values):
                    # Gather geometries
                    if field is None:
                        indices = numpy.arange(self.featureCount)
                    else:
                        indices = numpy.where(field_data == feat_value)[0]

                    geos = [shpwkb.loads(self[ind][0]) for ind in indices]

                    # Perform union
                    out_geo = shpwkb.dumps(ops.cascaded_union(geos))

                    # Write output geometries
                    out_feature = ogr.Feature(outLyrDefn)
                    out_feature.SetGeometry(ogr.CreateGeometryFromWkb(out_geo))
                    for name_tup, data in fields.items():
                        old_name, new_name = name_tup
                        if old_name == field:
                            write_data = feat_value
                        else:
                            if summary_method == 'first':
                                write_data = data[indices[0]]
                            else:
                                write_data = getattr(numpy, summary_method)([data[ind] for ind in indices])
                        out_feature.SetField(new_name, write_data)

                    # Add to output layer and clean up
                    outlyr.CreateFeature(out_feature)
                    out_feature.Destroy()

        return Vector(out_vect)  # Re-read Vector to ensure meta up to date

    def select(self, field, values):
        """
        Select by attribute
        :param field: field name
        :param value: select value(s)
        :return: new vector instance
        """
        # Check if the field exists before commencing the algorithm
        if field is not None and not field in self.fieldNames:
            raise VectorError('The field {} does not exist'.format(field))

        # Create empty output
        out_vect = self.empty(self.projection, self.fieldTypes, prestr='select')

        newFields = out_vect.fieldTypes

        # Collect the input features
        field_data = self[field]
        locations = numpy.where(numpy.in1d(field_data, values))[0]

        fields = {newField[0]: self.field_to_pyobj(self[oldField[0]][locations], self.fieldWidth[oldField[0]],
                                                   self.fieldPrecision[oldField[0]])
                  for oldField, newField in zip(self.fieldTypes, newFields)}

        # Open layers and populate with selected data
        with self.layer() as _:
            with out_vect.layer() as outlyr:
                # Output layer definition
                outLyrDefn = outlyr.GetLayerDefn()

                # Iterate indices and population output
                for field_i, i in enumerate(locations):
                    # Write output geometry
                    out_feature = ogr.Feature(outLyrDefn)
                    out_feature.SetGeometry(ogr.CreateGeometryFromWkb(self[i][0]))

                    # Write fields
                    for name, data in fields.items():
                        out_feature.SetField(name, data[field_i])

                    # Add to output layer and clean up
                    outlyr.CreateFeature(out_feature)
                    out_feature.Destroy()

        return Vector(out_vect)  # Re-read Vector to ensure meta up to date

    def overlay_operation(self, other_vector, prestr):
        """
        Boilerplate for vector overlay operations
        :param other_vector: this
        :return:
        """
        other_vector = Vector(other_vector).transform(self.projection)
        geos = [shpwkb.loads(geo) for geo in other_vector[:]]

        # Create an rtree of the geometries
        # Create spatial index using resulting multipolygon
        def gen_idx():
            """Generator for spatial index"""
            for i, geo in enumerate(geos):
                yield (i, geo.bounds, None)

        idx = index.Index(gen_idx())

        # Add fields from the other geometry, while making sure there are no duplicates
        fields = [(name, dtype) for name, dtype in self.fieldTypes]
        field_data = {name: (self.field_to_pyobj(self[name], self.fieldWidth[name], self.fieldPrecision[name]), 'i')
                      for name, dtype in self.fieldTypes}  # Save the field data source for positioning

        field_names = list(field_data.keys())
        for name, dtype in other_vector.fieldTypes:
            if name in field_names:
                new_name = name[:8] + '_1'
            else:
                new_name = name
            field_data[new_name] = (other_vector.field_to_pyobj(other_vector[name], other_vector.fieldWidth[name],
                                                                other_vector.fieldPrecision[name]), 'j')
            fields.append((new_name, dtype))

        # The simplest geometry is preserved
        simplest = {'Point': 0, 'MultiPoint': 1, 'LineString': 2, 'MultiLineString': 3,
                    'Polygon': 4, 'MultiPolygon': 5}
        geom_enum = [simplest[geos[0].geom_type], simplest[shpwkb.loads(self[0][0]).geom_type]]
        copy_instance = [other_vector, self][numpy.argmin(geom_enum)]
        out_vect = copy_instance.empty(self.projection, fields, prestr=prestr)

        return out_vect, fields, field_data, other_vector, idx, geos

    def intersect(self, other_vector):
        """
        Perform an intersect operation
        :param other_vector: Another vector
        :return: new Vector instance
        """
        out_vect, fields, field_data, other_vector, idx, geos = self.overlay_operation(other_vector, 'intersect')

        # Open layers and populate with intersected data
        with self.layer() as _:
            with out_vect.layer() as outlyr:
                # Output layer definition
                outLyrDefn = outlyr.GetLayerDefn()

                # Iterate parent geometries, find intersecting other geometries and perform intersect
                for i in range(self.featureCount):
                    parent_geo = shpwkb.loads(self[i][0])

                    for j in idx.intersection(parent_geo.bounds):
                        # Perform intersection
                        try:
                            intersection = parent_geo.intersection(geos[j])
                        except Exception as e:
                            print("Warning:\n{}".format(e))
                            # Likely a topological error
                            continue
                        if intersection.is_empty:
                            continue

                        # If the intersection creates a geometry collection, reduce it to the
                        # parent geometry type
                        if intersection.type == 'GeometryCollection':
                            intersection = [_geo for _geo in intersection if
                                            _geo.type.replace('Multi', '') == out_vect.geometryType.replace('Multi', '')]
                            intersection = getattr(
                                geometry, 'Multi' + out_vect.geometryType.replace('Multi', '')
                                )(intersection)

                        # Write output geometry
                        out_feature = ogr.Feature(outLyrDefn)
                        out_feature.SetGeometry(ogr.CreateGeometryFromWkb(shpwkb.dumps(intersection)))

                        # Write fields
                        for name, dtype in fields:
                            data, pos = field_data[name]
                            if pos == 'i':
                                out_feature.SetField(name, data[i])
                            else:
                                out_feature.SetField(name, data[j])

                        # Add to output layer and clean up
                        outlyr.CreateFeature(out_feature)
                        out_feature.Destroy()

        return Vector(out_vect)  # Re-read Vector to ensure meta up to date

    def explode(self, name_field=None, file_type='shp'):
        """
        Explode features into separate files
        """
        for i in range(self.featureCount):
            if name_field is not None:
                if name_field not in self.fieldNames:
                    raise ValueError('The field {} does not exist')
            if name_field is not None:
                out_name = self[name_field][i]
            else:
                out_name = i
            print("Extracting {}".format(out_name))
            try:
                geo = self[i]
            except AttributeError:
                print("Warning: Error with geometry, skipping the above")
                continue
            dtype = [(field, self[field].dtype) for field in self.fieldNames]
            fields = numpy.array([tuple([self[field][i] for field in self.fieldNames])], dtype=dtype)
            base_path = '.'.join(os.path.basename(self.path).split('.')[:-1]) + '__{}.{}'.format(
                out_name, file_type)
            out_path = os.path.join(os.path.dirname(self.path), base_path)
            Vector(geo, fields=fields, projection=self.projection).save(out_path)

    def fix(self):
        """
        Remove null and invalid features to ensure smooth sailing
        :return: Vector instance
        """
        # Create empty output
        out_vect = self.empty(self.projection, self.fieldTypes, prestr='fixed')

        # Transform geometries and populate output
        with self.layer() as inlyr:
            with out_vect.layer() as outlyr:
                # Output layer definition
                outLyrDefn = outlyr.GetLayerDefn()
                # Gather field values to write to output
                new_fields = self.check_fields(self.fieldTypes)
                fields = {newField[0]: self.field_to_pyobj(self[oldField[0]], self.fieldWidth[oldField[0]],
                                                           self.fieldPrecision[oldField[0]])
                          for oldField, newField in zip(self.fieldTypes, new_fields)}

                # Iterate geometries and populate where valid or not null
                removed = 0
                for i in range(self.featureCount):
                    # Gather and transform geometry
                    inFeat = inlyr.GetFeature(i)
                    geo = inFeat.GetGeometryRef()

                    # Do the test
                    passed = True
                    try:
                        _wkb = geo.ExportToWkb()
                        shp_geo = shpwkb.loads(_wkb)
                        if any([_wkb is None, not shp_geo.is_valid, shp_geo.is_empty]):
                            passed = False
                    except:
                        passed = False

                    if passed:
                        out_feature = ogr.Feature(outLyrDefn)
                        for name, dtype in new_fields:
                            out_feature.SetField(name, fields[name][i])
                        out_feature.SetGeometry(ogr.CreateGeometryFromWkb(_wkb))

                        # Add to output layer and clean up
                        outlyr.CreateFeature(out_feature)
                        out_feature.Destroy()
                    else:
                        removed += 1

        if removed > 0:
            print("Removed {} of {} geometries".format(removed, self.featureCount))
        else:
            print("All geometries passed test")

        return Vector(out_vect)  # Re-read Vector to ensure meta up to date

    def __getitem__(self, item):
        """
        :param item: If an index or slice is used, geometry wkb's are returned.f
        If a string is used, it will return a numpy array of the field values.
        If an instance of the Extent class is used, the output will be a clipped Vector instance
        :return: List of wkb's or numpy array of field
        """
        with self.layer() as lyr:
            if any([isinstance(item, obj) for obj in [slice, int, numpy.ndarray, list, tuple,
                                                      numpy.int64, numpy.int32, numpy.int16, numpy.int8]]):
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
                        data.append(shpwkb.dumps(getattr(geometry, self.geometryType)([])))
                    feature.Destroy()

            elif isinstance(item, str):
                # Check that the field exists
                try:
                    _, dtype = [ft for ft in self.fieldTypes if ft[0] == str(item)][0]
                except IndexError:
                    raise VectorError('No field named {} in the file {}'.format(item, self.path))

                data = [lyr.GetFeature(i).GetField(item) for i in numpy.arange(self.featureCount)]
                try:
                    data = numpy.array(data, dtype=dtype)
                except:
                    print("Warning: could not read field {} as {}, and it has been cast to string".format(item, dtype))
                    data = numpy.array(list(map(str, data)))

            # TODO: Finish this- need to figure out how ogr.Layer.Clip works. Maybe through an in-memory ds
            elif isinstance(item, Extent):
                # Clip and return a Vector instance
                data = self.empty(self.projection, self.fieldTypes, prestr='clip')
                # Perform a clip
                with item.layer as cliplyr:
                    with data.layer as outlyr:
                        if item.geo is not None:
                            # With geometry
                            cliplyr.Clip(cliplyr, outlyr)
                        else:
                            # With Extent
                            cliplyr.Clip(extlyr, outlyr)

        return data

    def __setitem__(self, item, val):
        """Set a field or geometry if the Vector is open for writing"""
        if self.mode not in ['r+', 'w']:
            raise VectorError('Current Vector open as read-only')

        # Check the type of input
        if isinstance(item, str):
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
            start, stop = item.start, item.stop
            # Make sure start and stop are not slice member descriptors
            if not isinstance(start, int):
                start = 0
            if not isinstance(stop, int):
                if self.featureCount == 0:
                    stop = len(val) if hasattr(val, '__iter__') else 1
                else:
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
        if not isinstance(val[0], (str, bytes)):
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
            prestr = ('A happy Vector named %s of house %s\n' %
                      (os.path.basename(self.path), self.driver))
        else:
            prestr = ('An orphaned Vector %s of house %s\n' %
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
        dtype = a.dtype.name
        if any([dtype[:5].lower() == 'bytes',
                dtype[0].lower() == 's',
                width is None,
                prec is None,
                'datetime' in dtype.lower()]):
            dtype = 's'
            fmt = '{}'
        elif 'float' in dtype.lower():
            fmt = '{' + ':{}.{}f'.format(width, prec) + '}'
        else:
            fmt = '{:d}'

        return list(map(fmt.format, a))

    @staticmethod
    def drivers():
        """
        Dict of drivers
        :return: dict of drivers
        """
        return {'shp': 'ESRI Shapefile',
                'kml': 'KML',
                'kmz': 'LIBKML',
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
            return Vector.drivers()[ext]
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
                  'int16': ogr.OFTInteger,
                  'uint16': ogr.OFTInteger,
                  'int32': ogr.OFTInteger,
                  'uint32': ogr.OFTInteger,
                  'int64': ogr.OFTInteger,
                  'uint64': ogr.OFTInteger,
                  'int8': ogr.OFTInteger,
                  'uint8': ogr.OFTInteger,
                  'bool': ogr.OFTInteger,
                  'datetime64': ogr.OFTDate,
                  's': ogr.OFTString,
                  'bytes': ogr.OFTBinary
                  }

        if dtype[0].lower() == 's':
            dtype = 's'
        elif dtype[:5].lower() == 'bytes':
            dtype = 'bytes'

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
                      'Date': 'datetime64',
                      'DateTime': 'datetime64'}

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
        return self.GEOMETRIES[self.geometryType]

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
                  3001: 'wkbPointZM',
                  3002: 'wkbLineStringZM',
                  3003: 'wkbPolygonZM',
                  3004: 'wkbMultiPointZM',
                  3005: 'wkbMultiLineStringZM',
                  3006: 'wkbMultiPolygonZM',
                  3007: 'wkbGeometryCollectionZM',
                  3008: 'wkbCircularStringZM',
                  3009: 'wkbCompoundCurveZM',
                  3010: 'wkbCurvePolygonZM',
                  3011: 'wkbMultiCurveZM',
                  3012: 'wkbMultiSurfaceZM',
                  3013: 'wkbCurveZM',
                  3014: 'wkbSurfaceZM',
                  3015: 'wkbPolyhedralSurfaceZM',
                  3016: 'wkbTINZM',
                  3017: 'wkbTriangleZM'}

        try:
            return _types[code]
        except KeyError:
            raise VectorError('Unrecognized Geometry type OGRwkbGeometryType: {}'.format(code))

    def rasterize(self, template_raster, attribute_field=None):
        """
        Create a Raster from the current instance
        :param template_raster: Raster to use specs from
        :param attribute_field: attribute field to use for Raster values (returns a mask if None)
            if the attribute field is a string, a categorical raster is created, and a mapping of
            the field is returned.
        :return: Raster instance
        """
        # Grab the Raster specs
        r = Raster(template_raster)
        top, left, nrows, ncols, csx, csy = r.top, r.left, r.shape[0], r.shape[1], r.csx, r.csy

        # Transform self if necessary
        vector = self.transform(r.projection)

        # Make sure they overlap
        if not Extent(r).intersects(Extent(vector)):
            return r.full(r.nodata)

        # Collect a mask vector from the raster and trim by half of the smallest cell size.
        #   This will ensure polygons that intersect the raster boundary will retain vertices at the edges
        extent = shpwkb.loads(r.extent_to_vector()[0][0]).buffer(min(r.csx, r.csy) / -2.)
        extent = ogr.CreateGeometryFromWkb(shpwkb.dumps(extent))

        # Grab the data type from the input field
        return_map = False
        if attribute_field is not None:
            try:
                dtype = [field[1] for field in vector.fieldTypes if field[0] == attribute_field][0]
            except IndexError:
                raise VectorError('Cannot find the field {} during rasterize'.format(attribute_field))
            # If dtype is a string, try to cast the field into a float, else enumerate
            write_data = vector[attribute_field]
            if 's' in dtype.lower():
                try:
                    write_data = numpy.float32(write_data)
                except:
                    return_map = True
                    unique_values, indices = numpy.unique(write_data, return_inverse=True)
                    values = numpy.arange(1, unique_values.size + 1)
                    write_data = values[indices]
                    value_map = dict(list(zip(values, unique_values)))
                    dtype = 'uint32'

            nodata = numpy.array(r.nodata).astype(dtype)
        else:
            nodata = 0
            dtype = 'bool'
            write_data = numpy.ones(shape=vector.featureCount)

        # Allocate output array and Raster for writing Raster values
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
            pixels = coords_to_indices(list(zip(*vertices)), top, left, csx, csy, r.shape)

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
                window_image.polygon(list(zip(col_inds, row_inds)), outline=1, fill=1)
            elif 'Line' in geom_type:
                # Draw a line mask (1)
                window_image.line(list(zip(col_inds, row_inds)), 1)

            # Return output image as array (which needs to be reshaped) and the insertion location
            return numpy.array(window, dtype='bool').reshape(nrows, ncols), i_insert, j_insert

        # Need to track where data have been inserted to ensure holes don't overwrite data from other features
        data_track = numpy.zeros(shape=outarray.shape, dtype='bool')
        for feature_index, wkb in enumerate(vector[:]):
            vertices = []
            geo = ogr.CreateGeometryFromWkb(wkb)

            # Intersect the geometry with the raster extent
            geo = extent.Intersection(geo)
            if geo is None or geo.IsEmpty():
                continue

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

        if return_map:
            return outrast, value_map

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


def merge_vectors(vectors, projection=None):
    """
    Merge a list of vectors of the same geometry type

    Arguments:
        vectors {iterable} -- datasets compatible with the Vector class
    """
    if len(vectors) == 1:
        return Vector(vectors[0])

    vectors = [Vector(v) for v in vectors]
    if not all([v.geometryType == vectors[0].geometryType for v in vectors[1:]]):
        raise VectorError('All input geometries must be alike')

    if projection is None:
        projection = vectors[0].projection

    vectors = [v.transform(projection) for v in vectors]
    field_keys = []
    for v in vectors:
        field_keys += [f[0] for f in v.fieldTypes]
    field_keys = numpy.unique(field_keys)

    output = vectors[0].empty()
    geos = []
    fields = dict(zip(field_keys, [[] for i in range(len(field_keys))]))

    for v in vectors:
        geos += v[:]
        for field in field_keys:
            try:
                data = list(v[field])
            except VectorError:
                data = [None for i in range(v.featureCount)]
            fields[field] += data

    output[:] = geos
    field_data = []
    for f in field_keys:
        dtype = numpy.array([i for i in fields[f] if i is not None]).dtype.name
        if 'bytes' in dtype:
            dtype = 'bytes'
        elif 'str' in dtype:
            dtype = 'str'
        if 'float' in dtype or 'int' in dtype:
            replace_val = 0
        else:
            replace_val = ''
        field_data.append(numpy.array([i if i is not None else replace_val for i in fields[f]], dtype=dtype))
    field_types = [f.dtype.name for f in field_data]
    output.add_fields(field_keys, field_types, field_data)

    return output


def vector_stats(polygons, datasets, out_csv):
    """
    Peform summary statistics on a list of datasets within specified polygons

    Arguments:
        polygons {Vector} -- Vector Polygon dataset
        datasets {iterable} -- Iterable of Rasters and Vectors
    """
    zones = Vector(polygons)
    if zones.geometryType != 'Polygon':
        raise VectorError('Only Polygon geometries are supported for vector stats')

    stats = ['min', 'max', 'mean', 'sum', 'std', 'var']

    with open(out_csv, 'w') as f:
        f.write('Dataset,{}\n'.format(','.join(stats)))
    for data in datasets:
        data = assert_type(data)(data)
        if isinstance(data, Raster):
            r = data.clip(zones)
            a = numpy.ma.masked_equal(r[:], r.nodata)
            with open(out_csv, 'a') as f:
                f.write('{}\n'.format(','.join([data.path] + [getattr(numpy, stat)(a) for stat in stats])))
        else:
            v = data.intersect(zones)
            for field, _ in data.fieldTypes:
                f.write('{}\n'.format(','.join(['{}: {}'.format(data.path, field)] +
                                               [getattr(numpy, stat)(v[field]) for stat in stats])))


def force_gdal(input_raster):
    r = Raster(input_raster)
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
    :return: object to instantiate (Raster, Vector, or Extent)
    """
    def raise_type_error():
        raise TypeError('Unable to provide a means to open {}'.format(data))

    if isinstance(data, Vector):
        return Vector
    if isinstance(data, Raster):
        return Raster
    if isinstance(data, Extent):
        return Extent
    if any([isinstance(data, t) for t in [list, tuple, numpy.ndarray]]) and len(data) == 4:
        # An iterable that is expected to be a bounding box
        return Extent

    if isinstance(data, str):
        # Check if gdal Raster
        ds = gdal.Open(data)
        if ds is not None:
            return Raster

        # Check if a bluegeo Raster
        if data.split('.')[-1].lower() == 'h5':
            with h5py.File(data, libver='latest', mode='r') as ds:
                # Just check for the format attribute
                if 'format' in list(dict(ds.attrs).keys()):
                    return Raster

        # Check if a Vector
        try:
            driver = Vector.get_driver_by_path(data)
        except:
            raise_type_error()
        if driver == 'table':
            return Vector
        else:
            # Try to open dataset
            driver = ogr.GetDriverByName(driver)
            _data = driver.Open(data)
            if _data is None:
                raise_type_error()
            else:
                return Vector

    else:
        raise ValueError('Unable to parse the data of type "{}" using the spatial library'.format(type(data).__name__))
