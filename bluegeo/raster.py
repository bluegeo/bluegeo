'''
Blue Geosimulation, 2016
Raster data interfacing and manipulation library
'''

import os
import numbers
import decimal
from datetime import datetime
import numpy
from osgeo import gdal, osr, gdalconst
import h5py
try:
    import numexpr as ne
    numexpr_ = True
except ImportError:
    numexpr_ = False


class RasterError(Exception):
    '''Custom Exception'''
    pass


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
    def __init__(self, data, mode='r', **kwargs):
        # Record mode
        if mode in ['r', 'r+', 'w']:
            self.mode = mode
        else:
            raise RasterError('Unrecognized file mode "%s"' % mode)

        # Check if data is a string
        if isinstance(data, basestring):
            # If in 'w' mode, write a new file
            if self.mode == 'w':
                self.save(data, **kwargs)
            # Check if data is a valid file
            elif not os.path.isfile(data):
                raise RasterError('%s is not a file' % data)
            else:
                # Try an HDF5 data source
                try:
                    ds = h5py.File(data)
                    self.load_from_hdf5(ds)
                    self.format = 'HDF5'
                except:
                    # Try for a gdal dataset
                    ds = gdal.Open(data)
                    try:
                        gt = ds.GetGeoTransform()
                        assert gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                        self.load_from_gdal(ds)
                        self.format = 'gdal'
                    except:
                        raise RasterError('Unable to open dataset %s' % data)
        # If not a string, maybe an osgeo dataset?
        elif isinstance(data, gdal.Dataset):
            self.load_from_gdal(data)
            self.format = 'gdal'
        # ...or an h5py dataset
        elif isinstance(data, h5py.File):
            self.load_from_hdf5(data)
            self.format = 'HDF5'
        # ...or a raster instance
        elif isinstance(data, raster):
            # Create a copy of the instance, except for dataset
            self.copy_ds('copy', data)

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
        if numexpr_:
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
            self.nodataValues.append(nd)
        self.path = ds.GetFileList()[0]

    def load_from_hdf5(self, ds):
        '''Load attributes from an HDF5 file'''
        d = {}
        for key, val in ds.attrs.iteritems():
            if not isinstance(val, numpy.ndarray) and val == 'None':
                val = None
            d[key] = val
        current_mode = self.mode
        self.__dict__.update(d)
        self.mode = current_mode
        self.shape = tuple(self.shape)
        self.path = str(ds.filename)

    def save(self, output_path, compression='lzf', empty=False, **kwargs):
        '''
        Save to a new file, and use with self
        '''
        # Get kwargs- for building a new raster
        build_raster = False
        if len(kwargs) > 0:
            build_raster = True
            self.projection = self.parse_projection(
                kwargs.get('projection', None))
            self.csy = float(kwargs.get('csy', 1))
            self.csx = float(kwargs.get('csx', 1))
            self.dtype = kwargs.get('dtype', None)
            self.left = float(kwargs.get('left', 0))
            self.shape = kwargs.get('shape', None)
            data = kwargs.get('data', None)
            if data is None and self.shape is None:
                raise RasterError('Either "data" or "shape" must be specified'
                                  ' when building a new raster.')
        # Build the new raster if necessary
        if build_raster:
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
                self.dtype, self.size = self.get_data_specs(self.dtype,
                                                            self.shape)
                # If array is 3-D, third dimension (axis 2) are the bands
                if data.ndim == 3:
                    self.bandCount = data.shape[2]
                else:
                    self.bandCount = 1
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
            # Use shape to build arbitary top if not specified
            self.top = float(kwargs.get('top', self.shape[0] * self.csy))
            self.nodataValues = kwargs.get('nodata', None)
            if self.nodataValues is None:
                self.nodataValues = [0 for i in self.bands]
            if type(self.nodataValues) != list:
                self.nodataValues = [self.nodataValues]

        # Recompute right and bottom to be certain they're right
        self.bottom = self.top - (self.csy * self.shape[0])
        self.right = self.left + (self.csx * self.shape[1])

        # Check compression
        if compression not in [None, 'lzf', 'gzip', 'szip']:
            raise RasterError('Unrecognized compression "%s"' % compression)

        # Check path
        if output_path.split('.')[-1] != 'h5':
            output_path = output_path + '.h5'

        # Change mode to read/write
        self.mode = 'r+'
        # Keep format in 'gdal' in case reading from self still
        if not hasattr(self, 'format'):
            self.format = 'HDF5'
        if not hasattr(self, 'path'):
            self.path = output_path

        # Create new HDF5 file
        newfile = h5py.File(output_path, mode='w')

        # Copy data from data source to new file
        try:
            prvb = self.activeBand
        except:
            self.activeBand = 1
            prvb = 1
        for band in self.bands:
            ds = newfile.create_dataset(str(band), self.shape,
                                        dtype=self.dtype,
                                        compression=compression,
                                        chunks=self.chunks)
            if build_raster:
                if data is not None:
                    if data.ndim == 3:
                        ds[:] = data[:, :, band - 1]
                    else:
                        ds[:] = data
                else:
                    if self.useChunks:
                        for a, s in self.iterchunks():
                            shape_ = self.gdal_args_from_slice(s, self.shape)
                            shape_ = (shape_[3], shape_[2])
                            ds[s] = numpy.full(shape_, self.nodata, self.dtype)
                    else:
                        ds[:] = numpy.full(self.shape, self.nodata, self.dtype)
            else:
                if not empty:
                    if self.useChunks:
                        for a, s in self.iterchunks():
                            ds[s] = a
                    else:
                        ds[:] = self.array
            del ds
        del newfile
        self.format = 'HDF5'
        self.path = output_path
        self.activeBand = prvb

        # Add attributes
        ds = self.ds
        for key, val in self.__dict__.iteritems():
            if key == 'garbage':
                continue
            if val is None:
                val = 'None'
            ds.attrs[key] = val

    def save_gdal_raster(self, output_path, compress=True):
        '''
        Save current instance to a gdal-raster.
        Future- add for support of more file types.
        '''
        # Check path
        if output_path.split('.')[-1] != 'tif':
            output_path = output_path + '.tif'

        # Create data source
        outraster = self.new_gdal_raster(output_path, self.shape,
                                         self.bandCount, self.dtype,
                                         self.left, self.top, self.csx,
                                         self.csy, self.projection,
                                         self.chunks, compress)

        # Add data and nodata attributes
        for band in self.bands:
            outraster.activeBand = band
            ds = outraster.ds
            band = ds.GetRasterBand(outraster.band)
            band.SetNoDataValue(self.nodata)
            del band, ds
            if self.useChunks:
                for a, s in self.iterchunks():
                    outraster[s] = a
            else:
                outraster[:] = self.array
        del outraster

    def copy_ds(self, file_suffix, template=None):
        '''
        Create a copy of the underlying dataset to use for writing
        temporarily.  Used when mode is 'r' and procesing is required,
        or self is instantiated using another raster instance.
        Defaults to HDF5 format.
        '''
        # Use another dataset as a template if specified
        if template is not None:
            self.__dict__.update(template.__dict__)

        path = self.generate_name(file_suffix, 'h5', True)

        # Create HDF5 file and datasets
        self.save(path, None, empty=True)

    @property
    def size(self):
        return (self.itemsize * self.shape[0] * self.shape[1] *
                self.bandCount / 1E9)

    @property
    def nodata(self):
        return self.nodataValues[self.band - 1]

    @property
    def ds(self):
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
            return ds
        else:
            try:
                return h5py.File(self.path, mode=self.mode)
            except:
                raise RasterError('Oops...the raster %s is now missing.' %
                                  self.path)

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
        '''**all data in memory**'''
        if self.format == 'gdal':
            ds = self.ds
            return ds.GetRasterBand(self.band).ReadAsArray()
        else:
            return self.ds[str(self.band)][:]

    @property
    def chunks(self):
        '''Chunk shape of self'''
        if self.format == 'gdal':
            ds = self.ds
            band = ds.GetRasterBand(self.band)
            chunks = band.GetBlockSize()
            # If no blocks exist, a single scanline will result- change this
            if chunks[0] == self.shape[1] and chunks[1] == 1:
                lines = int((256 * 256 * self.itemsize) /
                            (self.shape[0] * self.itemsize))
                chunks = (lines, self.shape[1])
            else:
                # Reverse to match numpy index notation
                chunks = (chunks[1], chunks[0])
        else:
            try:
                chunks = self.ds[str(self.band)].chunks
            except:
                chunks = (256, 256)
        if chunks[0] > self.shape[0]:
            chunks = (int(self.shape[0]), chunks[1])
        if chunks[1] > self.shape[1]:
            chunks = (chunks[0], int(self.shape[1]))
        return chunks

    @property
    def itemsize(self):
        '''Return number of bytes/element for a specific dtype'''
        return numpy.dtype(self.dtype).itemsize

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
        ychunks = zip(numpy.array(ychunks[:-1]) - expand,
                      numpy.array(ychunks[1:]) + expand)
        ychunks[0] = (0, ychunks[0][1])
        ychunks[-1] = (ychunks[-1][0], self.shape[0])
        xchunks = zip(numpy.array(xchunks[:-1]) - expand,
                      numpy.array(xchunks[1:]) + expand)
        xchunks[0] = (0, xchunks[0][1])
        xchunks[-1] = (xchunks[-1][0], self.shape[1])

        # Create a generator out of slices
        for ych in ychunks:
            for xch in xchunks:
                s = (slice(ych[0], ych[1]), slice(xch[0], xch[1]))
                yield self[s[0], s[1]], s

    @staticmethod
    def get_gdal_dtype(dtype):
        '''Return a gdal data type from a numpy counterpart'''
        datatypes = {
            'int8': 'Int16',
            'bool': 'Byte',
            'uint8': 'Byte',
            'int16': 'Int16',
            'int32': 'Int32',
            'uint16': 'UInt16',
            'uint32': 'UInt32',
            'int64': 'Int64',
            'uint64': 'UInt64',
            'float32': 'Float32',
            'float64': 'Float64',
            'int64': 'Long'
        }
        try:
            return gdal.GetDataTypeByName(datatypes[dtype])
        except KeyError:
            raise RasterError('Unrecognized data type "%s" encountered while'
                              ' trying to save a GDAL raster' % dtype)

    def astype(self, dtype):
        '''Change the data type of self'''
        # Check the input
        try:
            dtype = dtype.lower()
            assert dtype in ['bool', 'int8', 'uint8', 'int16', 'uint16',
                             'int32', 'uint32', 'int64', 'uint64', 'float32',
                             'float64']
        except:
            raise RasterError('Unrecognizable data type "%s"' % dtype)

        # Ensure they are already not the same
        if dtype == self.dtype:
            return self

        # Create a copy with new data type
        prvdtype = self.dtype
        self.dtype = dtype
        out = raster(self)
        self.dtype = prvdtype

        # Complete casting
        for band in self.bands:
            out.activeBand = band
            for a, s in self.iterchunks():
                out[s] = a.astype(dtype)

        # Return casted raster
        return out

    def define_projection(self, projection, overwrite=False):
        '''Define the raster projection'''
        if (self.projection != '' and
                self.projection is not None) and not overwrite:
            raise RasterError('The current raster already has a spatial'
                              ' reference, use "overwrite=True" to replace.')
        self.projection = self.parse_projection(projection)

    def match_raster(self, input_raster, tolerance=1E-09):
        '''
        Align extent and cells with another raster
        '''
        def isclose(input, values):
            values = [(val - tolerance, val + tolerance) for val in values]
            if any([lower < input < upper for lower, upper in values]):
                return True
            else:
                return False

        if isinstance(input_raster, raster):
            inrast = input_raster
        else:
            inrast = raster(input_raster)

        # Check if spatial references align
        insrs = osr.SpatialReference()
        insrs.ImportFromWkt(self.projection)
        outsrs = osr.SpatialReference()
        outsrs.ImportFromWkt(inrast.projection)
        samesrs = insrs.IsSame(outsrs)
        inrast_bbox = (inrast.top, inrast.bottom, inrast.left, inrast.right)

        # Check if cells align
        if all([isclose(self.csx, [inrast.csx]),
                isclose(self.csy, [inrast.csy]),
                isclose((self.top - inrast.top) % self.csy, [0, self.csy]),
                isclose((self.left - inrast.left) % self.csx, [0, self.csx]),
                samesrs]):
            # Simple slicing is sufficient
            self.change_extent(inrast_bbox)
        else:
            # Transform required
            print "Transforming to match rasters..."
            if samesrs:
                self.transform(csx=inrast.csx, csy=inrast.csy,
                               bbox=inrast_bbox,
                               interpolation=self.interpolation)
            else:
                self.transform(csx=inrast.csx, csy=inrast.csy,
                               projection=inrast, bbox=inrast_bbox,
                               interpolation=self.interpolation)

    def slice_from_bbox(self, bbox):
        '''
        Compute slice objects to using a bbox.

        Returns slicer for self, another for an array the size of bbox, and
        the shape of the output array from bbox.
        '''
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
        resid = (self.top - bottom) / self.csy
        bottom += (resid - int(round(resid))) * self.csy
        resid = (self.left - left) / self.csx
        left += (resid - int(round(resid))) * self.csx
        resid = (self.left - right) / self.csx
        right += (resid - int(round(resid))) * self.csx

        # Compute shape and slices
        shape = (int(round((top - bottom) / self.csy)),
                 int(round((right - left) / self.csx)))
        i = int(round((self.top - top) / self.csy))
        if i < 0:
            i = 0
            i_ = abs(i)
        else:
            i_ = 0
        j = int(round((left - self.left) / self.csx))
        if j < 0:
            j = 0
            j_ = abs(j)
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

    def change_extent(self, bbox):
        '''
        Slice self using bounding box coordinates.

        Note: the bounding box may not be honoured exactly.  To accomplish this
        use a transform.
        '''
        # Get slices
        self_slice, insert_slice, shape, bbox = self.slice_from_bbox(bbox)

        # Check if no change
        if all([self.top == bbox[0], self.bottom == bbox[1],
                self.left == bbox[2], self.right == bbox[3]]):
            return

        # Create output dataset with new extent, and transfer self to it
        path = self.generate_name('change_extent', 'h5', True)
        kwargs = {
            'projection': self.projection,
            'csx': self.csx,
            'csy': self.csy,
            'shape': shape + (self.bandCount,),
            'dtype': self.dtype,
            'top': bbox[0],
            'left': bbox[2],
            'nodata': self.nodata
        }
        outds = raster(path, mode='w', **kwargs)
        for band in outds.bands:
            outds.activeBand = band
            outds[insert_slice] = self[self_slice]
        self.__dict__.update(outds.__dict__)

    def transform(self, **kwargs):
        '''
        Change cell size, projection, or extent.
        ------------------------
        In Args

        "projection": output projection argument
            (wkt, epsg, osr.SpatialReference, raster instance)

        "csx": output cell size in the x-direction

        "cxy": output cell size in the y-direction

        "bbox": bounding box for output (top, bottom, left, right)
            Note- if projection specified, bbox must be in the output
            coordinate system
        '''
        def expand_extent(csx, csy, bbox, shape):
            '''Expand extent to snap to cell size'''
            top, bottom, left, right = bbox
            if csx is not None:
                # Expand extent to fit cell size
                resid = (right - left) - (csx * shape[1])
                if resid < 0:
                    resid /= -2.
                elif resid > 0:
                    resid = resid / (2. * csx)
                    resid = (numpy.ceil(resid) - resid) * csx
                left -= resid
                right += resid
            if csy is not None:
                # Expand extent to fit cell size
                resid = (top - bottom) - (csy * shape[0])
                if resid < 0:
                    resid /= -2.
                elif resid > 0:
                    resid = resid / (2. * csy)
                    resid = (numpy.ceil(resid) - resid) * csy
                bottom -= resid
                top += resid
            return (top, bottom, left, right)

        # Direct to input raster dataset
        if self.format != 'gdal':
            path = self.generate_name('copy', 'tif', True)
            self.save_gdal_raster(path)
            input_raster = raster(path)
            inds = input_raster.ds
        else:
            inds = self.ds

        # Get keyword args to determine what's done
        csx = kwargs.get('csx', None)
        if csx is not None:
            csx = float(csx)
        csy = kwargs.get('csy', None)
        if csy is not None:
            csy = float(csy)
        projection = self.parse_projection(kwargs.get('projection', None))
        bbox = kwargs.get('bbox', None)
        if bbox is not None:
            bbox = map(float, bbox)
            if any([bbox[0] < bbox[1],
                    bbox[1] > bbox[0],
                    bbox[3] < bbox[2],
                    bbox[2] > bbox[3]]):
                raise RasterError('Bounding box is invalid, check that the'
                                  ' coordinate positions are (top, bottom,'
                                  ' left, right)')
        if all([i is None for i in [csx, csy, projection, bbox]]):
            print ("Warning: Did not perform transform, as no arguments"
                   " provided.")
            return

        # Check for spatial reference change
        if projection is not None:
            insrs = osr.SpatialReference()
            insrs.ImportFromWkt(self.projection)
            outsrs = osr.SpatialReference()
            outsrs.ImportFromWkt(projection)
            if insrs.IsSame(outsrs):
                insrs, outsrs = None, None
        else:
            insrs, outsrs = None, None
        if all([insrs is None, outsrs is None, csx is None, csy is None,
                bbox is None]):
            return

        # Refine each of the inputs, based on each of the args
        # Projection
        if insrs is not None:
            coordTransform = osr.CoordinateTransformation(insrs, outsrs)
            left, top, _ = coordTransform.TransformPoint(self.left,
                                                         self.top)
            right, bottom, _ = coordTransform.TransformPoint(self.right,
                                                             self.bottom)
            ncsx = (right - left) / self.shape[1]
            ncsy = (top - bottom) / self.shape[0]
            # If bbox not specified, create one using new system
            if bbox is None:
                bbox = (top, bottom, left, right)
                # If cell sizes not specified, create those using new system
                if csx is None:
                    csx = ncsx
                if csy is None:
                    csy = ncsy
                # Expand extent if cell sizes not compatible
                shape = (int(round((top - bottom) / csy)),
                         int(round((right - left) / csx)))
                bbox = expand_extent(csx, csy, bbox, shape)
            else:
                # Raw bbox input is used as extent, check csx and csy
                if csx is None:
                    # Adjust new x cell size fit in bbox
                    dif = bbox[3] - bbox[2]
                    csx = dif / int(numpy.ceil(dif / ncsx))
                if csy is None:
                    # Adjust new y cell size fit in bbox
                    dif = bbox[0] - bbox[1]
                    csy = dif / int(numpy.ceil(dif / ncsy))

        # bbox
        if bbox is None:
            # Create using current extent, and expand as necessary, given cell
            #   size arguments
            shape = (int(round((self.top - self.bottom) / csy)),
                     int(round((self.right - self.left) / csx)))
            bbox = expand_extent(csx, csy, (self.top, self.bottom,
                                            self.left, self.right), shape)

        # csx
        if csx is None:
            # Set as current csx, and change to fit bbox if necessary
            dif = bbox[3] - bbox[2]
            csx = dif / int(numpy.ceil(dif / self.csx))

        # csy
        if csy is None:
            # Set as current csx, and change to fit bbox if necessary
            dif = bbox[0] - bbox[1]
            csy = dif / int(numpy.ceil(dif / self.csy))

        # Check that bbox and cell sizes align
        xresid = round((bbox[3] - bbox[2]) % csx, 9)
        yresid = round((bbox[0] - bbox[1]) % csy, 9)
        if xresid != round(csx, 9) and xresid != 0:
            raise RasterError('Transform cannot be completed due to an'
                              ' incompatible extent %s and cell size (%s) in'
                              ' the x-direction' % ((bbox[3], bbox[2]), csx))
        if yresid != round(csy, 9) and yresid != 0:
            raise RasterError('Transform cannot be completed due to an'
                              ' incompatible extent %s and cell size (%s) in'
                              ' the y-direction' % ((bbox[0], bbox[1]), csy))

        # Compute new shape
        shape = (int(round((bbox[0] - bbox[1]) / csy)),
                 int(round((bbox[3] - bbox[2]) / csx)))

        # Create output raster dataset
        if insrs is not None:
            insrs = insrs.ExportToWkt()
        if outsrs is not None:
            outsrs = outsrs.ExportToWkt()
            output_srs = outsrs
        else:
            output_srs = self.projection
        path = self.generate_name('transform', 'tif', True)
        out_raster = self.new_gdal_raster(path, shape, self.bandCount,
                                          self.dtype, bbox[2], bbox[0], csx,
                                          csy, output_srs, (256, 256))
        # Set/fill nodata value
        for band in out_raster.bands:
            ds = out_raster.ds
            band = ds.GetRasterBand(out_raster.band)
            band.SetNoDataValue(self.nodata)
            del band, ds
            for a, s in out_raster.iterchunks():
                shape = self.gdal_args_from_slice(s, self.shape)
                shape = (shape[3], shape[2])
                out_raster[s] = numpy.full(shape, self.nodata, self.dtype)

        outds = out_raster.ds
        gdal.ReprojectImage(inds, outds, insrs, outsrs, self.interpolation)
        del inds, outds

        # Change type to gdal and path to new raster
        self.format = 'gdal'
        self.path = path

        # Update attributes from new raster
        ds = gdal.Open(path)
        self.load_from_gdal(ds)

    def generate_name(self, suffix, extension, add_to_garbage=False):
        '''Generate a unique file name'''
        now = datetime.now()

        def getabsname(i):
            return ('%s_%s_%s%i.%s' %
                    (''.join(os.path.basename(self.path).split('.')[:-1])[:20],
                     str(suffix),
                     datetime.toordinal(now), i, str(extension))
                    )

        i = 1
        path = os.path.join(os.path.dirname(self.path), getabsname(i))
        while os.path.isfile(path):
            i += 1
            path = os.path.join(os.path.dirname(self.path), getabsname(i))

        # Add to garbage collector if necessary
        try:
            self.garbage.append(path)
        except:
            self.garbage = [path]

        return path

    @staticmethod
    def new_gdal_raster(output_path, shape, bands, dtype, left, top, csx, csy,
                        projection, chunks, compress=True):
        '''Generate a new gdal raster dataset'''
        driver = gdal.GetDriverByName('GTiff')
        if compress:
            comp = 'COMPRESS=LZW'
        else:
            comp = 'COMPRESS=NONE'
        # If the chunk size is the same as the raster size the TIFF driver
        #   goes bonkers, so do not tile in this case.
        if chunks[1] == shape[1] or chunks[0] == shape[0]:
            blockxsize, blockysize, tiled = ('BLOCKXSIZE=0', 'BLOCKYSIZE=0',
                                             'TILED=NO')
        else:
            blockxsize = 'BLOCKXSIZE=%s' % chunks[1]
            blockysize = 'BLOCKYSIZE=%s' % chunks[0]
            tiled = 'TILED=YES'
        parszOptions = [tiled, blockysize, blockxsize, comp]
        ds = driver.Create(output_path, shape[1], shape[0],
                           bands, raster.get_gdal_dtype(dtype),
                           parszOptions)
        if ds is None:
            raise RasterError('GDAL error trying to create new raster.')
        ds.SetGeoTransform((left, float(csx), 0, top, 0, csy * -1.))
        projection = raster.parse_projection(projection)
        if projection is None:
            projection = ''
        ds.SetProjection(projection)
        outraster = raster(ds, mode='r+')
        del ds
        return outraster

    @staticmethod
    def parse_projection(projection):
        '''Return a wkt from some argument'''
        def raise_re():
            raise RasterError('Unable to determine projection from %s' %
                              projection)
        if isinstance(projection, basestring):
            sr = osr.SpatialReference()
            sr.ImportFromWkt(projection)
            outwkt = sr.ExportToWkt()
            if outwkt == '':
                raise_re()
        elif isinstance(projection, osr.SpatialReference):
            return projection.ExportToWkt()
        elif isinstance(projection, int):
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(projection)
            outwkt = sr.ExportToWkt()
            if outwkt == '':
                raise_re()
        elif isinstance(projection, raster):
            # Return attribute from raster instance
            return projection.projection
        elif projection is None:
            return None
        else:
            raise_re()
        return outwkt

    @staticmethod
    def gdal_args_from_slice(s, shape):
        '''Supplementary to __getitem__ and __setitem__'''
        if type(s) == int:
            xoff = 0
            yoff = s
            win_xsize = shape[1]
            win_ysize = 1
        elif type(s) == tuple:
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

    def __getitem__(self, s):
        if self.format == 'gdal':
            # Tease gdal band args from s
            try:
                xoff, yoff, win_xsize, win_ysize =\
                    self.gdal_args_from_slice(s, self.shape)
            except:
                raise RasterError('Boolean and array indexing currently'
                                  ' unsupported for GDAL raster data sources.'
                                  ' Convert to HDF5 to use this'
                                  ' functionality.')
            ds = self.ds
            return ds.GetRasterBand(self.band).ReadAsArray(xoff=xoff,
                                                           yoff=yoff,
                                                           win_xsize=win_xsize,
                                                           win_ysize=win_ysize)
        else:
            return self.ds[str(self.band)][s]

    def __setitem__(self, s, a):
        if self.mode == 'r':
            raise RasterError('Dataset open as read-only.')
        if type(a) != numpy.ndarray:
            a = numpy.array(a)
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
            ds = self.ds
            ds.GetRasterBand(self.band).WriteArray(a, xoff=xoff, yoff=yoff)
        else:
            try:
                self.ds[str(self.band)][s] = a
            except:
                raise RasterError('Error writing raster data. Check that mode'
                                  ' is "r+" and that the arrays match.')

    def perform_operation(self, r, op):
        try:
            if all([isinstance(x, numbers.Number)
                    for x in (0, 0.0, 0j, decimal.Decimal(r))]):
                num = True
        except:
            if not isinstance(r, raster):
                raise RasterError('Expected a number or raster instance while'
                                  ' using the "%s" operator' % op)
            num = False
        out = r
        out.match_raster(self)
        outnd = out.nodata
        nd = self.nodata
        if num:
            # It's a scalar
            if self.useChunks:
                # Compute over chunks
                for a, s in out.iterchunks():
                    if self.aMethod == 'ne':
                        out[s] = ne.evaluate('where(a!=nd,a%sr,outnd)' % op)
                    elif self.aMethod == 'np':
                        m = a != nd
                        b = a[m]
                        a[m] = eval('b%sr' % op)
                        a[~m] = outnd
                        out[s] = a
            else:
                # Load into memory, then compute
                a = self.array
                if self.aMethod == 'ne':
                    out[:] = ne.evaluate('where(a!=nd,a%sr,outnd)' % op)
                elif self.aMethod == 'np':
                    m = a != nd
                    b = a[m]
                    a[m] = eval('b%sr' % op)
                    a[~m] = outnd
                    out[:] = a
        else:
            # It's another raster
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
        if self.mode == 'r':
            raise RasterError('%s open as read-only.' %
                              os.path.basename(self.path))
        try:
            if all([isinstance(x, numbers.Number)
                    for x in (0, 0.0, 0j, decimal.Decimal(r))]):
                nd = self.nodata
                if self.useChunks:
                    for a, s in self.iterchunks():
                        if self.aMethod == 'ne':
                            self[s] = ne.evaluate('where(a!=nd,a%sr,nd)' % op)
                        elif self.aMethod == 'np':
                            m = a != nd
                            b = a[m]
                            a[m] = eval('b%sr' % op)
                            self[s] = a
                else:
                    a = self.array
                    if self.aMethod == 'ne':
                        self[:] = ne.evaluate('where(a!=nd,a%sr,nd)' % op)
                    elif self.aMethod == 'np':
                        m = a != nd
                        b = a[m]
                        a[m] = eval('b%sr' % op)
                        self[:] = a
        except:
            if not isinstance(r, raster):
                raise RasterError('Expected a number or raster instance while'
                                  ' using the "%s=" operator')
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
        if not isinstance(r, raster):
            raise RasterError('Expected a raster instance while using the "=="'
                              ' operator')
        # Compare spatial reference
        insrs = osr.SpatialReference()
        insrs.ImportFromWkt(self.projection)
        outsrs = osr.SpatialReference()
        outsrs.ImportFromWkt(r.projection)
        samesrs = insrs.IsSame(outsrs)

        # Compare spatial dimensions and data
        if all([self.csx == r.csx, self.csy == r.csy,
               self.top == r.top, self.bottom == r.bottom,
               self.left == r.left, self.right == r.right,
               self.dtype == r.dtype, self.bandCount == r.bandCount, samesrs]):
            # Compare data
            for band in self.bands:
                r.activeBand = band
                for a, s in self.iterchunks():
                    if not numpy.all(a == r[s]):
                        return False
            return True
        else:
            return False

    def __ne__(self, r):
        if self == r:
            return False
        else:
            return True

    def __repr__(self):
        insr = osr.SpatialReference(wkt=self.projection)
        methods = {
            'ne': 'numexpr (Parallel, CPU cache-optimized)',
            'np': 'numpy (single-threaded, vectorized)'
        }
        return ('A happy raster named %s of house %s\n'
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
                '' % (os.path.basename(self.path), self.format.upper(),
                      self.bandCount, self.shape[0], self.shape[1], self.csx,
                      self.csy, (self.top, self.bottom, self.left, self.right),
                      self.dtype, round(self.size, 3),
                      insr.GetAttrValue('projcs'), insr.GetAttrValue('datum'),
                      self.activeBand, self.interpolationMethod,
                      self.useChunks, methods[self.aMethod]))

    def __del__(self):
        if hasattr(self, 'garbage'):
            for f in self.garbage:
                try:
                    os.remove(f)
                except Exception as e:
                    print "Could not clean up file %s because %s" % (f, e)
