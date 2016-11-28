'''
Blue Geosimulation, 2016
Raster data interfacing and manipulation library
'''

import os
from datetime import datetime
import numpy
from osgeo import gdal, osr, gdalconst
import h5py


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
    using standard __getitem__ syntax, or by iterating
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
                self.save(data, kwargs)
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
        self.dtype, self.size =\
            self.get_data_specs(gdal.GetDataTypeName(band.DataType),
                                self.shape)
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

    def save(self, output_path, compression='lzf', **kwargs):
        '''
        Save to a new file, and use with self
        '''
        # Check compression
        if compression not in [None, 'lzf', 'gzip', 'szip']:
            raise RasterError('Unrecognized compression "%s"' % compression)

        # Check path
        if output_path.split('.')[-1] != 'h5':
            output_path = output_path + '.h5'

        # Change mode to read/write
        self.mode = 'r+'

        # Create new HDF5 file
        newfile = h5py.File(output_path, mode='w')

        # Copy data from data source to new file
        # TODO use kwargs to make empty raster
        prvb = self.activeBand
        for band in self.bands:
            self.activeBand = band
            ds = newfile.create_dataset(str(band), self.shape,
                                        dtype=self.dtype,
                                        compression=compression,
                                        chunks=True)
            for a, s in self.iterchunks(ds.chunks):
                ds[s] = a
            del ds
        del newfile
        self.activeBand = prvb
        self.format = 'HDF5'
        self.path = output_path

        # Add attributes
        ds = self.ds
        for key, val in self.__dict__.iteritems():
            if val is None:
                val = 'None'
            ds.attrs[key] = val

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

        # Create output dataset
        now = datetime.now()

        def getabsname(i):
            return '%s_%s_%s_%i.h5' % (os.path.basename(self.path),
                                       str(file_suffix),
                                       datetime.toordinal(now), i)
        i = 1
        path = os.path.join(os.path.dirname(self.path), getabsname(i))
        while os.path.isfile(path):
            i += 1
            path = os.path.join(os.path.dirname(self.path), getabsname(i))

        # Create HDF5 file and datasets
        self.save(path, None)

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
            return gdal.Open(self.path, mode)
        else:
            return h5py.File(self.path, mode=self.mode)

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
            yield i

    @property
    def array(self):
        '''**all data in memory**'''
        if self.format == 'gdal':
            ds = self.ds
            return ds.GetRasterBand(self.band).ReadAsArray()
        else:
            return self.ds[str(self.band)][:]

    def iterchunks(self, custom_chunks=None):
        if custom_chunks is None:
            # Regenerate chunk slices in case data have changed
            if self.format == 'gdal':
                ds = self.ds
                band = ds.GetRasterBand(self.band)
                chunks = band.GetBlockSize()
                # Reverse to match numpy index notation
                chunks = (chunks[1], chunks[0])
            else:
                chunks = self.ds[str(self.band)].chunks
        else:
            try:
                chunks = map(int, custom_chunks)
                assert len(custom_chunks) == 2
            except:
                raise RasterError('Custom chunks must be a tuple or list of'
                                  ' length 2 containing integers')
        ychunks = range(0, self.shape[0], chunks[0]) + [self.shape[0]]
        xchunks = range(0, self.shape[1], chunks[1]) + [self.shape[1]]
        ychunks = zip(ychunks[:-1], ychunks[1:])
        xchunks = zip(xchunks[:-1], xchunks[1:])
        # Create a generator out of slices
        for ych in ychunks:
            for xch in xchunks:
                s = (slice(ych[0], ych[1]), slice(xch[0], xch[1]))
                yield self[s[0], s[1]], s

    def __getitem__(self, s):
        if self.format == 'gdal':
            # Tease gdal band args from s
            if type(s) == int:
                xoff = 0
                yoff = s
                win_xsize = self.shape[1]
                win_ysize = 1
            elif type(s) == tuple:
                if type(s[0]) == int:
                    yoff = s[0]
                    win_ysize = 1
                else:
                    yoff = s[0].start
                    start = yoff
                    if start is None:
                        start = 0
                        yoff = 0
                    stop = s[0].stop
                    if stop is None:
                        stop = self.shape[0]
                    win_ysize = stop - start
                if type(s[1]) == int:
                    xoff = s[1]
                    win_xsize = 1
                else:
                    xoff = s[1].start
                    start = xoff
                    if start is None:
                        start = 0
                        xoff = 0
                    stop = s[1].stop
                    if stop is None:
                        stop = self.shape[1]
                    win_xsize = stop - start
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
        a = numpy.array(a)
        if self.format == 'gdal':
            # Tease gdal band args from s
            if type(s) == int:
                xoff = 0
                yoff = s
                win_xsize = self.shape[1]
                win_ysize = 1
            elif type(s) == tuple:
                if type(s[0]) == int:
                    yoff = s[0]
                    win_ysize = 1
                else:
                    yoff = s[0].start
                    start = s[0].start
                    if start is None:
                        start = 0
                        yoff = 0
                    stop = s[0].stop
                    if stop is None:
                        stop = self.shape[0]
                    win_ysize = stop - start
                if type(s[1]) == int:
                    xoff = s[1]
                    win_xsize = 1
                else:
                    xoff = s[1].start
                    start = xoff
                    if start is None:
                        start = 0
                        xoff = 0
                    stop = s[1].stop
                    if stop is None:
                        stop = self.shape[1]
                    win_xsize = stop - start
            if (a.size > 1 and
                    (win_ysize != a.shape[0] or win_xsize != a.shape[1])):
                raise RasterError('Raster data of the shape %s cannot be'
                                  ' replaced with array of shape %s' %
                                  ((win_ysize, win_xsize), a.shape))
            if a.size == 1:
                a = numpy.full((win_ysize, win_xsize), a, a.dtype)
            ds = self.ds
            ds.GetRasterBand(self.band).WriteArray(a, xoff=xoff, yoff=yoff)
        else:
            try:
                self.ds[str(self.band)][s] = a
            except:
                raise RasterError('Dataset open as read-only.')

    @staticmethod
    def get_data_specs(dtype, shape):
        if dtype.lower() == 'uint8':
            return 'uint8', (float(shape[0]) * shape[1]) / 1E9
        elif dtype.lower() == 'bool':
            return 'bool', (float(shape[0]) * shape[1]) / 1E9 / 8
        elif dtype.lower() in ['byte', 'int8']:
            return 'int8', (float(shape[0]) * shape[1]) / 1E9
        elif dtype.lower() in ['uint16', 'int16']:
            return dtype.lower(), (2. * float(shape[0]) * shape[1]) / 1E9
        elif dtype.lower() in ['uint32', 'int32', 'float32']:
            return dtype.lower(), (4. * float(shape[0]) * shape[1]) / 1E9
        elif dtype.lower() in ['float64', 'int64', 'uint64']:
            return dtype.lower(), (8. * float(shape[0]) * shape[1]) / 1E9
        else:
            raise RasterError('Cannot read the input raster data type "%s"' %
                              dtype)

    def astype(self, dtype):
        '''Change the data type of self'''
        try:
            dtype = dtype.lower()
            assert dtype in ['bool', 'int8', 'uint8', 'int16', 'uint16',
                             'int32', 'uint32', 'int64', 'uint64', 'float32',
                             'float64']
        except:
            raise RasterError('Unrecognizable data type "%s"' % dtype)
        self.dtype = dtype
        if self.format == 'gdal' or self.mode == 'r':
            # Need to create a copy
            self.copy_ds(dtype)
        else:
            # Change current file
            ds = self.ds
            for band in self.bands:
                self.activeBand = band
                newband = ds.create_dataset(str(band) + '_',
                                            dtype=self.dtype,
                                            shape=self.shape,
                                            chunks=True)
                for a, s in self.iterchunks():
                    newband[s] = a
                del newband
                del ds[str(band)]
                ds[str(band)] = ds[str(band) + '_']
                del ds[str(band) + '_']
