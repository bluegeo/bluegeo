'''
Blue Geosimulation, 2016
Raster data interfacing and manipulation library
'''

import os
from datetime import datetime
import numpy
from osgeo import gdal, osr
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
    def __init__(self, data, mode='r'):
        # Record mode
        if mode in ['r', 'r+', 'w']:
            self.mode = mode
        else:
            raise RasterError('Unrecognized file mode "%s"' % mode)

        # Check if data is a string
        if isinstance(data, basestring):
            # If in 'w' mode, write a new file
            if self.mode == 'w':
                self.save(data)
            # Check if data is a valid file
            elif not os.path.isfile(data):
                raise RasterError('%s is not a file' % data)
            else:
                # Try for a gdal dataset
                ds = gdal.Open(data)
                try:
                    gt = ds.GetGeoTransform()
                    assert gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                    self.load_from_gdal(ds)
                    self.format = 'gdal'
                except:
                    # Try an HDF5 data source
                    try:
                        ds = h5py.File(data)
                        self.load_from_hdf5(ds)
                        self.format = 'HDF5'
                    except:
                        raise RasterError('Unable to open dataset %s' % data)
        # If not a string, maybe an osgeo dataset?
        elif isinstance(data, gdal.Dataset):
            self.load_from_gdal(ds)
            self.format = 'gdal'
        # ...or an h5py dataset
        elif isinstance(data, h5py.File):
            self.load_from_hdf5(data)
            self.format = 'HDF5'
        # ...or a raster instance
        elif isinstance(data, raster):
            # Create a copy of the instance, except for dataset
            self.copy_ds('copy')

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

    def load_from_HDF5(self, ds):
        '''Load attributes from an HDF5 file'''
        d = {}
        for key, val in ds.attrs.iteritems():
            if not isinstance(val, numpy.ndarray) and val == 'None':
                val = None
            d[key] = val
        self.__dict__.update(d)
        self.path = ds.name

    def save(self, output_path, compression='lzf'):
        '''
        Save to a new file, and use with self
        '''
        # Check compression
        if compression not in [None, 'lzf', 'gzip', 'szip']:
            raise RasterError('Unrecognized compression "%s"' % compression)

        # Check path
        if output_path.split('.')[-1] != 'h5':
            output_path = output_path + '.h5'
        self.path = output_path

        # Change mode to read/write
        self.mode = 'r+'

        # Create new HDF5 file
        newfile = h5py.File(output_path, 'w')

        # Copy data from data source to new file
        prvb = self.activeBand
        for band in self.bands:
            self.activeBand = band
            newfile.create_dataset(str(band), self.shape, dtype=self.dtype,
                                   compression=compression)
            for a, s in self.iterchunks:
                self[s] = a
        del newfile
        self.activeBand = prvb
        self.format = 'HDF5'

        # Add attributes
        for key, val in self.__dict__:
            if val is None:
                val = 'None'
            self.ds.attrs[key] = val

    def copy_ds(self, file_suffix):
        '''
        Create a copy of the underlying dataset to use for writing
        if the current mode is 'r', or self is instantiated using
        another raster instance.  Defaults to HDF5 format.
        '''
        # Create output dataset
        now = datetime.now()
        absname = '%s_%s_%s.h5' % (os.path.basename(self.path),
                                   str(file_suffix),
                                   datetime.toordinal(now))
        path = os.path.join(os.path.dirname(self.path), absname)

        # Create HDF5 file and datasets
        self.save(self.path, None)

        # Add temporary file creations to garbage collector
        if hasattr(self, garbage):
            self.garbage.append(self.path)
        else:
            self.garbage = [self.path]

    @property
    def nodata(self):
        return self.nodataValues[self.activeBand - 1]

    @property
    def ds(self):
        if self.format == 'gdal':
            return gdal.Open(self.path)
        else:
            return h5py.File(self.path)

    @property
    def band(self):
        try:
            b = int(self.activeBand)
            assert self.activeBand <= self.bandCount and self.activeBand > 0
        except:
            raise RasterError('Active band "%s" cannot be accessed' %
                              self.activeBand)
        return self.activeBand

    @property
    def bands(self):
        for i in range(1, self.bandCount + 1):
            yield i

    @property
    def array(self):
        '''**all data in memory**'''
        if self.format == 'gdal':
            band = self.ds.GetRasterBand(self.band)
            return band.ReadAsArray()
        else:
            return self.band[:]

    def iterchunks(self, custom_chunks=None):
        if custom_chunks is None:
            # Regenerate chunk slices in case data have changed
            if self.format == 'gdal':
                band = self.ds.GetRasterBand(self.band)
                chunks = band.GetBlockSize()
            else:
                chunks = self.ds.chunks
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
                yield (self[s[0], s[1]], s)

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
                    win_ysize = s[0].stop - s[0].start
                if type(s[1]) == int:
                    xoff = s[1]
                    win_xsize = 1
                else:
                    xoff = s[1].start
                    win_xsize = s[1].stop - s[1].start
            band = self.ds.GetRasterBand(self.band)
            return band.ReadAsArray(xoff=xoff, yoff=yoff,
                                    win_xsize=win_xsize,
                                    win_ysize=win_ysize)
        else:
            return self.band[s]

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
                    win_ysize = s[0].stop - s[0].start
                if type(s[1]) == int:
                    xoff = s[1]
                    win_xsize = 1
                else:
                    xoff = s[1].start
                    win_xsize = s[1].stop - s[1].start
            if ((win_ysize != a.shape[0] or win_xsize != a.shape[1]) and
                    a.size > 1):
                raise RasterError('Raster data of the shape %s cannot be'
                                  ' replaced with array of shape %s' %
                                  ((win_ysize, win_xsize), a.shape))
            if a.size == 1:
                a = numpy.full((win_ysize, win_xsize), a, a.dtype)
            band = self.ds.GetRasterBand(self.band)
            band.WriteArray(a, xoff=xoff, yoff=yoff)
        else:
            self.band[s] = a

    def __del__(self):
        # Collect garbage
        if hasattr(self, 'garbage'):
            os.remove(self.garbage)

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
