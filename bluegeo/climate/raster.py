"""
Raster abstraction
"""
from __future__ import print_function
from contextlib import contextmanager
import dask.array as da
import numpy as np
from osgeo import gdal, osr


class Dasker(object):
    def __init__(self, path):
        self.path = path
        self.__dict__.update(get_raster_specs(path))

    def __getitem__(self, s):
        """
        Collect raster data as a numpy array
        Data are pulled from the raster source directly with the slice

        :param s: getter object
        :return: numpy.ndarray
        """
        if s == (slice(0, 0, None), slice(0, 0, None)):
            # Dask calls with empty slices when using from_array
            return np.array([]).reshape((0, 0))

        # Collect a compatible slice
        i_start, i_stop, j_start, j_stop = self.parse_slice(s)

        if i_start > self.shape[0] - 1:
            raise IndexError('Index {} out of bounds for axis 0 with shape {}'.format(
                i_start, self.shape[0]))
        if j_start > self.shape[1] - 1:
            raise IndexError('Index {} out of bounds for axis 1 with shape {}'.format(
                j_start, self.shape[1]))

        # Allocate output
        shape = (i_stop - i_start, j_stop - j_start)
        out_array = np.empty(shape, self.dtype)

        with open_gdal(self.path) as ds:
            out_array[:] = ds.GetRasterBand(1).ReadAsArray(
                xoff=j_start, yoff=i_start, win_xsize=shape[1], win_ysize=shape[0]
                )

        return out_array

    def parse_slice(self, s):
        """
        Slices can:
            - Be a `slice` object
            - Be None
            - Be an Integer
            - vary from 0 to 3 in length

        :param s:
        :return:
        """
        if not hasattr(s, '__iter__'):
            s = [s]
        if len(s) > 3:
            raise IndexError(
                'Rasters must be indexed in a maximum of 3 dimensions')

        def check_index(item, i, start=True):
            if (item >= self.shape[i] and start) or (item > self.shape[i] and not start):
                raise IndexError('Index {} out for range for dimension {} of size {}'.format(
                    item, i, self.shape[i])
                    )

        def get_slice_item(item, i):
            if isinstance(item, int):
                check_index(item, i)
                return item, item + 1
            elif isinstance(item, slice):
                start = item.start or 0
                check_index(start, i)
                stop = item.stop or self.shape[i]
                check_index(stop, i, False)
                return start, stop
            elif item is None:
                return 0, self.shape[i]
            else:
                raise IndexError(
                    'Unsupported slice format {}'.format(type(item).__name__))

        slicers = ()
        for i, item in enumerate(s):
            slicers += get_slice_item(item, i)

        return slicers


@contextmanager
def open_gdal(source):
    ds = gdal.Open(source)
    if ds is None:
        raise ValueError('Unable to open the raster "{}"'.format(source))
    yield ds
    ds = None


def get_raster_specs(path):
    with open_gdal(path) as ds:
        # Populate the spatial reference using the input raster
        gt = ds.GetGeoTransform()
        csy = float(abs(gt[5]))
        csx = float(gt[1])
        top = float(gt[3])
        left = float(gt[0])
        shape = (ds.RasterYSize, ds.RasterXSize)
        bottom = top - (csy * shape[0])
        right = left + (csx * shape[1])
        dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
        if dtype.lower() == 'byte':
            dtype = 'uint8'
        return dict(
            projection=ds.GetProjectionRef(),
            left=left,
            csx=csx,
            top=top,
            csy=csy,
            shape=shape,
            bottom=bottom,
            right=right,
            nodata=ds.GetRasterBand(1).GetNoDataValue(),
            bbox=(left, bottom, right, top),
            dtype=dtype,
            ndim=2
            )


def read_array(path, bbox=None, bbox_projection=None):
    """Read a raster into a numpy array, which clipping to the bounding box

    Arguments:
        path {str} -- Input raster path
        bbox {iterable} -- bounding box in the form xmin, ymin, xmax, ymax
        bbox_projection {str} -- wkt
    """
    spec = get_raster_specs(path)

    with open_gdal(path) as ds:
        if bbox is not None:
            if bbox_projection is None:
                raise ValueError('A bbox projection is required with a bbox')
            insr = osr.SpatialReference()
            outsr = osr.SpatialReference()
            insr.ImportFromWkt(bbox_projection)
            outsr.ImportFromWkt(spec['projection'])
            trans = osr.CoordinateTransformation(insr, outsr)
            # Coordinates in a GCS are returned as (lat, long)
            ymax, xmin, _ = trans.TransformPoint(bbox[0], bbox[3])
            ymin, xmax, _ = trans.TransformPoint(bbox[2], bbox[1])

            # Expand to snap to the input
            # top
            span = (spec['top'] - ymax) / spec['csy']
            resid = (span - int(span)) * spec['csy']
            ymax += resid if resid >= 0 else spec['csy'] + resid
            # bottom
            span = (ymin - spec['bottom']) / spec['csy']
            resid = (span - int(span)) * spec['csy']
            ymin -= resid if resid >= 0 else spec['csy'] + resid
            # left
            span = (xmin - spec['left']) / spec['csx']
            resid = (span - int(span)) * spec['csx']
            xmin -= resid if resid >= 0 else spec['csx'] + resid
            # right
            span = (spec['right'] - xmax) / spec['csx']
            resid = (span - int(span)) * spec['csx']
            xmax += resid if resid >= 0 else spec['csx'] + resid

            # Read the array
            out_ds = gdal.Warp('mem', path, outputBounds=(xmin, ymin, xmax, ymax), format='MEM')
            if out_ds is None:
                raise RuntimeError(
                    'There was an error reading an array from {}. Check the spatial reference'.format(path)
                    )
            a = out_ds.ReadAsArray()

            # Return the bounding box and cell sizes for comparison to the input
            gt = out_ds.GetGeoTransform()
            csy = float(abs(gt[5]))
            csx = float(gt[1])
            nodata = out_ds.GetRasterBand(1).GetNoDataValue()
            out_ds = None
            return np.ma.masked_equal(a, nodata), (xmin, ymin, xmax, ymax), csx, csy
        else:
            with open_gdal(path) as ds:
                a = ds.ReadAsArray()
            return np.ma.masked_equal(a, spec['nodata'])


def dask_array(path, chunks):
    """Load a raster as a dask array"""
    d = Dasker(path)
    return da.ma.masked_equal(da.from_array(d, chunks=chunks), d.nodata)


def save_array(path, a, nodata, top, left, csx, csy, projection, format="GTiff"):
    if a.ndim != 2:
        raise ValueError('Expected a 2-d array for writing')

    if path.split('.')[-1].lower() != 'tif':
        path += '.tif'

    driver = gdal.GetDriverByName(format)

    if format == "GTiff":
        comp = 'COMPRESS=LZW'
        if a.shape[0] > 256 and a.shape[1] > 256:
            blockxsize = 'BLOCKXSIZE=%s' % 256
            blockysize = 'BLOCKYSIZE=%s' % 256
            tiled = 'TILED=YES'
        else:
            blockxsize, blockysize, tiled = 'BLOCKXSIZE=0', 'BLOCKYSIZE=0', 'TILED=NO'
        parszOptions = [tiled, blockysize, blockxsize, comp]
    else:
        parszOptions = []

    dtype = 'Byte' if a.dtype.name in [
        'uint8', 'int8', 'bool'] else a.dtype.name
    dtype = 'int32' if dtype == 'int64' else dtype
    dtype = 'uint32' if dtype == 'uint64' else dtype
    dst = driver.Create(path, a.shape[1], a.shape[0], 1, gdal.GetDataTypeByName(dtype), parszOptions)

    if dst is None:
        raise IOError('Unable to save the file {}'.format(path))

    dst.SetProjection(projection)
    if csy > 0:
        csy = -csy
    dst.SetGeoTransform((left, csx, 0, top, 0, csy))
    band = dst.GetRasterBand(1)
    # pylint: disable=no-member
    band.SetNoDataValue(nodata)

    band.WriteArray(a)
    dst.FlushCache()
    band = None

    if format == 'MEM':
        return dst

    dst = None


def bilinear(in_raster, spec_raster, output=None):
    """Resample an input grid to the specs of the input raster using bilinear interpolation

    Arguments:
        in_raster {str} -- Input raster to resample
        spec_raster {str} -- Raster specifications for the output

    Keyword Arguments:
        output {str} -- Output raster path (default: {None}) If not specified, an array will be returned
    """
    if output is None:
        out_format = 'MEM'
        out_path = 'mem'
    else:
        out_format = 'GTiff'
        out_path = output

    with open_gdal(spec_raster) as spec:
        gt = spec.GetGeoTransform()
        csx = gt[1]
        xmin = gt[0]
        xmax = xmin + (csx * spec.RasterXSize)
        csy = abs(gt[5])
        ymax = gt[3]
        ymin = ymax - (csy * spec.RasterYSize)

        parsz_options = []
        if format == 'GTiff':
            comp = 'COMPRESS=LZW'
            if spec.RasterXSize > 256 and spec.RasterYSize > 256:
                blockxsize = 'BLOCKXSIZE=%s' % 256
                blockysize = 'BLOCKYSIZE=%s' % 256
                tiled = 'TILED=YES'
            else:
                blockxsize, blockysize, tiled = 'BLOCKXSIZE=0', 'BLOCKYSIZE=0', 'TILED=NO'
            parsz_options = [tiled, blockysize, blockxsize, comp]

        warp_options = {
            'dstSRS': spec.GetProjectionRef(),
            'outputBounds': (xmin, ymin, xmax, ymax),
            'format': out_format,
            'xRes': csx,
            'yRes': csy,
            'creationOptions': parsz_options,
            'resampleAlg': 'bilinear'
            }

        safety = 0
        while True:
            if safety == 5:
                raise SyntaxError('Unable to warp dataset')
            try:
                ds = gdal.Warp(out_path, in_raster, **warp_options)
                break
            except SystemError:
                safety += 1

    if output is None:
        a = ds.ReadAsArray()
        ds = None
        return a
