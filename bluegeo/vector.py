"""
Custom vector analysis library
"""
from osgeo import ogr, osr
from contextlib import contextmanager
import os
import shutil
import numpy
from util import parse_projection, generate_name


class VectorError(Exception):
    pass


class extent(object):
    def __init__(self, data):
        """
        Extent to be used to control geometries
        :param data: iterable of (top, bottom, left, right) or vector class instance
        """
        if any(isinstance(data, o) for o in [tuple, list, numpy.ndarray]):
            self.bounds = data
            self.geo = None
        elif isinstance(data, vector):
            self.bounds = (data.top, data.bottom, data.left, data.right)
            self.geo = data
        else:
            raise VectorError('Unsupported extent argument of type {}'.format(type(data).__name__))


class vector(object):

    def __init__(self, data, mode='r', fields=[]):
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
        if isinstance(data, basestring):
            if os.path.isfile(data):
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
                    self.path = data
            else:
                # Try to open as wkt
                # TODO: Implement wkt reader
                pass

        elif isinstance(data, ogr.DataSource):
            # Instantiate as ogr instance
            _data = data
            self.path = None

        # Collect meta
        layer = _data.GetLayer()
        layerDefn = layer.GetLayerDefn()

        # Spatial Reference
        insr = layer.GetSpatialRef()
        if insr is not None:
            self.projection = str(insr.ExportToWkt())
        else:
            self.projection = ''

        # Extent
        self.left, self.right, self.bottom, self.top = layer.GetExtent()

        # Geometry type
        self.geometryType = str(self.geom_wkb_to_name(layer.GetGeomType()))

        # Feature count
        self.featureCount = layer.GetFeatureCount()

        # Fields
        self.fieldCount = layerDefn.GetFieldCount()
        self.fieldTypes = []
        for i in range(self.fieldCount):
            fieldDefn = layerDefn.GetFieldDefn(i)
            name = fieldDefn.GetName()
            width = fieldDefn.GetWidth()
            dtype = self.ogr_dtype_to_numpy(fieldDefn.GetFieldTypeName(fieldDefn.GetType()),
                                            name, width)
            precision = fieldDefn.GetPrecision()
            self.fieldTypes.append((name, dtype, width, precision))
        self.fieldNames = [meta[0] for meta in self.fieldTypes]

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
            for name, dtype, width, precision in newFields:
                # Create field definition
                fieldDefn = ogr.FieldDefn(name, self.numpy_dtype_to_ogr(dtype))
                fieldDefn.SetWidth(width)
                fieldDefn.SetPrecision(precision)
                # Create field
                outlyr.CreateField(fieldDefn)

            with self.layer() as inlyr:
                # Output layer definition
                outLyrDefn = outlyr.GetLayerDefn()
                fields = {newField[0]: self.field_to_pyobj(self[oldField[0]])
                          for oldField, newField in zip(self.fieldTypes, newFields)}

                # Iterate and population geo's
                for i in range(self.featureCount):
                    # Gather and transform geometry
                    inFeat = inlyr.GetFeature(i)
                    geo = inFeat.GetGeometryRef()

                    # Write output fields and geometries
                    outFeat = ogr.Feature(outLyrDefn)
                    for name, dtype, width, precision in newFields:
                        outFeat.SetField(name, fields[name][i])
                    outFeat.SetGeometry(geo)

                    # Add to output layer and clean up
                    outlyr.CreateFeature(outFeat)
                    outFeat.Destroy()

            ds.Destroy()

    def empty(self, spatial_reference=None, fields=[], prestr='copy'):
        """
        Create a copy of self as an output shp without features
        :return: Fresh vector instance
        """
        # Create an output file path
        outPath = generate_name(self.path, prestr, 'shp')
        # Check that fields are less than 10 chars
        fields = self.check_fields(fields)

        # Generate output projection
        outsr = osr.SpatialReference()
        if spatial_reference is not None:
            outsr.ImportFromWkt(parse_projection(spatial_reference))
        else:
            outsr.ImportFromWkt(self.projection)

        # Generate file and layer
        driver = ogr.GetDriverByName(self.get_driver_by_path(outPath))
        ds = driver.CreateDataSource(outPath)
        layer = ds.CreateLayer('bluegeo vector', outsr, self.geometry_type)

        # Add fields
        for name, dtype, width, precision in fields:
            # Create field definition
            fieldDefn = ogr.FieldDefn(name, self.numpy_dtype_to_ogr(dtype))
            fieldDefn.SetWidth(width)
            fieldDefn.SetPrecision(precision)
            # Create field
            layer.CreateField(fieldDefn)

        # Clean up and return vector instance
        del layer
        ds.Destroy()
        return vector(outPath, mode='r+')

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
                fields = {newField[0]: self.field_to_pyobj(self[oldField[0]])
                          for oldField, newField in zip(self.fieldTypes, newFields)}

                # Iterate geometries and populate output with transformed geo's
                for i in range(self.featureCount):
                    # Gather and transform geometry
                    inFeat = inlyr.GetFeature(i)
                    geo = inFeat.GetGeometryRef()
                    geo.Transform(coordTransform)

                    # Write output fields and geometries
                    outFeat = ogr.Feature(outLyrDefn)
                    for name, dtype, width, precision in newFields:
                        outFeat.SetField(name, fields[name][i])
                    outFeat.SetGeometry(geo)

                    # Add to output layer and clean up
                    outlyr.CreateFeature(outFeat)
                    outFeat.Destroy()

        return vector(outVect.path)

    @staticmethod
    def check_fields(fields):
        """
        Ensure fields have names that are less than 10 characters
        :param fields: input field types argument (list of tuples)
        :return: new field list
        """
        outputFields = []
        cursor = []
        for name, dtype, width, precision in fields:
            name = name[:10]
            i = 0
            while name in cursor:
                i += 1
                name = name[:10-len(str(r))] + str(i)
            cursor.append(name)
            outputFields.append((name, dtype, width, precision))
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
            get_next(ogr.CreateGeometryFromWkb(wkb), vertices)

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
                    data.append(geo.ExportToWkb())
                    feature.Destroy()

            elif isinstance(item, basestring):
                # Check that the field exists
                try:
                    _, dtype, _, _ = [ft for ft in self.fieldTypes if ft[0] == str(item)][0]
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

    @property
    def filenames(self):
        """
        Return list of files associated with self
        :return: list of file paths
        """
        if self.driver == 'ESRI Shapefile':
            prestr = '.'.join(os.path.basename(self.path).split('.')[:-1])
            d = os.path.dirname(self.path)
            return [os.path.join(d, f) for f in os.listdir(d)
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
    def field_to_pyobj(a):
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
                  's': str
        }

        dtype = a.dtype.name
        if dtype[0].lower() == 's':
            dtype = 's'

        return map(_types[dtype], a)

    @property
    def drivers(self):
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

    def get_driver_by_path(self, path):
        """
        Return a supported ogr (or internal) driver using a file extension
        :param path: File path
        :return: driver name
        """
        ext = path.split('.')[-1].lower()
        if len(ext) == 0:
            raise VectorError('File path does not have an extension')

        try:
            return self.drivers[ext]
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
    def ogr_dtype_to_numpy(field_type, field_name, width):
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
