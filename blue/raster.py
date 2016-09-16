import numpy
from ast import literal_eval
import tempfile
import os
import shutil
import multiprocessing

try:
    from osgeo.gdalconst import *
    from osgeo import gdal
    from osgeo import osr
    datadriver = 'GDAL'
except ImportError:
    try:
        import arcpy
        datadriver = 'ESRI'
    except ImportError:
        datadriver = None
        print("Warning! Only .brf raster types will be able to be read due"
              " a lack of GDAL and ESRI drivers")


class raster:
    '''
    Raster interfacing class
    Data can be a GDAL/ESRI supported file, a blue raster format file, an
    osgeo.gdal.Dataset instance, an arcpy.Raster instance, or another raster
    instance (basically a way to create a copy).

    Data are first read virtually (access all attribtues). If a raster file is
    the source, the .read() method writes an on-disk memory mapped array that
    can be accessed using the .load() method. Data are not loaded into
    memory beyond the tilesize argument (GB), they are a view into an on-disk
    numpy array.
    TODO: Fix project truncation at right
    TODO: Check on tile edges
    TODO: Add band support to memmap array (use z-index)
    '''

    def __init__(self, data, tilesize=0.5, parallelize=True, tmpspace=None):
        datapath = data
        data = self.parseData(data)
        self.tilesize = tilesize
        self.parallelize = parallelize
        if tmpspace is not None:
            if not os.path.isdir(tmpspace):
                raise Exception('The temporary workspace "%s" does not exist' %
                                tmpspace)
        self.tmpspace = tmpspace
        if not data:
            raise Exception('Cannot open the input raster data')
        if data[0] == 'blueraster':
            # Create a copy
            self.__dict__.update(dict(data[1].__dict__))
            if hasattr(self, 'array'):
                try:
                    del self.tempfiles
                except:
                    pass
                a = self.writeFile()
                shutil.copy(self.array, a)
                self.array = a
            self.createTileDefn()
        elif data[0] == 'gdalraster':
            if datadriver == 'GDAL':
                self.populateFromGDALRaster(data[1])
            elif datadriver == 'ESRI':
                self.populateFromESRIRaster(data[1])
        elif data[0] == 'brf':
            self.__dict__.update(data[1])
            self.array = datapath
        elif data[0] == 'arcpyraster':
            self.populateFromESRIRaster(data[1])
        self.filetype = data[0]

    def populateFromGDALRaster(self, ds):
        self.projection = ds.GetProjectionRef()
        gt = ds.GetGeoTransform()
        self.left = float(gt[0])
        self.csx = float(gt[1])
        self.top = float(gt[3])
        self.csy = float(abs(gt[5]))
        self.shape = (ds.RasterYSize, ds.RasterXSize)
        self.bottom = self.top - (self.csy * self.shape[0])
        self.right = self.left + (self.csx * self.shape[1])
        self.bands = ds.RasterCount
        band = ds.GetRasterBand(1)
        self.dtype, self.size =\
            self.getDataSpecs(gdal.GetDataTypeName(band.DataType), self.shape)
        self.nodata = []
        for i in range(1, self.bands + 1):
            band = ds.GetRasterBand(i)
            try:
                nd = band.GetNoDataValue()
            except TypeError:
                nd = getndtype(self.datatype)
            self.nodata.append(nd)
        self.ds = ds
        self.createTileDefn()

    def populateFromESRIRaster(self, ds):
        d = arcpy.Describe(ds)
        self.top = d.extent.YMax
        self.bottom = d.extent.YMin
        self.left = d.extent.XMin
        self.right = d.extent.XMax
        self.csx = d.meanCellWidth
        self.csy = d.meanCellHeight
        self.shape = (d.height, d.width)
        self.projection = d.SpatialReference
        self.projection = self.projection.exportToString()
        self.bands = d.bandCount
        self.dtype = self.parseESRIDatatype(d.pixelType)
        self.dtype, self.size = self.getDataSpecs(self.dtype, self.shape)
        # TODO Fix to read each band nodata value
        self.nodata = [d.noDataValue]
        self.ds = ds
        self.createTileDefn()

    def createTileDefn(self):
        if self.size < self.tilesize:
            self.tiles = [[0, 0, self.shape[1], self.shape[0]]]
            return
        num_tiles = self.size / self.tilesize
        sq = numpy.sqrt(num_tiles)
        ratio = self.shape[0] / float(self.shape[1])
        itiles = int(round(sq * ratio))
        jtiles = int(round(num_tiles / itiles))
        ispan, jspan = (int(round(self.shape[0] / itiles)),
                        int(round(self.shape[1] / jtiles)))
        tiledefn = [[j * jspan, i * ispan, jspan, ispan]
                    for i in range(itiles) for j in range(jtiles)]
        if tiledefn[-1][0] + tiledefn[-1][2] > self.shape[1]:
            tiledefn[-1][2] = self.shape[1] - tiledefn[-1][0]
        elif tiledefn[-1][0] + tiledefn[-1][2] < self.shape[1]:
            tiledefn[-1][2] += self.shape[1] - (tiledefn[-1][0] + tiledefn[-1][2])
        if tiledefn[-1][1] + tiledefn[-1][3] > self.shape[0]:
            tiledefn[-1][3] = self.shape[0] - tiledefn[-1][1]
        elif tiledefn[-1][1] + tiledefn[-1][3] < self.shape[0]:
            tiledefn[-1][3] += self.shape[0] - (tiledefn[-1][1] + tiledefn[-1][3])
        self.tiles = tiledefn

    def read(self, band=1, path=None):
        '''Create a brf raster (on-disk array of raster data)'''
        if hasattr(self, 'array') and not hasattr(self, 'ds'):
            if path is not None:
                raise Exception('No file written to specified path, as an'
                                ' array already exists. Use the .save(path)'
                                ' method.')
            return
        print ('Reading raster from data source')
        if path is None:
            self.array = self.writeFile(dtype=self.dtype, shape=self.shape)
        else:
            self.array = self.writeFile(path=path)
        mmap = self.load('r+')
        if datadriver == 'GDAL':
            band = self.ds.GetRasterBand(band)
            for tile in self.tiles:
                xoff, yoff, win_xsize, win_ysize = tile
                a = band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=win_xsize,
                                     win_ysize=win_ysize)
                mmap[yoff: yoff + win_ysize, xoff: xoff + win_xsize] = a
        elif datadriver == 'ESRI':
            for tile in self.tiles:
                xoff, yoff, win_xsize, win_ysize = tile
                blc, ncols, nrows = self.tileToESRICoordinates(tile)
                a = arcpy.RasterToNumPyArray(self.ds, blc, ncols,
                                             nrows)
                if self.bands > 1:
                    a = a[band - 1, :, :]
                mmap[yoff:yoff + win_ysize, xoff:xoff + win_xsize] = a
        del self.ds
        mmap.flush()
        del mmap

    def changeDataType(self, new_datatype):
        '''Change to a different data type'''
        new_datatype = new_datatype.lower()
        if self.dtype == new_datatype:
            return
        hierarchy = ['bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32',
                     'int32', 'uint64', 'int64', 'float32', 'float64']
        if new_datatype not in hierarchy:
            raise Exception('Unsupported data type "%s"' % new_datatype)
        if ([i for i in range(len(hierarchy)) if
             new_datatype == hierarchy[i]][0] <
            [i for i in range(len(hierarchy)) if
             self.dtype == hierarchy[i]][0]):
            print "Warning: Loss of precision with data type change"
        self.read()
        # Create a writeable copy
        copyfile = self.writeFile(dtype=new_datatype, shape=self.shape)
        self.runTiles(changedtypetask, args=(new_datatype, copyfile))
        self.array = copyfile
        self.dtype = new_datatype
        for band in range(self.bands):
            self.nodata[band] = numpy.asscalar(numpy.array(
                self.nodata[band]).astype(new_datatype))
        self.createTileDefn()

    def tileToESRICoordinates(self, tile):
        xoff, yoff, win_xsize, win_ysize = tile
        y = (self.top - ((yoff + win_ysize) * self.csy))
        x = (self.left + (xoff * self.csx))
        return arcpy.Point(x, y), win_xsize, win_ysize

    def save(self, path):
        '''save current self as brf'''
        if not hasattr(self, 'array'):
            self.read(path=path)
        else:
            path = self.writeFile(path=path)

    def saveRaster(self, path=None):
        '''
        Save to a raster file.  If not data driver is available,
        a surfer .grd will be created.  Returns the output path
        if so can be used if temporary (no path specified).
        '''
        if datadriver == 'GDAL':
            if path:
                if path.split('.')[-1] != 'tif':
                    path = path + '.tif'
            else:
                path = self.writeFile('tif')
            driver = gdal.GetDriverByName('GTiff')
            if hasattr(self, 'array'):
                # Need to create a new dataset since underlying data may have
                # been changed.
                outdtype = self.numpyToGDALDtype(self.dtype)
                if outdtype == 'Byte':
                    self.changeDataType('int8')
                self.ds = driver.Create(path,
                                        self.shape[1],
                                        self.shape[0],
                                        self.bands,
                                        gdal.GetDataTypeByName(outdtype))
                mmap = self.load('r')
                for band in range(1, self.bands + 1):
                    b = self.ds.GetRasterBand(band)
                    b.SetNoDataValue(self.nodata[band - 1])
                    for tile in self.tiles:
                        xoff, yoff, win_xsize, win_ysize = tile
                        b.WriteArray(mmap[yoff:yoff +
                                     win_ysize, xoff:xoff + win_xsize],
                                     xoff=xoff, yoff=yoff)
                    b.FlushCache()
                    b = None
            else:
                # Just create a copy and update the attributes
                self.ds = driver.CreateCopy(path, self.ds)
                for band in range(1, self.bands + 1):
                    b = self.ds.GetRasterBand(band)
                    b.SetNoDataValue(self.nodata[band - 1])
                    b.FlushCache()
                    b = None
            self.ds.SetProjection(self.projection)
            self.ds.SetGeoTransform((self.left, self.csx, 0, self.top, 0,
                                     self.csy * -1))
            del self.ds
            if not hasattr(self, 'array'):
                self.ds = gdal.Open(path)
        elif datadriver == 'ESRI':
            if path:
                path = self.getCleanPath(path, 'tif')
            else:
                path = self.writeFile('tif')
            # Get tempfile name
            outpath = self.writeFile('')
            self.read()
            rasters = []
            mmap = numpy.load(self.array, mmap_mode='r')
            for i, tile in enumerate(self.tiles):
                xoff, yoff, win_xsize, win_ysize = tile
                blc, ncols, nrows = self.tileToESRICoordinates(tile)
                a = mmap[yoff: yoff + win_ysize,
                         xoff: xoff + win_xsize]
                r = arcpy.NumPyArrayToRaster(a, blc, self.csx, self.csy,
                                             self.nodata[0])
                r.save(outpath + str(i) + '.tif')
                rasters.append(outpath + str(i) + '.tif')
                del a, r
            proj = arcpy.SpatialReference()
            proj.loadFromString(self.projection)
            arcpy.MosaicToNewRaster_management(rasters, os.path.dirname(path),
                                               os.path.basename(path),
                                               proj,
                                               self.numpyToESRIDtype(
                                                   self.dtype),
                                               number_of_bands=self.bands)
            [os.remove(rast) for rast in rasters]
        else:
            # Save as a surfer ascii grid
            if path:
                if path.split('.')[-1] != 'grd':
                    path = path + '.grd'
            else:
                path = self.writeFile('grd')
            # Change data type to double
            self.changeDataType('float64')
            # Get info for header (extent, min, max)
            bottom = self.bottom + (abs(self.csy) * 0.5)
            top = bottom + (abs(self.csy) * (self.shape[0] - 1))
            left = self.left + (self.csx * 0.5)
            right = left + (self.csx * (self.shape[1] - 1))
            minval = float(numpy.finfo('float64').max)
            maxval = float(numpy.finfo('float64').min)
            mmap = numpy.load(self.array, mmap_mode='r')
            for tile in self.tiles:
                xoff, yoff, win_xsize, win_ysize = tile
                a_ = mmap[yoff:yoff + win_ysize, xoff:xoff + win_xsize]
                m = a_ == self.nodata[0]
                # Change nodata to surfer blanking value
                a_[m] = 1.70141e+38
                a_ = a_[~m]
                if a_.size == 0:
                    continue
                imax = numpy.max(a_)
                imin = numpy.min(a_)
                if imax > maxval:
                    maxval = imax
                if imin < minval:
                    minval = imin
            header = 'DSAA\n%i %i\n%s %s\n%s %s\n%s %s\n' % (
                self.shape[1],
                self.shape[0],
                left,
                right,
                bottom,
                top,
                minval,
                maxval
            )

            with open(path, 'w') as f:
                f.write(header)
                # Write line by line, backwards
                [f.write(' '.join(
                         self.array[i, :].astype('str').tolist()
                         ) + '\n')
                    for i in range(self.shape[0] - 1, -1, -1)]
        return path

    def project(self, projection, transformation, out_csx=None, out_csy=None,
                snap_point=None):
        '''
        Reproject a raster into a desired projection.
        projection: wkt, wkb, epsg, osr.SpatialReference instance,
            ESRI projection name, or ESRI spatialReference instance
        Available transformations: ('nearest', 'bilinear', 'cubic',
        'mean', 'mode')
        snap_point is a tuple (x,y) in the output coordinate system,
        which is aligned to in the output if specified.
        TODO: Add band support and parallelize
        '''
        projection = self.parseProjection(projection)
        transformation = self.getRelevantTransform(transformation)
        if datadriver == 'GDAL':
            # Recompute bounding coordinates
            insr = osr.SpatialReference()
            insr.ImportFromWkt(self.projection)
            outsr = osr.SpatialReference()
            outsr.ImportFromWkt(projection)
            transform = osr.CoordinateTransformation(insr, outsr)
            left, top, _ = transform.TransformPoint(self.left,
                                                    self.top)
            right, bottom, _ = transform.TransformPoint(self.right,
                                                        self.bottom)
            # Recompute new cell size if necessary
            if out_csx is None:
                csx = (right - left) / self.shape[1]
            else:
                csx = float(out_csx)
            if out_csy is None:
                csy = (top - bottom) / self.shape[0]
            else:
                csy = float(out_csy)
            # Change to snap if necessary
            if snap_point is not None:
                left, top = self.snapToPoint((left, top), snap_point, csx, csy)
                right, bottom = self.snapToPoint((right, bottom), snap_point,
                                                 csx, csy, False)
            # Compute new shape
            shape = (int(round((top - bottom) / csy)),
                     int(round((right - left) / csx)))
            # Save temporary raster if has array instance
            if hasattr(self, 'array'):
                temprast = self.saveRaster()
                self.ds = gdal.Open(temprast)
            # Create temporary output raster
            driver = gdal.GetDriverByName('GTiff')
            outrast = self.writeFile('tif')
            dst = driver.Create(outrast,
                                shape[1],
                                shape[0], 1,
                                gdal.GetDataTypeByName(
                                    self.numpyToGDALDtype(self.dtype)))
            dst.SetGeoTransform((left, csx, 0, top, 0, csy * -1))
            dst.SetProjection(projection)
            for band in range(1, self.bands + 1):
                b = dst.GetRasterBand(band)
                b.SetNoDataValue(self.nodata[band - 1])
                b.FlushCache()
                b = None
            gdal.ReprojectImage(self.ds, dst, self.projection, projection,
                                transformation)
            dst = None
            dst = gdal.Open(outrast)
            # Update Attributes
            self.populateFromGDALRaster(dst)
        elif datadriver == 'ESRI':
            outcs = None
            if out_csx is not None and out_csy is not None:
                if out_csx != out_csy:
                    raise Exception('Using the ESRI datadriver the output cell'
                                    ' sizes must be square. I know, right!?')
                outcs = out_csx
            elif out_csx is not None:
                outcs = out_csx
            elif out_csy is not None:
                outcs = out_csy
            proj = arcpy.SpatialReference()
            proj.loadFromString(projection)
            if snap_point is not None:
                # Create a scratch snap raster for the env (stupid, right?)
                a = numpy.ones(shape=(1, 1), dtype='int8')
                snaprast = self.writeFile('tif')
                if outcs is not None:
                    r = arcpy.NumPyArrayToRaster(a,
                                                 arcpy.Point(snap_point[0],
                                                             snap_point[1]),
                                                 outcs, outcs, 0)
                else:
                    raise Exception('A snap point cannot be used with the ESRI'
                                    ' data driver when no output cell size is'
                                    ' included in the arguments.')
                r.save(snaprast)
                del r
                arcpy.DefineProjection_management(snaprast, proj)
                arcpy.env.snapRaster = snaprast
            if hasattr(self, 'array'):
                temprast = self.saveRaster()
                self.ds = temprast
            outrast = self.writeFile('tif')
            if outcs is not None:
                arcpy.ProjectRaster_management(self.ds, outrast, proj,
                                               transformation, float(outcs))
            else:
                arcpy.ProjectRaster_management(self.ds, outrast, proj,
                                               transformation)
            dst = arcpy.Raster(outrast)
            # Update Attributes
            self.populateFromESRIRaster(dst)
            if snap_point is not None:
                arcpy.env.snapRaster = None
        else:
            raise Exception("No data driver is available to complete the"
                            " projection (No PROJ library exists for the .brf"
                            " type alone)")
        # Clean up and replace data if necessary
        if hasattr(self, 'array'):
            self.ds = None
            self.ds = dst
            self.read()
        else:
            self.ds = dst

    def transform(self, transformation, out_csx, out_csy, snap_point=None):
        '''
        Transform a raster to a different cell size
        Use when not reprojecting with a brf type
        for much faster transformations.
        transformation can be: ('spline', 'bilinear', )
        NOT IMPLEMENTED
        '''
        transformation = self.getRelevantTransform(transformation)
        csx, csy = float(out_csx), float(out_csy)
        if snap_point is not None:
            top, left = self.snapToPoint((left, top), snap_point, csx, csy)

    def changeExtent(self, bbox, snap_tolerance=1E-5):
        '''
        Fit to the bounding box (rounded extent-outwards).
        bbox is (top, bottom, left, right)
        '''
        bbox = map(float, bbox)
        if bbox[0] < bbox[1]:
            raise Exception('The top is below the bottom.')
        if bbox[2] > bbox[3]:
            raise Exception('The left is to the right of the right.')
        allempty = False
        if any([bbox[0] < self.bottom, bbox[1] > self.top,
                bbox[2] > self.right, bbox[3] < self.left]):
            allempty = True
        if not allempty:
            self.read()
        # Find new shape and extent
        selfbbox = (self.top, self.bottom, self.left, self.right)
        newbbox, shape = self.parseExtent(bbox, selfbbox, self.csx, self.csy,
                                          tol=snap_tolerance)
        top, bottom, left, right = newbbox
        # Allocate output
        copyarray = self.writeFile(dtype=self.dtype, shape=shape)
        outmmap = numpy.load(copyarray, 'r+')
        inmmap = self.load('r')
        if not allempty:
            # Find insertion location
            ioffset = int((self.top - top) / self.csy)
            joffset = int((left - self.left) / self.csx)
            if ioffset < 0:
                insi = abs(ioffset)
                ioffset = 0
            else:
                insi = 0
            if joffset < 0:
                insj = abs(joffset)
                joffset = 0
            else:
                insj = 0
            ispan = shape[0] + ioffset
            jspan = shape[1] + joffset
            if ispan > self.shape[0]:
                ispan = self.shape[0]
                insispan = shape[0] - (ispan - self.shape[0])
            else:
                insispan = shape[0]
            if jspan > self.shape[1]:
                jspan = shape[1]
                insjspan = shape[1] - (jspan - self.shape[1])
            else:
                insjspan = shape[1]
            outmmap[insi:insispan, insj:insjspan] = inmmap[ioffset:ispan,
                                                           joffset:jspan]
        # Update attributes
        self.array = copyarray
        self.top, self.bottom, self.left, self.right = top, bottom, left, right
        self.shape = shape
        # Update size and tile attributes
        _, self.size = self.getDataSpecs(self.dtype, self.shape)
        self.createTileDefn()
        del outmmap

    def matchRaster(self, inrast, transformation, snap_tolerance=1E-5):
        '''
        Modify self to match the input raster.
        Add transform in the future...
        '''
        if not isinstance(inrast, raster):
            raise Exception('Expected a raster instance as input and got %s' %
                            type(inrast))
        if inrast.projection != self.projection and datadriver is None:
            raise Exception('Cannot match rasters: no datadriver is available'
                            ' and projections do not match.')
        snap_point = (inrast.left, inrast.top)
        bbox = (inrast.top, inrast.bottom, inrast.left, inrast.right)
        # Check if can be reduced before operations
        if datadriver == 'GDAL':
            insr = osr.SpatialReference()
            insr.ImportFromWkt(inrast.projection)
            outsr = osr.SpatialReference()
            outsr.ImportFromWkt(self.projection)
            transform = osr.CoordinateTransformation(insr, outsr)
            left, top, _ = transform.TransformPoint(inrast.left,
                                                    inrast.top)
            right, bottom, _ = transform.TransformPoint(inrast.right,
                                                        inrast.bottom)
        elif datadriver == 'ESRI':
            intlc = arcpy.Point(inrast.left, inrast.top)
            inblc = arcpy.Point(inrast.right, inrast.bottom)
            proj = arcpy.SpatialReference()
            proj.loadFromString(self.projection)
            inproj = arcpy.SpatialReference()
            inproj.loadFromString(inrast.projection)
            pointtlc = arcpy.PointGeometry(intlc, inproj)
            pointblc = arcpy.PointGeometry(inblc, inproj)
            selftlc = pointtlc.projectAs(proj)
            selftlc = map(float, str(selftlc.centroid).split(' '))
            selfblc = pointblc.projectAs(proj)
            selfblc = map(float, str(selfblc.centroid).split(' '))
            left, top = selftlc[0], selftlc[1]
            right, bottom = selfblc[0], selfblc[1]
        else:
            top, bottom, left, right = (self.top, self.bottom, self.left,
                                        self.right)
        newbbox, shape = self.parseExtent((top, bottom, left, right),
                                          (self.top, self.bottom,
                                           self.left, self.right),
                                          self.csx, self.csy,
                                          tol=snap_tolerance)
        if (shape[0] * shape[1]) < (self.shape[0] * self.shape[1]):
            self.changeExtent(newbbox, snap_tolerance=snap_tolerance)
        # Perform transformation
        self.project(inrast.projection, transformation, out_csx=inrast.csx,
                     out_csy=inrast.csy, snap_point=snap_point)
        # Align extents
        self.changeExtent(bbox, snap_tolerance=snap_tolerance)
        # Test output
        if self.shape != inrast.shape:
            raise Exception('The rasters could not be matched.  A raster with'
                            ' the shape %s and extent %s was created from an'
                            ' input raster with the shape %s and extent %s' %
                            (self.shape, (self.top, self.bottom, self.left,
                                          self.right), inrast.shape,
                             (inrast.top, inrast.bottom, inrast.left,
                             inrast.right)))

    def rastermin(self, band=1):
        '''Return minimum value'''
        self.read()
        minima = numpy.array(self.runTiles(mintask))
        return numpy.min(minima[minima != self.nodata[band - 1]])

    def rastermax(self, band=1):
        '''Return maximum value'''
        self.read()
        maxs = numpy.array(self.runTiles(maxtask, band))
        return numpy.max(maxs[maxs != self.nodata[band - 1]])

    def rastermean(self, band=1):
        '''Return mean value'''
        self.read()
        means = numpy.array(self.runTiles(meantask, band))
        return numpy.mean(means[means != self.nodata[band - 1]])

    def rastersum(self, band=1):
        '''Return sum'''
        self.read()
        sums = numpy.array(self.runTiles(sumtask, band))
        return numpy.sum(sums[sums != self.nodata[band - 1]])

    def runTiles(self, function, band=1, args=None):
        '''Run a function on tiles'''
        nodataiter = [self.nodata[band - 1] for b in range(len(self.tiles))]
        afiter = [self.array for b in range(len(self.tiles))]
        if args is not None:
            if not hasattr(args, '__iter__'):
                raise Exception('Args must be tuple or list.')
            iterable = [args for b in range(len(self.tiles))]
            iterable = [(afiter[i], self.tiles[i], nodataiter[i]) +
                        iterable[i] for i in range(len(self.tiles))]
        else:
            iterable = zip(afiter, self.tiles, nodataiter)
        if not self.parallelize:
            return [function(i) for i in iterable]
        # Check available resources
        cores = multiprocessing.cpu_count() - 1
        if cores <= 0:
            cores = 1
        # Initiate pool of workers
        p = multiprocessing.Pool(cores)
        try:
            ret = list(p.imap_unordered(function, iterable))
        except Exception as e:
            import sys
            p.close()
            p.join()
            raise e, None, sys.exc_info()[2]
        else:
            p.close()
            p.join()
            return ret

    def writeFile(self, extension='brf', path=None, dtype=None, shape=None):
        '''
        Get a clean file path, create a new or temporary dataset,
        and/or allocate an on-disk array.
        Empty use returns tempfile path with the chosen extension.
        Using no path, but dtype and shape produces a temporary empty file
            for writing.
        Assigning a path copies self- dtype and shape are ignored in this case.
        '''
        if path is not None:
            dirname = os.path.dirname(path)
            if not os.path.isdir(dirname) and len(dirname) > 0:
                raise Exception('The chosen file path "%s" does not exist' %
                                os.path.dirname(path))
            if path.split('.')[-1] != extension:
                path = path + '.' + extension
            outpath = path
            tempa = numpy.lib.format.open_memmap(outpath, 'w+', self.dtype,
                                                 self.shape)
            if not hasattr(self, 'array'):
                # Create a new file for writing with outpath
                del tempa
                self.array = outpath
                writedict = dict(self.__dict__)
                try:
                    del writedict['tempfiles']
                except:
                    pass
                try:
                    del writedict['ds']
                except:
                    pass
            else:
                # Rewrite self with attribtues
                mmap = numpy.load(self.array, mmap_mode='r')
                for tile in self.tiles:
                    xoff, yoff, win_xsize, win_ysize = tile
                    tempa[yoff: yoff + win_ysize, xoff: xoff + win_xsize] =\
                        mmap[yoff: yoff + win_ysize, xoff: xoff + win_xsize]
                tempa.flush()
                del tempa
                writedict = {k: v for k, v in self.__dict__.iteritems() if
                             k != 'array'}
                writedict['array'] = outpath
                try:
                    del writedict['tempfiles']
                except:
                    pass
                try:
                    del writedict['ds']
                except:
                    pass
            with open(outpath, 'a') as f:
                f.write('%s%s' % ('brfmeta', writedict))
            return outpath
        else:
            with tempfile.NamedTemporaryFile() as tf:
                outpath = '%s.%s' % (tf.name, extension)
            if self.tmpspace is not None:
                outpath = os.path.join(self.tmpspace,
                                       os.path.basename(outpath))
            if not hasattr(self, 'tempfiles'):
                self.tempfiles = []
            self.tempfiles.append(outpath)
            if dtype is not None and shape is not None:
                tempa = numpy.lib.format.open_memmap(outpath, 'w+', dtype,
                                                     shape)
                del tempa
                writedict = {k: v for k, v in self.__dict__.iteritems()
                             if k != 'array'}
                writedict['array'] = outpath
                try:
                    del writedict['tempfiles']
                except:
                    pass
                try:
                    del writedict['ds']
                except:
                    pass
                with open(outpath, 'a') as f:
                    f.write('%s%s' % ('brfmeta', writedict))
        return outpath

    def load(self, mode='r', in_memory=False):
        '''
        Load underlying data into memory-mapped array.
        Modes are:
        r (read)
        r+ (read-write)
        w+ (replace and write)
        c (read only, but change in-memory data)
        '''
        if not hasattr(self, 'array'):
            self.read()
        mma = numpy.load(self.array, mmap_mode=mode)
        if in_memory:
            return numpy.copy(mma)
        else:
            return mma

    @staticmethod
    def parseExtent(new_bbox, old_bbox, csx, csy, tol=1E-5):
        '''Get the specs for an extent change'''
        left, top = raster.snapToPoint((new_bbox[2], new_bbox[0]),
                                       (old_bbox[2], old_bbox[0]), csx, csy,
                                       tol=tol)
        right, bottom = raster.snapToPoint((new_bbox[3], new_bbox[1]),
                                           (old_bbox[3], old_bbox[1]), csx,
                                           csy, False, tol=tol)
        shape = (int(round((top - bottom) / csy)),
                 int(round((right - left) / csx)))
        return (top, bottom, left, right), shape

    @staticmethod
    def snapToPoint(input_point, snap_point, csx, csy, topleft=True, tol=1E-5):
        '''
        Snap a point (x, y) to a snap point (x, y).
        topleft to force up and left, and not topleft to force right and down
        '''
        xresidual = (input_point[0] - snap_point[0]) % csx
        if any([(xresidual + tol) > csx > (xresidual - tol),
                (xresidual + tol) > 0 > (xresidual - tol)]):
            xresidual = 0
        yresidual = (snap_point[1] - input_point[1]) % csy
        if any([(yresidual + tol) > csy > (yresidual - tol),
                (yresidual + tol) > 0 > (yresidual - tol)]):
            yresidual = 0
        if ((input_point[0] - xresidual) - snap_point[0]) % csx == 0:
            x = input_point[0] - xresidual
        else:
            x = input_point[0] + xresidual
        if ((input_point[1] - yresidual) - snap_point[1]) % csy == 0:
            y = input_point[1] - yresidual
        else:
            y = input_point[1] + yresidual
        if topleft:
            if x > input_point[0]:
                x -= csx
            if y < input_point[1]:
                y += csy
        else:
            if x < input_point[0]:
                x += csx
            if y > input_point[1]:
                y -= csy
        return x, y

    @staticmethod
    def parseData(data):
        # First look for extension
        if isinstance(data, basestring):
            ext = data.split('.')[-1]
            if ext == 'brf':
                # load brf dataset
                brfdata = raster.assertBrf(data)
                if brfdata is None:
                    raise Exception('Cannot read the input data: "%s"' % data)
                return 'brf', brfdata
            else:
                rasterdata = raster.assertGDALRaster(data)
                if rasterdata is None:
                    raise Exception('Cannot read the input data: "%s"' % data)
                return 'gdalraster', rasterdata
        elif isinstance(data, raster):
            return 'blueraster', data
        elif datadriver == 'GDAL':
            if isinstance(data, gdal.Dataset):
                return 'gdalraster', data
        elif datadriver == 'ESRI':
            if isinstance(data, arcpy.Raster):
                return 'arcpyraster', data
        else:
            raise Exception('Unrecognized input data of type "%s"' %
                            type(data))

    @staticmethod
    def assertGDALRaster(path):
        if datadriver == 'GDAL':
            ds = gdal.Open(path)
            return ds
        elif datadriver == 'ESRI':
            try:
                ds = arcpy.Raster(path)
                return ds
            except IOError:
                return None
        else:
            raise Exception('Dataset cannot be openend: No data driver'
                            ' library (i.e. GDAL or ESRI has been found'
                            ' in the current environment')

    @staticmethod
    def assertBrf(path):
        try:
            with open(path, 'r') as f:
                line, i = '', 0
                while line != 'brfmeta':
                    i -= 1
                    f.seek(i, 2)
                    line = f.read()
                    if line[:7] == 'brfmeta':
                        return literal_eval(line[7:])
        except:
            return None

    @staticmethod
    def getDataSpecs(dtype, shape):
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
            raise Exception('Cannot read the input raster data type "%s"' %
                            dtype)

    @staticmethod
    def parseESRIDatatype(dtype):
        datatypes = {
            'U1': 'int8',
            'U4': 'int8',
            'U8': 'uint8',
            'S8': 'int8',
            'U16': 'uint16',
            'S16': 'int16',
            'U32': 'uint32',
            'S32': 'int32',
            'F32': 'float32',
            'F64': 'float64'
        }
        try:
            return datatypes[dtype]
        except KeyError:
            raise Exception('Cannot recognize the ESRI data type "%s".'
                            ' Check the local dictionary.' % dtype)

    @staticmethod
    def parseGDALDatatype(dtype):
        datatypes = {
            'Byte': 'uint8',
            'Int16': 'int16',
            'Int32': 'int32',
            'UInt16': 'uint16',
            'UInt32': 'uint32',
            'Int64': 'int64',
            'UInt64': 'uint64',
            'Float32': 'float32',
            'Float64': 'float64',
            'Long': 'int64',
        }
        try:
            return datatypes[dtype]
        except KeyError:
            raise Exception('Cannot recognize the GDAL data type "%s".'
                            ' Check the local dictionary.' % dtype)

    @staticmethod
    def numpyToESRIDtype(dtype):
        datatypes = {
            'bool': '1_BIT',
            'uint8': '8_BIT_UNSIGNED',
            'int8': '8_BIT_SIGNED',
            'uint16': '16_BIT_UNSIGNED',
            'int16': '16_BIT_SIGNED',
            'uint32': '32_BIT_UNSIGNED',
            'int32': '32_BIT_SIGNED',
            'float32': '32_BIT_FLOAT',
            'float64': '64_BIT',
            'uint64': '64_BIT',
            'int64': '64_BIT'
        }
        try:
            return datatypes[dtype]
        except KeyError:
            raise Exception('Cannot recognize the numpy data type "%s".'
                            ' Check the local dictionary.' % dtype)

    @staticmethod
    def numpyToGDALDtype(dtype):
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
            return datatypes[dtype]
        except KeyError:
            raise Exception('Cannot recognize the numpy data type "%s".'
                            ' Check the local dictionary.' % dtype)

    @staticmethod
    def getRelevantTransform(transformation):
        '''Return a driver-based transformation'''
        if datadriver == 'ESRI':
            types = {
                'bilinear': 'BILINEAR',
                'nearest': 'NEAREST',
                'cubic': 'CUBIC',
                'mode': 'MAJORITY'
            }
        elif datadriver == 'GDAL':
            types = {
                'bilinear': GRA_Bilinear,
                'nearest': GRA_NearestNeighbour,
                'cubic': GRA_Cubic,
                'cubic spline': GRA_CubicSpline,
                'lanczos': GRA_Lanczos,
                'mean': GRA_Average,
                'mode': GRA_Mode
            }
        else:
            types = {
                'Sorry, no options': None
            }
        try:
            return types[transformation]
        except KeyError:
            raise Exception('Unsupported transormation type "%s" using the %s'
                            ' data driver. Use one of: %s' %
                            (transformation, datadriver, types.keys()))

    @staticmethod
    def parseProjection(projection):
        '''Return a wkt from some argument'''
        if datadriver == 'GDAL':
            if isinstance(projection, basestring):
                sr = osr.SpatialReference()
                sr.ImportFromWkt(projection)
                outwkt = sr.ExportToWkt()
                if outwkt == '':
                    raise Exception('Cannot recognize the input string'
                                    ' "%s"' % projection)
            elif isinstance(projection, osr.SpatialReference):
                return projection.ExportToWkt()
            elif isinstance(projection, int):
                sr = osr.SpatialReference()
                sr.ImportFromEPSG(projection)
                outwkt = sr.ExportToWkt()
                if outwkt == '':
                    raise Exception('Cannot recognize the input projection'
                                    ' "%s"' % projection)
            return outwkt
        elif datadriver == 'ESRI':
            if isinstance(projection, arcpy.SpatialReference):
                sr = projection
            else:
                try:
                    sr = arcpy.SpatialReference(projection)
                except:
                    try:
                        sr = arcpy.SpatialReference()
                        sr.loadFromString(projection)
                    except:
                        raise Exception('Cannot recognize the input projection'
                                        ' "%s"' % projection)
            return sr.exportToString()
        else:
            raise Exception('No data driver exists to parse projection')

    def __del__(self):
        if hasattr(self, 'ds'):
            self.ds = None
        if hasattr(self, 'tempfiles'):
            for rast in self.tempfiles:
                try:
                    os.remove(rast)
                except Exception as e:
                    print "Unable to clean up file %s because: %s" % (rast, e)

'''
Other supporting methods
'''


def changedtypetask(args):
    af, tile, nodata, outdtype, outf = args
    xoff, yoff, win_xsize, win_ysize = tile
    af = numpy.load(af, mmap_mode='r')
    outf = numpy.load(outf, mmap_mode='r+')
    outf[yoff:yoff + win_ysize, xoff:xoff + win_xsize] =\
        af[yoff:yoff + win_ysize, xoff:xoff + win_xsize].astype(outdtype)
    outf.flush()


def mintask(args):
    af, tile, nodata = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r')
    a = numpy.copy(mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize])
    a = a[a != nodata]
    if a.size == 0:
        return nodata
    return numpy.min(a)


def maxtask(args):
    af, tile, nodata = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r')
    a = numpy.copy(mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize])
    a = a[a != nodata]
    if a.size == 0:
        return nodata
    return numpy.max(a)


def meantask(args):
    af, tile, nodata = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r')
    a = numpy.copy(mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize])
    a = a[a != nodata]
    if a.size == 0:
        return nodata
    return numpy.mean(a)


def sumtask(args):
    af, tile, nodata = args
    xoff, yoff, win_xsize, win_ysize = tile
    mma = numpy.load(af, mmap_mode='r')
    a = numpy.copy(mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize])
    a = a[a != nodata]
    if a.size == 0:
        return nodata
    return numpy.sum(a)
