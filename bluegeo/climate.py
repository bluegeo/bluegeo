"""
Downscale climate data using a grid or an input file with points

blueGeo, 2018
"""

import numpy
import h5py
from contextlib import contextmanager
import os
from osgeo import gdal, osr
import csv
from numba import jit
import time
import multiprocessing as mp
import shutil


class Climate(object):
    """
    Compute downscaled climate grids, or output a group of points to a .csv

    The Climate NA software is not necessary, although the folder of GCM/RCM grids is required
    """

    PRISM_PARAMETERS = ['Elevation', 'Tmax01', 'Tmax02', 'Tmax03', 'Tmax04', 'Tmax05', 'Tmax06',
                        'Tmax07', 'Tmax08', 'Tmax09', 'Tmax10', 'Tmax11', 'Tmax12', 'Tmin01',
                        'Tmin02', 'Tmin03', 'Tmin04', 'Tmin05', 'Tmin06', 'Tmin07', 'Tmin08',
                        'Tmin09', 'Tmin10', 'Tmin11', 'Tmin12', 'PPT01', 'PPT02', 'PPT03',
                        'PPT04', 'PPT05', 'PPT06', 'PPT07', 'PPT08', 'PPT09', 'PPT10', 'PPT11',
                        'PPT12', 'RAD01', 'RAD02', 'RAD03', 'RAD04', 'RAD05', 'RAD06', 'RAD07',
                        'RAD08', 'RAD09', 'RAD10', 'RAD11', 'RAD12']

    def __init__(self, climate_na_path, output_directory):
        """
        Hard coded grid specs and load prism data
        :param grid_path: path to Climate NA directory
        """
        self.output_directory = output_directory
        self.wkdir = climate_na_path
        self.garbage = []

        self.normal_csx = 0.5
        self.normal_csy = 0.5
        self.normal_shape = (144, 258)
        self.normal_top, self.normal_left = (84.994, -179.667)

        self.gcm_csx = 1
        self.gcm_csy = 1
        self.gcm_shape = (72, 129)
        self.gcm_top, self.gcm_left = (84.994, -179.667)

        self.prism_csx = 2.5 / 60
        self.prism_csy = 2.5 / 60
        self.prism_shape = (1651, 3038)
        # (83.238, -179.238)
        self.prism_top, self.prism_left = (83.23, -179.23)
        self.prismBottom = self.prism_top - (self.prism_shape[0] * self.prism_csy)
        self.prismRight = self.prism_left + (self.prism_shape[1] * self.prism_csx)
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4269)
        self.projection = sr.ExportToWkt()

        self.h5file = os.path.join(output_directory, 'climate.h5')
        if not os.path.isfile(self.h5file):
            print "PRISM and delta data must be loaded from the source.  This will take a while."
            f = h5py.File(self.h5file, mode='w', libver='latest')
            f.close()
            # Load PRISM data
            print "Loading PRISM data"
            # Unsure why this is needed:
            # with open(os.path.join(os.path.join(self.wkdir, 'prismdat'), 'index.dat')) as f:
            #     lines = f.readlines()[0]
            # self.prism_index = numpy.array([float(lines[i:_i].replace(' ', '')) for i, _i in zip(
            #     range(0, len(lines) + 1, 7)[:-1], range(0, len(lines) + 1, 7)[1:-1] + [len(lines)]
            # )], dtype='int64')

            with open(os.path.join(os.path.join(self.wkdir, 'prismdat'), 'prism_dv.dat')) as f:
                self.prism_data = [line.split(',') for line in f.readlines()[0].split()]

            # Need to fix two lines improperly delimited ?
            rpl1 = self.prism_data[8107][:50]
            rpl1[-1] = rpl1[-1][:6]
            rpl2 = self.prism_data[8107][49:]
            rpl2[0] = rpl2[0][6:]
            rpl3 = self.prism_data[45187][:50]
            rpl3[-1] = rpl3[-1][:6]
            rpl4 = self.prism_data[45187][49:]
            rpl4[0] = rpl4[0][6:]
            prism_data = numpy.vstack([self.prism_data[:8107], rpl1, rpl2,
                                       self.prism_data[8108:45187], rpl3, rpl4,
                                       self.prism_data[45188:]]).astype('int32')

            self.prism_index = numpy.unravel_index(prism_data[:, 0], self.prism_shape)

            # Save datasets to the h5 file
            template = numpy.full(self.prism_shape, -99990, 'float32')
            for i, name in enumerate(self.PRISM_PARAMETERS):
                template[self.prism_index] = prism_data[:, i + 1].astype('float32')
                if 'rad' in name.lower():
                    # Need to divide radiation data by 10 as radiation is saved as a integer with 1 decimal place
                    template[template != -99990] /= 10.
                elif 'ppt' not in name.lower() and name != 'Elevation':
                    # All remaining parameters are saved as integers with 2 decimal places
                    template[template != -99990] /= 100.
                self.save_ds(template, name)

            # Collect all of the scenarios
            period_dir = os.path.join(self.wkdir, 'Perioddat')
            periods = [os.path.join(period_dir, f) for f in os.listdir(period_dir) if f.lower() != 'cru_index.dat']
            period_index = \
            [os.path.join(period_dir, f) for f in os.listdir(period_dir) if f.lower() == 'cru_index.dat'][0]

            gcm_dir = os.path.join(self.wkdir, 'GCMdat')
            gcms = [os.path.join(gcm_dir, f) for f in os.listdir(gcm_dir) if
                    f.lower() not in ['gcm_index.dat', 'annual']]
            gcm_index = [os.path.join(gcm_dir, f) for f in os.listdir(gcm_dir) if f.lower() == 'gcm_index.dat'][0]

            annual_gcm_dir = os.path.join(gcm_dir, 'annual')
            annual_gcm_dirs = [os.path.join(annual_gcm_dir, f) for f in os.listdir(annual_gcm_dir)]
            annual_gcms = []
            for d in annual_gcm_dirs:
                for f in os.listdir(d):
                    gcm_or_dir = os.path.join(d, f)
                    if os.path.isdir(gcm_or_dir):
                        annual_gcms += [os.path.join(gcm_or_dir, f) for f in os.listdir(gcm_or_dir)]
                    else:
                        annual_gcms.append(gcm_or_dir)

            # Load all of the scenarios into the h5 file
            print "Loading normal deltas..."
            self.load_scenarios(periods, period_index)
            print "Loading GCM deltas..."
            self.load_scenarios(gcms, gcm_index)
            print "Loading annual GCM deltas..."
            self.load_scenarios(annual_gcms, gcm_index)

        else:
            print "Using the existing climate file:\n{}".format(self.h5file)

    def save_ds(self, a, name):
        with h5py.File(self.h5file, libver='latest') as f:
            f.create_dataset(name, data=a, compression='lzf')

    @contextmanager
    def dataset(self, name):
        f = h5py.File(self.h5file, libver='latest')
        ds = f[name]
        yield ds
        ds = None
        f.close()

    @property
    def scenarios(self):
        with h5py.File(self.h5file, libver='latest') as f:
            return f['scens'].keys()

    @staticmethod
    def parse_input(data):
        """Determine the input data type and try to read it"""
        if data.split('.')[-1].lower() == 'csv':
            type = 'csv'
            with open(data, 'rU') as f:
                data = [line for line in csv.reader(f)]
            data = numpy.array(
                map(tuple, data[1:]),
                dtype=[(n, t)
                       for n, t in zip(data[0], ['S50' if any([i.isalpha() for i in e])
                                                 else 'float32' for e in data[1]])]
            )
        else:
            # Try to read a raster
            type = 'raster'
            ds = gdal.Open(data)
            if ds is None:
                raise Exception('Unable to read the input file "{}"'.format(data))
            # Gather specs
            projection = ds.GetProjectionRef()
            gt = ds.GetGeoTransform()
            left = float(gt[0])
            csx = float(gt[1])
            top = float(gt[3])
            csy = float(abs(gt[5]))
            shape = (ds.RasterYSize, ds.RasterXSize)
            bottom = top - (csy * shape[0])
            right = left + (csx * shape[1])
            band = ds.GetRasterBand(1)
            dtype = gdal.GetDataTypeName(band.DataType).lower()
            if dtype == 'byte':
                dtype = 'uint8'
            a = band.ReadAsArray()
            nodata = band.GetNoDataValue()
            data = {'left': left, 'csx': csx, 'top': top, 'csy': csy, 'shape': shape, 'a': a,
                    'bottom': bottom, 'right': right, 'dtype': dtype, 'nodata': nodata}

        return data, type

    def load_scenarios(self, paths, index_path):
        """
        Load GCM or Period delta grids from the ClimateNA software package
        :param paths: List of paths to Period or GCM files
        :param index_path: Path to the Period or GCM index file
        :return: None
        """
        # Check that the files exist
        if any([not os.path.isfile(f) for f in paths]):
            raise Exception('Unable to find the file {}'.format([f for f in paths if not os.path.isfile(f)][0]))

        # Open the index data
        if not os.path.isfile(index_path):
            raise Exception('Cannot find the index path {}'.format(index_path))
        with open(index_path, 'r') as f:
            index = numpy.array(map(int, [line.strip() for line in f.readlines()[1:]]))

        if 'gcm' in os.path.basename(index_path).lower():
            scen_type = 'gcm'
            shape = self.gcm_shape
        else:
            scen_type = 'normal'
            shape = self.normal_shape

        # Load a template into memory
        template = numpy.full(shape, -99990, 'float32')
        write_locations = numpy.unravel_index(numpy.where(~(index == 0)), shape)

        progress = '0'
        for for_prog, path in enumerate(paths):
            with open(path, 'r') as f:
                lines = [line.strip().split(',') for line in f.readlines()]

            # Generate a scenario name
            d = os.path.basename(path)
            name = ['_'.join(d.split('.')[:-1])]
            n_path = os.path.dirname(path)
            while d not in ['GCMdat', 'Perioddat']:
                d = os.path.basename(n_path)
                name = [d] + name
                n_path = os.path.dirname(n_path)

            name = ' '.join(name).replace('_', ' ')

            # Iterate parameters and load
            for param in self.PRISM_PARAMETERS[1:]:  # Avoid elevation from first column
                # Need to wrangle some text for the parameters because of differences
                parameter = param.lower().replace('ppt', 'pre')
                if scen_type == 'gcm':
                    # GCM's have different headings
                    parameter = parameter.replace('rad', 'rsds')
                    if 'pre' in parameter and parameter[3] == '0':
                        parameter = parameter.replace('0', '')
                    elif 'pre' not in parameter and parameter[4] == '0':
                        parameter = parameter.replace('0', '')
                # Some weird stuff
                elif any(['rad03.1' in line.lower().replace('_', '') for line in lines[0]]) and parameter == 'rad04':
                    parameter = 'rad03.1'

                # Find the column of the parameter of interest as an index
                col = [i for i, e in enumerate(lines[0]) if parameter == e.replace('"', '').replace('_', '').lower()]
                if len(col) == 0:
                    raise Exception('Parameter {} not found'.format(parameter))
                if len(col) > 1:
                    raise Exception('Found more than one parameter match (found {})'.format(len(col)))
                col = col[0]

                # Write the data to the h5 file
                template[write_locations] = map(float, [line[col] for line in lines[1:]])

                self.save_ds(template, 'scens/{}^^^{}/{}'.format(scen_type, name, param))

            new_progress = '{:.0f}'.format(float(for_prog + 1) / len(paths) * 100.)
            if new_progress != progress and new_progress[-1] in ['5', '0']:
                progress = new_progress
                print progress + '%'

    def write_csv(self, rows):
        """Write an output csv using point data generated during downscaling"""
        headings = ['ID', 'Longitude', 'Latitude', 'Elevation', 'Source', 'Month', 'Parameter', 'Value']

        path = os.path.join(os.path.dirname(self.h5file), 'climate_output')
        cnt = 0
        while os.path.isfile(path + '.csv'):
            cnt += 1
            path += '_{}'.format(cnt)

        path += '.csv'
        with open(path, 'wb') as f:
            f.write(','.join(headings) + '\n')
            f.write('\n'.join([','.join(row) for row in rows]))

        print 'Successfully wrote "{}"'.format(path)

    def downscale(self, input_data, only_include=None):
        """
        Downscale a raster or coordinates and elevations from an input file
        :param input_data: a raster file or a .csv in the Climate NA format
        :param only_include: Limit the output to a certain set of models scenarios
        :return: None
        """
        # Parse the input data, which may be a raster or a .csv
        data, data_type = self.parse_input(input_data)

        if data_type == 'raster':
            self.downscale_raster(data, only_include)

        elif data_type == 'csv':
            # Iterate points
            now = time.time()
            iterable = []
            ui = -1
            for lat, lon, elev, name in zip(data['Latitude'], data['Longitude'], data['Elevation'], data['ID2']):
                ui += 1
                iterable.append((lat, lon, self.prism_shape, self.prism_top, self.prism_left, self.prism_csy,
                                 self.prism_csx, self.h5file, name, self.PRISM_PARAMETERS, elev, only_include,
                                 self.scenarios, self.gcm_top, self.gcm_left, self.gcm_csx, self.gcm_csy,
                                 self.normal_top, self.normal_left, self.normal_csx, self.normal_csy, ui))

            # Process in parallel
            cores = mp.cpu_count()
            p = mp.Pool(cores)
            try:
                row_ret = list(p.imap_unordered(downscale_point, iterable))
            except Exception as e:
                import sys
                p.close()
                p.join()
                raise e, None, sys.exc_info()[2]
            else:
                p.close()
                p.join()

            rows = []
            for row in row_ret:
                rows += row

            self.write_csv(rows)

            print "Completed downscaling in {:.0f} seconds".format(time.time() - now)

    # def downscale_raster(self, data, scenarios, parameters):
    #     """
    #     Perform a downscaling operation on scenarios and parameters using the scale of the provided DEM
    #     :param elevation_path: Elevation data used to downscale the climate parameters
    #     :param scenarios: list of tuples of type (scenario, index)
    #     :param parameters: list of parameters to include in output
    #     :return: None
    #     """
    #     # Collect mask and binary data from DEM to avoid repetitive file access
    #     dem.numpyArray()
    #     mask = dem.array == dem.nodata
    #     dem_dump_path = next(tempfile._get_candidate_names())
    #     with open(dem_dump_path, 'wb') as f:
    #         numpy.save(f, dem.array)
    #
    #     outsr = osr.SpatialReference()
    #     outsr.ImportFromWkt(self.projection)
    #     insr = osr.SpatialReference()
    #     insr.ImportFromWkt(dem.projection)
    #     changeCoords = osr.CoordinateTransformation(insr, outsr)
    #     left1, top1, _ = changeCoords.TransformPoint(dem.left, dem.top)
    #     right1, top2, _ = changeCoords.TransformPoint(dem.right, dem.top)
    #     left2, bottom1, _ = changeCoords.TransformPoint(dem.left, dem.bottom)
    #     right2, bottom2, _ = changeCoords.TransformPoint(dem.right, dem.bottom)
    #     bbox = (max(top1, top2), min(bottom1, bottom2), min(left1, left2), max(right1, right2))
    #
    #     # Load the DEM from the PRISM data
    #     prismDEM = raster(self.dem)
    #     prismDEM.changeExtent(bbox, 1E-09)
    #
    #     if not hasattr(prismDEM, 'array'):
    #         prismDEM.numpyArray()
    #     prismDEM.array = prismDEM.array.astype('float32')
    #     prismDEM.datatype = 'float32'
    #
    #     # Generate a transform grid once and pass to all future transform operations
    #     grid = bilinear.generate_grid(prismDEM, elevation_path).astype('float32')
    #     with tempfile.NamedTemporaryFile() as tf:
    #         grid_path = tf.name + '.h5'
    #     self.garbage.append(grid_path)
    #     gridfile = h5py.File(grid_path, mode='w', libver='latest')
    #     gridfile.create_dataset('grid', data=grid, compression='lzf')
    #     gridfile.close()
    #
    #     # Interpolate the prismDEM and save
    #     b = bilinear(prismDEM, elevation_path, transform_grid=grid)
    #     dem.array = b.interpolate(prismDEM.array, prismDEM.nodata)
    #     del grid
    #     dem.nodata = prismDEM.nodata
    #     if self.temp_dir is not None:
    #         with tempfile.NamedTemporaryFile() as tf:
    #             demRast = tf.name + '.pfa'
    #     else:
    #         demRast = _tempfile(self.temp_dir) + '.pfa'
    #     self.garbage.append(demRast)
    #     dem.saveArray(demRast)
    #     del dem
    #
    #     # Remove average parameters, and replace with min and max T, reserve derived params
    #     averages = []
    #     params = []
    #     pas = []
    #     eref = []
    #     dd5 = []
    #     for p in parameters:
    #         p = p.lower().replace('_', '')
    #         if 'ddabvfive' in p:
    #             dd5.append(p)
    #             params.append(p.replace('ddabvfive', 'tmin'))
    #             params.append(p.replace('ddabvfive', 'tmax'))
    #             averages.append(p.replace('ddabvfive', 'tave'))
    #         elif 'pas' in p:
    #             pas.append(p)
    #             params.append(p.replace('pas', 'tmin'))
    #             params.append(p.replace('pas', 'tmax'))
    #             params.append(p.replace('pas', 'ppt'))
    #             averages.append(p.replace('pas', 'tave'))
    #         elif 'eref' in p:
    #             eref.append(p)
    #             params.append(p.replace('eref', 'tmin'))
    #             params.append(p.replace('eref', 'tmax'))
    #             params.append(p.replace('eref', 'rad'))
    #             averages.append(p.replace('eref', 'tave'))
    #         elif 'ave' in p:
    #             averages.append(p)
    #             params.append(p.replace('ave', 'min'))
    #             params.append(p.replace('ave', 'max'))
    #         else:
    #             params.append(p)
    #     params = numpy.unique(params)
    #     averages = numpy.unique(averages)
    #
    #     for scen in scenarios:
    #         scen, index = scen
    #         iterable = []
    #         print "Preparing for multiprocessing of {}".format(scen)
    #         for param in params:
    #             # Load prism data
    #             try:
    #                 prismCol = [i + 2 for i, p in enumerate(self.prismParams) if p.lower().replace('_', '') == param][0]
    #             except IndexError:
    #                 raise ClimateError('Unable to find the parameter {}'.format(param))
    #             prism = numpy.full(self.prism_index.shape, -99990, 'float32')
    #             if 'ppt' in param:
    #                 prism[self.prism_data[:, 0]] = self.prism_data[:, prismCol].astype('float32')
    #             elif 'rad' in param:
    #                 radData = self.prism_data[:, prismCol].astype('float32')
    #                 radData[radData != -99990] /= 10.
    #                 prism[self.prism_data[:, 0]] = radData
    #             else:
    #                 pData = self.prism_data[:, prismCol].astype('float32')
    #                 pData[pData != -99990] /= 100.
    #                 prism[self.prism_data[:, 0]] = pData
    #             prismRast = raster(self.prismTemplate)
    #             prismRast.array = prism.reshape(self.prism_shape)
    #             prismRast.changeExtent(bbox, 1E-09)
    #             if self.temp_dir is None:
    #                 with tempfile.NamedTemporaryFile() as tf:
    #                     prpath = tf.name + '.pfa'
    #             else:
    #                 prpath = _tempfile(self.temp_dir) + '.pfa'
    #             self.garbage.append(prpath)
    #             prismRast.saveArray(prpath)
    #             iterable.append((grid_path, demRast, scen, index, param, bbox, prismDEM,
    #                              elevation_path, output_directory, prpath, self.temp_dir, self.wkdir,
    #                              self.normalShape, self.normalCsx, self.normalCsy, self.normalTop, self.normalLeft,
    #                              self.gcmShape, self.gcmCsx, self.gcmCsy, self.gcmTop, self.gcmLeft, self.projection,
    #                              mask, dem_dump_path))
    #
    #         # ret = [downscale_task(i) for i in iterable]
    #
    #         # Parallelize parameters
    #         cores = mp.cpu_count() - 1
    #         if cores <= 0:
    #             cores = 1
    #         # Initiate pool of workers
    #         p = mp.Pool(cores)
    #         try:
    #             ret = list(p.imap_unordered(downscale_task, iterable))
    #         except Exception as e:
    #             import sys
    #             p.close()
    #             p.join()
    #             raise e, None, sys.exc_info()[2]
    #         else:
    #             p.close()
    #             p.join()
    #
    #         savedParams = []
    #         for r in ret:
    #             savedParams += r
    #
    #         tempRast = raster(elevation_path)
    #         avgIterable = []
    #         iterable = []
    #
    #         for param in averages:
    #             avgIterable.append((tempRast.shape, tempRast.csx, tempRast.csy, tempRast.top, tempRast.left,
    #                             tempRast.projection, savedParams, param, output_directory,
    #                             elevation_path, scen, mask))
    #             # Need to save tave path for subsequent params
    #             scenSaveName = '{}_{}'.format(os.path.basename(os.path.dirname(scen)),
    #                                           os.path.basename(scen).split('.')[0])
    #             savePath = '{}_{}_{}.pfa'.format(os.path.basename(elevation_path).split('.')[0], scenSaveName, param)
    #             savePath = os.path.join(output_directory, savePath)
    #             savedParams.append(savePath)
    #
    #         for param in pas:
    #             iterable.append((tempRast.shape, tempRast.csx, tempRast.csy, tempRast.top, tempRast.left,
    #                             tempRast.projection, savedParams, ('pas', param), output_directory,
    #                             elevation_path, scen, mask))
    #
    #         for param in eref:
    #             iterable.append((tempRast.shape, tempRast.csx, tempRast.csy, tempRast.top, tempRast.left,
    #                             tempRast.projection, savedParams, ('eref', param), output_directory,
    #                             elevation_path, scen, mask))
    #         for param in dd5:
    #             iterable.append((tempRast.shape, tempRast.csx, tempRast.csy, tempRast.top, tempRast.left,
    #                              tempRast.projection, savedParams, ('ddabvfive', param), output_directory,
    #                              elevation_path, scen, mask))
    #
    #         # ret = [avg_task(i) for i in avgIterable]
    #
    #         # Parallelize averages parameters
    #         p = mp.Pool(cores)
    #         try:
    #             ret = list(p.imap_unordered(avg_task, avgIterable))
    #         except Exception as e:
    #             import sys
    #             p.close()
    #             p.join()
    #             raise e, None, sys.exc_info()[2]
    #         else:
    #             p.close()
    #             p.join()
    #
    #         # ret = [derived_task(i) for i in iterable]
    #
    #         # Parallelize remaining parameters
    #         p = mp.Pool(cores)
    #         try:
    #             ret = list(p.imap_unordered(derived_task, iterable))
    #         except Exception as e:
    #             import sys
    #             p.close()
    #             p.join()
    #             raise e, None, sys.exc_info()[2]
    #         else:
    #             p.close()
    #             p.join()

    def clean_garbage(self):
        for ds in self.garbage:
            try:
                os.remove(ds)
            except:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.clean_garbage()


def downscale_point(args):
    """Parallel task for downscaling a point"""
    # Collect the arguments
    # lat, lon, self.prism_shape, self.prism_top, self.prism_left, self.prism_csy, self.prism_csx, self.h5file, name, self.PRISM_PARAMETERS, elev, only_include, self.scenarios, self.gcm_top, self.gcm_left, self.gcm_csx, self.gcm_csy, self.normal_top, self.normal_left, self.normal_csx, self.normal_csy
    lat, lon, prism_shape, prism_top, prism_left, prism_csy, prism_csx,\
    h5file, name, prism_parameters, elev, only_include, scenarios, gcm_top,\
    gcm_left, gcm_csx, gcm_csy, normal_top, normal_left, normal_csx, normal_csy,\
    unique_index = args

    # Collect the grid position
    i = numpy.floor((prism_top - lat) / prism_csy).astype('int64')
    j = numpy.floor((lon - prism_left) / prism_csx).astype('int64')

    # Use a copy of the .h5 file for parallelism
    shutil.copy(h5file, h5file.replace('.h5', '_pid{}.h5'.format(unique_index)))
    h5file = h5file.replace('.h5', '_pid{}.h5'.format(unique_index))

    # Check if within PRISM bounds
    with h5py.File(h5file, libver='latest') as f:
        dem = f['Elevation']
        try:
            assert dem[i, j] != -99990
        except:
            print "Point ({}, {}) is outside of the PRISM data bounds".format(lon, lat)
            return []

        print "Working on {}".format(name)

        # Calculate index offsets to get the window
        i_fr, i_to = max(0, i - 1), min(prism_shape[0], i + 2)
        j_fr, j_to = max(0, j - 1), min(prism_shape[1], j + 2)

        # Bilinear interpolation at point using prismDEM
        prism_elev = bilinear_point(lon, lat, prism_top, prism_left, prism_csx, prism_csy, dem, -99990)
        if prism_elev is None:
            print "Point ({}, {}) is outside of the PRISM elevation data bounds".format(lon, lat)
            return []
        prism_dem = dem[i_fr:i_to, j_fr:j_to]

        # Iterate parameters
        rows = []
        for param in prism_parameters[1:]:  # Avoid elevation at index 0
            # Load respective prism data for the parameter
            prism_data = f[param]

            # Calculate regression parameters (m, b, r2) using prismDEM 9x9 grid and prism parameter 9x9 grid
            m, b, r2 = lapse_rate(prism_dem, prism_data[i_fr:i_to, j_fr:j_to], -99990, -99990)
            lapse_index = m.shape[0] - 2, m.shape[1] - 2
            m, b, r2 = m[lapse_index], b[lapse_index], r2[lapse_index]

            # Bilinear interpolation of prism parameter
            prism_value = bilinear_point(lon, lat, prism_top, prism_left, prism_csx, prism_csy, prism_data, -99990)
            if prism_value is None:
                print "Point ({}, {}) is outside of the PRISM data bounds for {}".format(lon, lat, param)
                continue

            # Calculate the new prism parameter using the regression parameters and the elevation from the input file
            prism_value = prism_value + (((m * (elev - prism_elev)) + b) * r2)

            # Iterate scenarios and apply delta to the prism value
            for scen in scenarios:
                # Skip if only include is not specified
                if only_include is not None and scen not in only_include:
                    continue

                scen_type, scen_name = scen.split('^^^')

                # Load the respective delta coordinate from the GCM grid
                scen_data = f['scens/{}/{}'.format(scen, param)]

                if scen_type == 'gcm':
                    top, left, csx, csy = gcm_top, gcm_left, gcm_csx, gcm_csy
                else:
                    top, left, csx, csy = normal_top, normal_left, normal_csx, normal_csy

                # Perform bilinear interpolation on the delta
                delta = bilinear_point(lon, lat, top, left, csx, csy, scen_data, -99990)
                if delta is None:
                    delta = 0

                # Add the delta to the interpolated prism point
                if 'ppt' in param.lower():
                    # Do not allow precipitation to be less than 0
                    climate_out = max(0, prism_value + (prism_value * (delta / 100.)))
                else:
                    climate_out = prism_value + delta

                # Track output
                rows.append(map(str, (name, lon, lat, elev, scen_name, param[-2:], param[:-2],
                                      '{:.2f}'.format(climate_out))))

    os.remove(h5file)
    return rows


def degree_days_five(month, mean_temp):
    """
    Calculate Degree Days greater than 5 degrees C
    :param month: month (int)
    :param mean_temp: Mean Temp (raster)
    :return: ndarray
    """
    # From:
    # http://journals.plos.org/plosone/article/file?id=info%3Adoi/10.1371/journal.pone.0156720.s003&type=supplementary
    p = [[1, 12.0, 337.5699, 10.0241, 3.36, 30.0966, -140.0, 7.6, 0.986],
           [2, 12.0, 302.8633, 10.006, 3.31, 28.0429, -140.0, 7.5, 0.988],
           [3, 12.0, 363.818, 10.5024, 3.44, 30.1222, -140.0, 10.3, 0.991],
           [4, 12.0, 339.8059, 10.3516, 3.26, 29.4187, -140.0, 7.4, 0.997],
           [5, 12.0, 327.1587, 10.053, 2.79, 30.1473, -140.0, 3.6, 0.999],
           [6, 13.0, 370.5585, 11.1296, 3.13, 29.9647, -150.0, 2.2, 1.0],
           [7, 15.0, 410.0218, 11.6278, 3.12, 30.7456, -150.0, 1.7, 1.0],
           [8, 15.0, 412.2794, 11.6613, 3.13, 30.7429, -150.0, 1.6, 1.0],
           [9, 13.0, 342.8546, 10.5144, 2.96, 29.7243, -145.0, 2.1, 1.0],
           [10, 12.0, 344.9987, 10.2648, 3.19, 30.411, -145.0, 5.0, 0.999],
           [11, 11.0, 304.7169, 9.5882, 3.15, 29.4263, -140.0, 7.0, 0.995],
           [12, 12.0, 341.0866, 10.0869, 3.29, 30.1269, -140.0, 7.0, 0.99]]
    p = p[month - 1]
    s = {key: i for i, key in enumerate(['month', 'k', 'a', 'b', 'To', 'B', 'c', 'sigma', 'R2'])}

    # Read mean temp
    tMean = raster(mean_temp)
    tMean.numpyArray()
    tNodata = tMean.nodata
    tMean = tMean.array

    # Allocate output
    dd = numpy.full(tMean.shape, tNodata, 'float32')

    # Calculate DD
    m = (tMean > p[s['k']]) & (tMean != tNodata)
    dd[m] = p[s['a']] / (1 + (numpy.e**(((tMean[m] - p[s['To']]) / p[s['b']]) * -1)))
    m = (tMean <= p[s['k']]) & (tMean != tNodata)
    dd[m] = p[s['c']] + (p[s['B']] * tMean[m])

    return dd


def precip_as_snow(month, precip, mean_temp, nodata):
    """
    Calculate precipitation as snow
    :param month: month of calculation
    :return:
    """
    p = {1: {'To': -2.5114, 'b': -4.1625, 'sigma': 0.13},
         2: {'To': -1.7031, 'b': -2.6996, 'sigma': 0.13},
         3: {'To': -1.2583, 'b': -1.7860, 'sigma': 0.07},
         4: {'To': -1.4152, 'b': 1.7672, 'sigma': 0.05},
         5: {'To': -2.2797, 'b': 1.4390, 'sigma': 0.01},
         6: {'To': -2.2797, 'b': 1.4390, 'sigma': 0.01},
         7: {'To': -2.1302, 'b': 2.3201, 'sigma': 0.3165},
         8: {'To': -1.9808, 'b': 3.2012, 'sigma': 0.308},
         9: {'To': -1.9808, 'b': 3.2012, 'sigma': 0.01},
         10: {'To': -1.4464, 'b': 2.3486, 'sigma': 0.03},
         11: {'To': -1.4617, 'b': -1.6709, 'sigma': 0.05},
         12: {'To': -1.5327, 'b': -3.0127, 'sigma': 0.12}}
    tMean = raster(mean_temp)
    tMean.numpyArray()
    tNodata = tMean.nodata
    tMean = tMean.array
    precip = raster(precip)
    pNodata = precip.nodata
    precip.numpyArray()
    precip = precip.array
    m = (precip != pNodata) & (tMean != tNodata)
    snow = numpy.full(precip.shape, nodata, 'float32')
    snow[m] = precip[m] * (1 / (1 + numpy.exp(-((tMean[m] - p[month]['To']) / p[month]['b']))))
    return snow


def reference_evaporation(month, mean_temp, min_temp, rad, nodata):
    """
    Calculate reference evaporation using Hargreaves (1985)
    """
    # Number of days in month
    from calendar import monthrange
    d = monthrange(2010, month)[1]

    # Calculate latitude
    meanTemp = raster(mean_temp)
    insr = osr.SpatialReference()
    insr.ImportFromWkt(meanTemp.projection)
    outsr = osr.SpatialReference()
    outsr.ImportFromEPSG(4269)
    changeCoords = osr.CoordinateTransformation(insr, outsr)
    _, top, _ = changeCoords.TransformPoint(meanTemp.left, meanTemp.top)
    _, bottom, _ = changeCoords.TransformPoint(meanTemp.left, meanTemp.bottom)
    lat = numpy.repeat(
        numpy.linspace(bottom, top, meanTemp.shape[0])[::-1].reshape(meanTemp.shape[0], 1),
        meanTemp.shape[1], 1)

    # Load data
    meanTemp.numpyArray()
    meanNodata = meanTemp.nodata
    meanTemp = meanTemp.array
    minTemp = raster(min_temp)
    minTemp.numpyArray()
    minNodata = minTemp.nodata
    minTemp = minTemp.array
    rad = raster(rad)
    rad.numpyArray()
    radNodata = rad.nodata
    rad = rad.array
    m = (meanTemp != meanNodata) & (minTemp != minNodata) & (rad != radNodata)

    # Calculate hargreaves evaporation
    Eref = numpy.full(meanTemp.shape, nodata, 'float32')
    Eref[m] = numpy.where(
        meanTemp[m] >= 0, 0.0023 * d * rad[m] * (meanTemp[m] + 17.8) * ((meanTemp[m] - minTemp[m]) ** 0.5),
        0)
    # Correct for latitude
    Eref[m] *= 1.18 - (0.0065 * lat[m])

    Eref[(Eref < 0) & (Eref != nodata)] = 0

    return Eref


def avg_task(args):
    shape, csx, csy, top, left, projection, savedParams, param, output_directory, elevation_path, scen, mask = args
    climRast = raster()
    climRast.defineNew(shape[1], shape[0], csx, csy, top, left, 'float32', projection)

    # Compute averages
    # Load min and max
    minName = [p for p in savedParams if param.replace('ave', 'min') in os.path.basename(p)][0]
    try:
        minParam = raster(minName)
        minParam.numpyArray()
    except Exception as e:
        raise Exception('Unable to open {} because:\n{}'.format(minName, e))
    maxName = [p for p in savedParams if param.replace('ave', 'max') in os.path.basename(p)][0]
    try:
        maxParam = raster(maxName)
        maxParam.numpyArray()
    except Exception as e:
        raise Exception('Unable to open {} because:\n{}'.format(maxName, e))

    m = (minParam.array != minParam.nodata) & (maxParam.array != maxParam.nodata)
    clim = numpy.full(climRast.shape, climRast.nodata, 'float32')
    clim[m] = (maxParam.array[m] + minParam.array[m]) / 2

    clim[mask] = climRast.nodata
    climRast.array = numpy.round(clim, 1)
    scenSaveName = '{}_{}'.format(os.path.basename(os.path.dirname(scen)),
                                  os.path.basename(scen).split('.')[0])
    savePath = '{}_{}_{}.pfa'.format(os.path.basename(elevation_path).split('.')[0], scenSaveName, param)
    savePath = os.path.join(output_directory, savePath)
    savedParams.append(savePath)
    climRast.saveArray(savePath)
    print "Saved {}".format(os.path.basename(savePath))


def derived_task(args):
    shape, csx, csy, top, left, projection, savedParams, params, output_directory, elevation_path, scen, mask = args
    paramType, param = params
    climRast = raster()
    climRast.defineNew(shape[1], shape[0], csx, csy, top, left, 'float32', projection)

    # Precipitation as snow
    if paramType == 'pas':
        # Load mean and precip
        meanParam = [p for p in savedParams if param.replace('pas', 'tave') in os.path.basename(p)][0]
        precipParam = [p for p in savedParams if param.replace('pas', 'ppt') in os.path.basename(p)][0]
        month = param[-2:]
        try:
            month = int(month)
        except:
            month = int(month[-1])
        snow = precip_as_snow(month, precipParam, meanParam, climRast.nodata)
        snow[mask] = climRast.nodata
        climRast.array = numpy.round(snow, 1)
        scenSaveName = '{}_{}'.format(os.path.basename(os.path.dirname(scen)),
                                      os.path.basename(scen).split('.')[0])
        savePath = '{}_{}_{}.pfa'.format(os.path.basename(elevation_path).split('.')[0], scenSaveName, param)
        savePath = os.path.join(output_directory, savePath)
        climRast.saveArray(savePath)
        print "Saved {}".format(os.path.basename(savePath))

    # Reference Evaporation
    if paramType == 'eref':
        # Load mean and min temp and rad paths
        meanParam = [p for p in savedParams if param.replace('eref', 'tave') in os.path.basename(p)][0]
        minParam = [p for p in savedParams if param.replace('eref', 'tmin') in os.path.basename(p)][0]
        rad = [p for p in savedParams if param.replace('eref', 'rad') in os.path.basename(p)][0]
        month = param[-2:]
        try:
            month = int(month)
        except:
            month = int(month[-1])
        Eref = reference_evaporation(month, meanParam, minParam, rad, climRast.nodata)
        Eref[mask] = climRast.nodata
        climRast.array = numpy.round(Eref, 1)
        scenSaveName = '{}_{}'.format(os.path.basename(os.path.dirname(scen)),
                                      os.path.basename(scen).split('.')[0])
        savePath = '{}_{}_{}.pfa'.format(os.path.basename(elevation_path).split('.')[0], scenSaveName, param)
        savePath = os.path.join(output_directory, savePath)
        climRast.saveArray(savePath)
        print "Saved {}".format(os.path.basename(savePath))

    # Degree days above 5
    if paramType == 'ddabvfive':
        # Load mean temp
        meanParam = [p for p in savedParams if param.replace('ddabvfive', 'tave') in os.path.basename(p)][0]
        # Get month
        month = param[-2:]
        try:
            month = int(month)
        except:
            month = int(month[-1])
        # Compute
        dd = degree_days_five(month, meanParam)
        dd[mask] = climRast.nodata

        # Save output
        climRast.array = numpy.round(dd, 1)
        scenSaveName = '{}_{}'.format(os.path.basename(os.path.dirname(scen)),
                                      os.path.basename(scen).split('.')[0])
        savePath = '{}_{}_{}.pfa'.format(os.path.basename(elevation_path).split('.')[0], scenSaveName, param)
        savePath = os.path.join(output_directory, savePath)
        climRast.saveArray(savePath)
        print "Saved {}".format(os.path.basename(savePath))


def copy_raster(template_raster, data, nodata, temp_dir):
    """
    Create a copy of a raster and return the empty path
    :param template_raster:
    :return:
    """
    if temp_dir is None:
        with tempfile.NamedTemporaryFile() as tf:
            outPath = tf.name + '.tif'
    else:
        outPath = _tempfile(temp_dir) + '.tif'
    r = raster(template_raster)
    if data.shape != r.shape:
        raise ClimateError("Shape of data does not match that of the template raster")
    r.array = data.astype(r.datatype)
    r.nodata = nodata
    r.saveRaster(outPath)
    return outPath


@jit(nopython=True, nogil=True)
def lapse_rate(x, y, xNodata, yNodata):
    """jit-optimized function for lapse regression"""
    # Combination of neighbourhood
    iterator = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            iterator.append((i, j))
    nbrs = []
    for it1 in range(len(iterator)):
        for it2 in range(it1 + 1, len(iterator)):
            nbrs.append(iterator[it1] + iterator[it2])

    iSh = y.shape[0]
    jSh = y.shape[1]
    m = numpy.empty(y.shape, numpy.float32)
    b = numpy.empty(y.shape, numpy.float32)
    r2 = numpy.empty(y.shape, numpy.float32)
    for i in range(iSh):
        for j in range(jSh):
            if x[i, j] == xNodata or y[i, j] == yNodata:
                m[i, j] = xNodata
                b[i, j] = xNodata
                r2[i, j] = xNodata
                continue
            xSet = []
            ySet = []
            for nbr in nbrs:
                i1, j1, i2, j2 = i + nbr[0], j + nbr[1], i + nbr[2], j + nbr[3]
                if (i1 < 0 or i2 < 0 or j1 < 0 or j2 < 0 or i1 > iSh - 1 or
                            i2 > iSh - 1 or j1 > jSh - 1 or j2 > jSh - 1):
                    continue
                x1, x2, y1, y2 = x[i1, j1], x[i2, j2], y[i1, j1], y[i2, j2]
                if x1 != xNodata and x2 != xNodata and y1 != yNodata and y2 != yNodata:
                    xSet.append(x1 - x2)
                    ySet.append(y1 - y2)
            # Compute least squares constants
            n = len(xSet)
            if n < 2:
                m[i, j] = 0
                b[i, j] = 0
                r2[i, j] = 0
                continue
            sprod, sumx, sumy, sumx2, sumy2 = 0., 0., 0., 0., 0.
            for _i in range(n):
                _x, _y = xSet[_i], ySet[_i]
                sprod += _x * _y
                sumx += _x
                sumy += _y
                sumx2 += _x ** 2
                sumy2 += _y ** 2
            denom = (n * sumx2) - (sumx ** 2)
            if denom != 0:
                m[i, j] = ((n * sprod) - (sumx * sumy)) / denom
            else:
                m[i, j] = 0
            b[i, j] = (sumy - (m[i, j] * sumx)) / n
            denom = numpy.sqrt((((n * sumx2) - (sumx ** 2)) * ((n * sumy2) - (sumy ** 2))))
            if denom != 0:
                r2[i, j] = (((n * sprod) - (sumx * sumy)) / denom) ** 2
            else:
                r2[i, j] = 0
    return m, b, r2

def write_and_copy(template, a, nodata):
    driver = gdal.GetDriverByName('MEM')
    outds = driver.CreateCopy('', template, 0)

    band = outds.GetRasterBand(1)
    band.WriteArray(a)
    band.SetNoDataValue(nodata)
    band = None

    return outds


def load_data(ds):
    band = ds.GetRasterBand(1)
    a = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    band = None
    return a, nodata


def downscale_task(args):
    """
    Factory for downscaling each parameter
    :return: list of saved paths
    """
    now = time.time()
    grid_path, demRast, scen, index, param, bbox, prismDEM, elevation_path, output_directory, prismRast, temp_dir,\
    wkdir, normalShape, normalCsx, normalCsy, normalTop, normalLeft, gcmShape, gcmCsx, gcmCsy,\
    gcmTop, gcmLeft, projection, mask, dem_dump_path = args

    # Get some vars started
    savedParams = []
    templateRaster = raster(elevation_path)
    with open(dem_dump_path, 'rb') as f:
        targetRast = numpy.load(f)

    # Load transformation grid
    gridfile = h5py.File(grid_path, libver='latest')
    grid = gridfile['grid']

    # Calculate the lapse rate using linear regression over elevation-climate variable gradients
    pdema = raster(prismDEM)
    prismnd = pdema.nodata
    pdema.numpyArray()
    pdema = pdema.array
    prasta = raster(prismRast)
    prasta.numpyArray()
    prastand = prasta.nodata
    prasta = prasta.array
    mRast, bRast, r2Rast = lapse_rate(pdema, prasta, prismnd, -99990)
    del pdema

    # Create instance of bilinear interpolation to use for prism downscaling
    prismB = bilinear(prismRast, elevation_path, transform_grid=grid)

    # Perform bilinear interpolation on climate variable and load elevation
    # Also dump to disk to save memory
    clim_fo = FileObj(prismB.interpolate(prasta, prastand))
    clim = clim_fo.a
    del prasta
    mRast_fo = FileObj(prismB.interpolate(mRast, prismnd))
    mRast = mRast_fo.a
    bRast_fo = FileObj(prismB.interpolate(bRast, prismnd))
    bRast = bRast_fo.a
    r2Rast_fo = FileObj(prismB.interpolate(r2Rast, prismnd))
    del prismB
    r2Rast = r2Rast_fo.a
    fo6 = raster(demRast)
    demNodata = fo6.nodata
    demRast = fo6.darray
    targetNodata = templateRaster.nodata

    # Adjust using the lapse rates
    ne_mask = '(clim!=prastand)&(mRast!=prismnd)&(bRast!=prismnd)&' \
              '(r2Rast!=prismnd)&(demRast!=demNodata)&(targetRast!=targetNodata)'
    expr = 'clim+(((mRast*(targetRast-demRast))+bRast)*r2Rast)'
    clim = ne.evaluate('where({},{},clim)'.format(ne_mask, expr))

    # Load anomaly data
    anomaly = load_anomaly(scen, index, param, wkdir,
                           normalShape, normalCsx, normalCsy,
                           normalTop, normalLeft, gcmShape, gcmCsx,
                           gcmCsy, gcmTop, gcmLeft, projection)
    if numpy.all(anomaly.array == anomaly.nodata):
        anomaly = numpy.zeros(shape=templateRaster.shape, dtype='float32')
        anomalyNodata = 0
    else:
        b = bilinear(anomaly, elevation_path, transform_grid=grid)
        anomalyNodata = anomaly.nodata
        anomaly_fo = FileObj(b.interpolate(anomaly.array, anomalyNodata))
        del b
        anomaly = anomaly_fo.a

    gridfile.close()

    # Add perturbation
    if 'ppt' in param:
        expr = 'clim+(clim*(anomaly/100.))'
    else:
        expr = 'clim+anomaly'
    clim = ne.evaluate('where((clim!=prastand)&(anomaly!=anomalyNodata),{},clim)'.format(expr))
    if 'ppt' in param:
        clim = ne.evaluate('where((clim!=prastand)&(anomaly!=anomalyNodata)&(clim<0),0,clim)')

    # Save output
    scenSaveName = '{}_{}'.format(os.path.basename(os.path.dirname(scen)),
                                  os.path.basename(scen).split('.')[0])
    savePath = '{}_{}_{}.pfa'.format(os.path.basename(elevation_path).split('.')[0], scenSaveName, param)
    savePath = os.path.join(output_directory, savePath)
    clim[mask] = templateRaster.nodata
    templateRaster.array = numpy.round(clim, 1)
    templateRaster.saveArray(savePath)
    savedParams.append(savePath)
    print "Saved {}: {}".format(os.path.basename(savePath), time.time() - now)
    return savedParams


class FileObj(object):
    def __init__(self, a):
        with tempfile.NamedTemporaryFile() as f:
            path = os.path.join('/rasters/uploads/tmp', f.name)
        self.path = path
        self.file = h5py.File(path, mode='w', libver='latest')
        self.a = self.file.create_dataset('a', data=a, compression='lzf')

    def close(self):
        self.file.close()
        del self.file
        self.a = None

    def open(self):
        self.file = h5py.File(self.path, libver='latest')
        self.a = self.file['a']

    def __del__(self):
        try:
            del self.a
            self.file.close()
            os.remove(self.path)
        except:
            pass


def bilinear_point(x, y, top, left, csx, csy, grid, grid_nodata):
    """Complete bilinear interpolation from an input grid to get the value at x, y"""
    # Record for printouts
    prvx, prvy = numpy.copy(x), numpy.copy(y)

    # Convert x and y to grid coordinates
    y, x = (top - (csy * 0.5) - y) / csy, (x - (left + (csx * 0.5))) / csx
    # Check that the point is within the domain
    if any([x < 0, y < 0, x >= grid.shape[1], y >= grid.shape[0]]):
        print "Coordinate ({}, {}) out of bounds".format(prvx, prvy)
        return None

    # Convert the coordinates to grid indexes
    y0, x0 = numpy.floor(y).astype('int64'), numpy.floor(x).astype('int64')

    # Generate the grid offset coordinates
    x1 = numpy.clip(x0 + 1, 0, grid.shape[1] - 1)
    y1 = numpy.clip(y0 + 1, 0, grid.shape[0] - 1)

    # Collect the grid values
    Ia = grid[y0, x0]
    Ib = grid[y1, x0]
    Ic = grid[y0, x1]
    Id = grid[y1, x1]

    if any([Ia == grid_nodata, Ib == grid_nodata, Ic == grid_nodata, Id == grid_nodata]):
        print "No data present at point ({}, {})".format(prvx, prvy)
        return None

    return (((x1 - x)*(y1 - y) * Ia) +
            ((x1 - x) * (y - y0) * Ib) +
            ((x - x0) * (y1 - y) * Ic) +
            ((x - x0) * (y - y0) * Id))


class bilinear(object):
    def __init__(self, input_raster, template_raster, transform_grid=None):
        """Construct a raster interpolation grid space, and run over any similar grids"""
        # Read rasters
        inrast, template = raster(input_raster), raster(template_raster)

        # Load mesh grid
        if transform_grid is None:
            grid = self.generate_grid(input_raster, template_raster)
        else:
            grid = transform_grid

        self.shape = template.shape

        # Convert coordinates to intersected grid coordinates
        self.x = (grid[0] - inrast.left) / inrast.csx
        self.xfo = FileObj(self.x)
        self.x = self.xfo.a
        self.y = (inrast.top - grid[1]) / inrast.csy
        self.yfo = FileObj(self.y)
        self.y = self.yfo.a

        # Grid lookup as integers for fancy indexing
        self.x0 = numpy.floor(self.x).astype(int)
        self.x1 = self.x0 + 1
        self.y0 = numpy.floor(self.y).astype(int)
        self.y1 = self.y0 + 1

        # Ensure no coordinates outside bounds
        self.x0 = numpy.clip(self.x0, 0, inrast.shape[1] - 1)
        self.fo1 = FileObj(self.x0)
        self.x0 = self.fo1.a
        self.x1 = numpy.clip(self.x1, 0, inrast.shape[1] - 1)
        self.fo2 = FileObj(self.x1)
        self.x1 = self.fo2.a
        self.y0 = numpy.clip(self.y0, 0, inrast.shape[0] - 1)
        self.fo3 = FileObj(self.y0)
        self.y0 = self.fo3.a
        self.y1 = numpy.clip(self.y1, 0, inrast.shape[0] - 1)
        self.fo4 = FileObj(self.y1)
        self.y1 = self.fo4.a

    @staticmethod
    def generate_grid(input_raster, template_raster):
        inrast, template = raster(input_raster), raster(template_raster)
        grid = bilinear.mgrid(template)

        # Check if coordinate systems match
        insr = osr.SpatialReference()
        outsr = osr.SpatialReference()
        insr.ImportFromWkt(inrast.projection)
        outsr.ImportFromWkt(template.projection)
        if not insr.IsSame(outsr):
            # Transform the grid coordinates of the template raster
            inpyproj = pj.Proj(insr.ExportToProj4())
            outpyproj = pj.Proj(outsr.ExportToProj4())
            grid = pj.transform(outpyproj, inpyproj, grid[1].ravel(), grid[0].ravel())
        else:
            # Use grid directly
            grid = (grid[1].ravel(), grid[0].ravel())
        return numpy.array(grid)

    @staticmethod
    def mgrid(rast):
        """return grid coordinate space of raster"""
        top_c = rast.top - (rast.csy * 0.5)
        left_c = rast.left + (rast.csx * 0.5)
        ishape, jshape = rast.shape
        return numpy.mgrid[top_c:top_c - (rast.csy * (ishape - 1)):ishape * 1j,
               left_c:left_c + (rast.csx * (jshape - 1)):jshape * 1j]

    def interpolate(self, im, null):
        """
        Interpolate an image using the inherent grid space
        :param im: image to be interpolated (shape of input raster)
        :return: interpolated array
        """
        # Create pointers to instance attributes so can be evaluated in numexpr
        x, y, y0, y1, x0, x1 = self.x, self.y, self.y0, self.y1, self.x0, self.x1

        # Lookup slices
        Ia = im[y0, x0]
        fo1 = FileObj(Ia)
        Ia = fo1.a
        Ib = im[y1, x0]
        fo2 = FileObj(Ib)
        Ib = fo2.a
        Ic = im[y0, x1]
        fo3 = FileObj(Ic)
        Ic = fo3.a
        Id = im[y1, x1]
        fo4 = FileObj(Id)
        Id = fo4.a

        # Mask to avoid nodata values in calculation
        nd = null
        mask = '(Ia!=nd)&(Ib!=nd)&(Ic!=nd)&(Id!=nd)'

        # Perform interpolation and write to output
        expr = '((x1-x)*(y1-y)*Ia)+((x1-x)*(y-y0)*Ib)+((x-x0)*(y1-y)*Ic)+((x-x0)*(y-y0)*Id)'
        return ne.evaluate('where({},{},nd)'.format(mask, expr)).reshape(self.shape)
