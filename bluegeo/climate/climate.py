"""
Use grids provided in the ClimateNA software to downscale climate data based on elevation
"""
from __future__ import print_function
import os
import time
from calendar import monthrange
from tempfile import gettempdir, _get_candidate_names
import multiprocessing
from osgeo import osr
from numba import jit
import numpy as np
import dask.array as da
from .specs import Specs
from .raster import get_raster_specs, read_array, save_array, dask_array, bilinear


NODATA = -99990
CHUNKS = 1024, 1024


class Climate(object):
    """
    Selectively downscale and summarize climate parameters using a DEM and a
    """

    def __init__(self, climate_grids, dem_path, output_dir):
        """Init using input data

        Arguments:
            climate_grids {str} -- Path to a directory with rasters created using the build_grids function
            dem_path {str} -- Path to an elevation raster used to downscale climate parameters
        """
        self.__dict__.update(get_raster_specs(dem_path))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            print('Created new output directory {}'.format(output_dir))
        self.output_dir = output_dir

        print('Reading input DEM')
        self.mask = read_array(dem_path).mask

        # Record temporary files to enable cleaning
        self.scratch = []

        self.input_dem = dem_path
        self.climate_grids = climate_grids
        # Prepare the PRISM elevation dataset for downscaling
        print('Preparing a raster template of size {}'.format(self.shape))
        self.resampled_prism_dem = self.scratch_file()
        bilinear(os.path.join(climate_grids, 'elevation.tif'), self.input_dem, self.resampled_prism_dem)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clean()

    def __del__(self):
        self.clean()

    def clean(self):
        for f in self.scratch:
            try:
                os.remove(f)
            except:
                pass

    def scratch_file(self):
        """Temporary raster storage"""
        path = os.path.join(gettempdir(), next(_get_candidate_names()) + '.tif')
        self.scratch.append(path)
        return path

    def downscale(self, scenarios, parameters):
        """
        Downscale a set of parameters for a set of scenarios
        """
        # A sequence of parameters is necessary to compute parameters
        # that are required by derived parameters first
        average_params, params, pas_params, eref_params, dd5_params = self.filter_parameters(parameters)

        now = time.time()

        # Load the prism dem into an array in advance
        prism_dem = read_array(
            os.path.join(self.climate_grids, 'elevation.tif'), self.bbox, self.projection
            )[0]

        param_iterable = []
        avg_iterable = []
        pas_iterable = []
        eref_iterable = []
        dd5_iterable = []
        for scen in scenarios:
            param_iterable += [(scen, p, prism_dem, self.__dict__) for p in params]
            avg_iterable += [(scen, p, self.__dict__) for p in average_params]
            pas_iterable += [(scen, p, self.__dict__) for p in pas_params]
            eref_iterable += [(scen, p, self.__dict__) for p in eref_params]
            dd5_iterable += [(scen, p, self.__dict__) for p in dd5_params]

        print('Calculating {} parameters'.format(len(param_iterable)))
        self.async_task(downscale_params, param_iterable)
        print('Calculating {} average parameters'.format(len(avg_iterable)))
        self.async_task(averages, avg_iterable)
        print('Calculating {} precipitation as snow parameters'.format(len(pas_iterable)))
        self.async_task(pas, pas_iterable)
        print('Calculating {} reference evaporation parameters'.format(len(eref_iterable)))
        self.async_task(eref, eref_iterable)
        print('Calculating {} degree days above 5 parameters'.format(len(dd5_iterable)))
        self.async_task(dd5, dd5_iterable)

        print('Completed downscaling operation in {} minutes'.format(round((time.time() - now) / 60), 1))

    @staticmethod
    def async_task(method, iterable):
        # Debugging
        _ = [method(args) for args in iterable]

        # # Initiate pool of workers
        # p = multiprocessing.Pool(multiprocessing.cpu_count())
        # try:
        #     _ = p.map(method, iterable)
        # except Exception as e:
        #     import sys
        #     p.close()
        #     p.join()
        #     raise e, None, sys.exc_info()[2]
        # else:
        #     p.close()
        #     p.join()

    @staticmethod
    def filter_parameters(parameters):
        averages = []
        params = []
        pas = []
        eref = []
        dd5 = []
        for p in parameters:
            p = p.lower().replace('_', '')
            if 'ddabvfive' in p:
                dd5.append(p)
                params.append(p.replace('ddabvfive', 'tmin'))
                params.append(p.replace('ddabvfive', 'tmax'))
                averages.append(p.replace('ddabvfive', 'tave'))
            elif 'pas' in p:
                pas.append(p)
                params.append(p.replace('pas', 'tmin'))
                params.append(p.replace('pas', 'tmax'))
                params.append(p.replace('pas', 'ppt'))
                averages.append(p.replace('pas', 'tave'))
            elif 'eref' in p:
                eref.append(p)
                params.append(p.replace('eref', 'tmin'))
                params.append(p.replace('eref', 'tmax'))
                params.append(p.replace('eref', 'rad'))
                averages.append(p.replace('eref', 'tave'))
            elif 'ave' in p:
                averages.append(p)
                params.append(p.replace('ave', 'min'))
                params.append(p.replace('ave', 'max'))
            else:
                params.append(p)

        params = np.unique(params)
        averages = np.unique(averages)
        return averages, params, pas, eref, dd5


def scratch_file():
    return os.path.join(gettempdir(), next(_get_candidate_names()) + '.tif')


def save_output(data, path, self):
    output_path = os.path.join(self['output_dir'], path)
    data[self['mask']] = NODATA
    save_array(
        output_path, data, NODATA, self['top'], self['left'], self['csx'], self['csy'], self['projection']
        )


def downscale_params(args):
    """Downscale operation

    Arguments:
        args {[type]} -- [description]
    """
    scen, param_name, prism_dem, self = args

    # Load the PRISM parameter
    param, bbox, csx, csy = read_array(os.path.join(
        self['climate_grids'], param_name + '.tif'), self['bbox'], self['projection'])

    # Calculate lapse rates
    m, b, r2 = lapse_rate(prism_dem, param, NODATA, NODATA)

    # Save all four grids so gdal may interpolate them
    projection = osr.SpatialReference()
    projection.ImportFromEPSG(4269)
    projection = projection.ExportToWkt()

    param_path = scratch_file()
    save_array(param_path, param, NODATA, bbox[3], bbox[0], csx, csy, projection)
    m_path = scratch_file()
    save_array(m_path, m, NODATA, bbox[3], bbox[0], csx, csy, projection)
    b_path = scratch_file()
    save_array(b_path, b, NODATA, bbox[3], bbox[0], csx, csy, projection)
    r2_path = scratch_file()
    save_array(r2_path, r2, NODATA, bbox[3], bbox[0], csx, csy, projection)

    # Interpolate all four grids
    param_int_path = scratch_file()
    bilinear(param_path, self['input_dem'], param_int_path)
    os.remove(param_path)
    m_int_path = scratch_file()
    bilinear(m_path, self['input_dem'], m_int_path)
    os.remove(m_path)
    b_int_path = scratch_file()
    bilinear(b_path, self['input_dem'], b_int_path)
    os.remove(b_path)
    r2_int_path = scratch_file()
    bilinear(r2_path, self['input_dem'], r2_int_path)
    os.remove(r2_path)

    # Adjust the normals using the lapse rates - calculate using dask
    param = dask_array(param_int_path, CHUNKS)
    m = dask_array(m_int_path, CHUNKS)
    b = dask_array(b_int_path, CHUNKS)
    r2 = dask_array(r2_int_path, CHUNKS)
    prism_dem = dask_array(self['resampled_prism_dem'], CHUNKS)

    # Downscale the PRISM data
    dem_data = dask_array(self['input_dem'], CHUNKS)
    climate_data = param + ((m * (dem_data - prism_dem)) + b) * r2

    # Load the anomaly data
    scen_path = 'scens_{}_{}.tif'.format(scen.lower(), param_name.lower())
    scen_path = os.path.join(self['climate_grids'], scen_path)
    scen_int_path = scratch_file()
    bilinear(scen_path, self['input_dem'], scen_int_path)
    scen_data = dask_array(scen_int_path, CHUNKS)

    # Perturb
    if 'ppt' in param_name:
        # Precip is a percentage
        climate_data += climate_data * (scen_data / 100.)
        mask = da.ma.getmaskarray(climate_data)
        climate_data = da.where(mask | (climate_data >= 0), climate_data, 0)
    else:
        # Other parameters are deltas
        climate_data += scen_data

    # Round to one decimal place
    climate_data = da.round(climate_data, 1)

    # Compute and save
    save_output(climate_data.compute(), '{}_{}.tif'.format(scen.lower(), param_name.lower()), self)

    # Clean up
    os.remove(param_int_path)
    os.remove(m_int_path)
    os.remove(b_int_path)
    os.remove(r2_int_path)
    os.remove(scen_int_path)


def averages(args):
    """Calculate the average using the downscaled min and max"""
    scen, param, self = args
    min_path = os.path.join(
        self['output_dir'], '{}_{}.tif'.format(scen.lower(), param.lower().replace('ave', 'min'))
        )
    min_data = dask_array(min_path, CHUNKS)

    max_path = os.path.join(
        self['output_dir'], '{}_{}.tif'.format(scen.lower(), param.lower().replace('ave', 'max'))
        )
    max_data = dask_array(max_path, CHUNKS)

    avg_data = da.round((max_data + min_data) / 2., 1)

    save_output(avg_data.compute(), '{}_{}.tif'.format(scen.lower(), param.lower()), self)


def pas(args):
    """Calculate precipitation as snow"""
    scen, param, self = args

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

    mean_temp_path = os.path.join(
        self['output_dir'], '{}_{}.tif'.format(scen.lower(), param.lower().replace('pas', 'tave'))
        )
    mean_temp = dask_array(mean_temp_path, CHUNKS)

    precip_path = os.path.join(
        self['output_dir'], '{}_{}.tif'.format(scen.lower(), param.lower().replace('pas', 'ppt'))
        )
    precip_data = dask_array(precip_path, CHUNKS)

    month = int(param[-2:])
    snow = precip_data * (1 / (1 + da.exp(-((mean_temp - p[month]['To']) / p[month]['b']))))
    snow = da.round(snow, 1)

    save_output(snow.compute(), '{}_{}.tif'.format(scen.lower(), param.lower()), self)


def eref(args):
    """Calculate the reference evaporation"""
    scen, param, self = args

    mean_temp_path = os.path.join(
        self['output_dir'], '{}_{}.tif'.format(scen.lower(), param.lower().replace('eref', 'tave'))
        )
    mean_temp = dask_array(mean_temp_path, CHUNKS)

    min_temp_path = os.path.join(
        self['output_dir'], '{}_{}.tif'.format(scen.lower(), param.lower().replace('eref', 'tmin'))
        )
    min_temp = dask_array(min_temp_path, CHUNKS)

    rad_path = os.path.join(
        self['output_dir'], '{}_{}.tif'.format(scen.lower(), param.lower().replace('eref', 'rad'))
        )
    rad = dask_array(rad_path, CHUNKS)

    # Number of days in month
    month = int(param[-2:])
    d = monthrange(2010, month)[1]

    # Calculate (approximate and fast) latitude
    insr = osr.SpatialReference()
    insr.ImportFromWkt(self['projection'])
    outsr = osr.SpatialReference()
    outsr.ImportFromEPSG(4269)
    changeCoords = osr.CoordinateTransformation(insr, outsr)
    _, top, _ = changeCoords.TransformPoint(self['left'], self['top'])
    _, bottom, _ = changeCoords.TransformPoint(self['left'], self['bottom'])
    latitude = da.repeat(da.linspace(bottom, top, self['shape'][0])[
                         ::-1].reshape(self['shape'][0], 1), self['shape'][1], 1)

    # Calculate hargreaves evaporation
    eref = da.where(mean_temp >= 0, 0.0023 * d * rad * (mean_temp + 17.8) * ((mean_temp - min_temp) ** 0.5), 0)
    # Correct for latitude
    eref *= 1.18 - (0.0065 * latitude)

    eref = da.round(da.where((eref < 0) & (eref != NODATA), 0, eref), 1)

    save_output(eref.compute(), '{}_{}.tif'.format(scen.lower(), param.lower()), self)


def dd5(args):
    """Calculate degree days above 5"""
    scen, param, self = args

    mean_temp_path = os.path.join(
        self['output_dir'], '{}_{}.tif'.format(scen.lower(), param.lower().replace('ddabvfive', 'tave'))
        )
    mean_temp = dask_array(mean_temp_path, CHUNKS)
    month = int(param[-2:])

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

    # Calculate DD
    dd = da.where(
        mean_temp > p[s['k']],
        p[s['a']] / (1 + (np.e**(((mean_temp - p[s['To']]) / p[s['b']]) * -1))),
        NODATA
        )
    dd = da.round(da.where(
        mean_temp <= p[s['k']],
        p[s['c']] + (p[s['B']] * mean_temp),
        dd
        ), 1)

    save_output(dd.compute(), '{}_{}.tif'.format(scen.lower(), param.lower()), self)


def build_grids(data_dir, output_dir):
    """Build rasters from the ClimateNA data distribution

    Arguments:
        data_dir {str} -- directory to save output raster data
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        print('Created new directory at {}'.format(output_dir))

    with open(os.path.join(os.path.join(data_dir, 'prismdat'), 'prism_dv.dat')) as f:
        print('Reading PRISM data')
        prism_data = [line.split(',') for line in f.readlines()[0].split()]

    # Need to fix two lines improperly delimited ?
    rpl1 = prism_data[8107][:50]
    rpl1[-1] = rpl1[-1][:6]
    rpl2 = prism_data[8107][49:]
    rpl2[0] = rpl2[0][6:]
    rpl3 = prism_data[45187][:50]
    rpl3[-1] = rpl3[-1][:6]
    rpl4 = prism_data[45187][49:]
    rpl4[0] = rpl4[0][6:]
    prism_data = np.vstack([prism_data[:8107], rpl1, rpl2,
                            prism_data[8108:45187], rpl3, rpl4,
                            prism_data[45188:]]).astype('int32')

    prism_index = np.unravel_index(prism_data[:, 0], Specs.prism_shape)

    # Save datasets
    print('Saving PRISM datasets')
    template = np.full(Specs.prism_shape, NODATA, 'float32')
    for i, name in enumerate(Specs.prism_parameters):
        template[prism_index] = prism_data[:, i + 1].astype('float32')
        if 'rad' in name.lower():
            # Need to divide radiation data by 10 as radiation is saved as a integer with 1 decimal place
            template[template != NODATA] /= 10.
        elif 'ppt' not in name.lower() and name != 'Elevation':
            # All remaining parameters are saved as integers with 2 decimal places
            template[template != NODATA] /= 100.
        save_array(os.path.join(output_dir, name.lower()), template, NODATA, Specs.prism_top,
                   Specs.prism_left, Specs.prism_csx, Specs.prism_csy, Specs.projection)

    # Collect all of the scenarios
    period_dir = os.path.join(data_dir, 'Perioddat')
    periods = [os.path.join(period_dir, f) for f in os.listdir(period_dir) if f.lower() != 'cru_index.dat']
    period_index = \
        [os.path.join(period_dir, f) for f in os.listdir(period_dir) if f.lower() == 'cru_index.dat'][0]

    gcm_dir = os.path.join(data_dir, 'GCMdat')
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

    # Scenario data
    print("Loading normal deltas")
    load_scenarios(periods, period_index, output_dir, Specs.normal_top, Specs.normal_left,
                   Specs.normal_csx, Specs.normal_csy)
    print("Loading GCM deltas")
    load_scenarios(gcms, gcm_index, output_dir, Specs.gcm_top, Specs.gcm_left, Specs.gcm_csx, Specs.gcm_csy)
    print("Loading annual GCM deltas")
    load_scenarios(annual_gcms, gcm_index, output_dir, Specs.gcm_top,
                   Specs.gcm_left, Specs.gcm_csx, Specs.gcm_csy)


def load_scenarios(paths, index_path, output_dir, top, left, csx, csy):
    # Check that the files exist
    if any([not os.path.isfile(f) for f in paths]):
        raise Exception('Unable to find the file {}'.format([f for f in paths if not os.path.isfile(f)][0]))

    # Open the index data
    if not os.path.isfile(index_path):
        raise ValueError('Cannot find the index path {}'.format(index_path))
    with open(index_path, 'r') as f:
        index = np.array(map(int, [line.strip() for line in f.readlines()[1:]]))

    if 'gcm' in os.path.basename(index_path).lower():
        scen_type = 'gcm'
        shape = Specs.gcm_shape
    else:
        scen_type = 'normal'
        shape = Specs.normal_shape

    # Load a template into memory
    template = np.full(shape, NODATA, 'float32')
    write_locations = np.unravel_index(np.where(~(index == 0)), shape)

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
        for param in Specs.prism_parameters[1:]:  # Avoid elevation from first column
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
                raise ValueError('Parameter {} not found'.format(parameter))
            if len(col) > 1:
                raise ValueError('Found more than one parameter match (found {})'.format(len(col)))
            col = col[0]

            # Write the data
            template[write_locations] = list(map(float, [line[col] for line in lines[1:]]))

            save_array(os.path.join(output_dir, 'scens_{}_{}_{}'.format(
                scen_type.lower(), name.replace(' ', '_').lower(), param.lower()
                )), template, NODATA, top, left, csx, csy, Specs.projection)

        new_progress = '{:.0f}'.format(float(for_prog + 1) / len(paths) * 100.)
        if new_progress != progress and new_progress[-1] in ['5', '0']:
            progress = new_progress
            print(progress + '%')


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
    m = np.empty(y.shape, np.float32)
    b = np.empty(y.shape, np.float32)
    r2 = np.empty(y.shape, np.float32)
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
            denom = np.sqrt((((n * sumx2) - (sumx ** 2)) * ((n * sumy2) - (sumy ** 2))))
            if denom != 0:
                r2[i, j] = (((n * sprod) - (sumx * sumy)) / denom) ** 2
            else:
                r2[i, j] = 0
    return m, b, r2
