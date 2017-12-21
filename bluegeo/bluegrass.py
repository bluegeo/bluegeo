import subprocess
import sys
import tempfile
import time
import util
from spatial import *


# Global temporary directory
TEMP_DIR = None


class BlueGrassError(Exception):
    pass


class GrassSession(object):
    def __init__(self, src=None, grassbin='grass',
                 persist=False):

        # If temp is specified, use a different temporary directory
        if TEMP_DIR is not None:
            self.tempdir = TEMP_DIR
        else:
            self.tempdir = tempfile.gettempdir()
        self.persist = persist

        # if src
        if type(src) == int:
            # Assume epsg code
            self.location_seed = "EPSG:{}".format(src)
        else:
            # Assume georeferenced vector or raster
            self.location_seed = src

        self.grassbin = grassbin
        # TODO assert grassbin is executable and supports what we need

        startcmd = "{} --config path".format(grassbin)

        # Adapted from
        # http://grasswiki.osgeo.org/wiki/Working_with_GRASS_without_starting_it_explicitly#Python:_GRASS_GIS_7_without_existing_location_using_metadata_only
        p = subprocess.Popen(startcmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise Exception("ERROR: Cannot find GRASS GIS 7 start script ({})".format(startcmd))
        self.gisbase = out.strip('\n')

        self.gisdb = os.path.join(self.tempdir, 'mowerdb')
        self.location = "loc_{}".format(str(time.time()).replace(".","_"))
        self.mapset = "PERMANENT"

        os.environ['GISBASE'] = self.gisbase
        os.environ['GISDBASE'] = self.gisdb

    def gsetup(self):
        path = os.path.join(self.gisbase, 'etc', 'python')
        sys.path.append(path)
        os.environ['PYTHONPATH'] = ':'.join(sys.path)

        import grass.script.setup as gsetup
        gsetup.init(self.gisbase, self.gisdb, self.location, self.mapset)



    def create_location(self):
        try:
            os.stat(self.gisdb)
        except OSError:
            os.mkdir(self.gisdb)

        createcmd = "{0} -c {1} -e {2}".format(
            self.grassbin,
            self.location_seed,
            self.location_path)

        p = subprocess.Popen(createcmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise Exception("ERROR: GRASS GIS 7 start script ({}) because:\n{}".format(createcmd, err))

    @property
    def location_path(self):
        return os.path.join(self.gisdb, self.location)

    def cleanup(self):
        if os.path.exists(self.location_path) and not self.persist:
            shutil.rmtree(self.location_path)
        if 'GISRC' in os.environ:
            del os.environ['GISRC']

    def __enter__(self):
        self.create_location()
        self.gsetup()
        return self

    def __exit__(self, type, value, traceback):
        self.cleanup()


def external(input_raster):
    r = raster(input_raster)
    if r.format == 'HDF5':
        path = util.generate_name(r.path, 'copy', 'tif')
        r.save(path)
    else:
        path = r.path
    return path


# r. functions
def watershed(dem, flow_direction='SFD', accumulation_path=None, direction_path=None, positive_fd=True):
    """
    Calculate hydrologic routing networks
    :param dem: Digital Elevation Model raster
    :param flow_direction: Specify one of 'SFD' for single flow direction (D8) or 'MFD' for multiple flow direction (D-inf)
    :param accumulation_path: Path of output flow accumulation dataset if desired
    :param direction_path: Path of output flow direction dataset if desired
    :param positive_fd: Return positive flow direction values only
    :return: Raster instances of flow direction, and flow accumulation, respectively
    """
    # Ensure input raster is valid and in a gdal format
    dem = external(dem)

    # Write flags using args
    flags = ''
    if positive_fd:
        flags += 'a'
    if flow_direction.lower() == 'sfd':
        flags += 's'

    # Parse output paths
    if accumulation_path is None:
        accupath = util.generate_name(dem, 'acc', 'tif')
    else:
        if accumulation_path.split('.')[-1].lower() != 'tif':
            accupath = accumulation_path + '.tif'
        else:
            accupath = accumulation_path
    if direction_path is None:
        dirpath = util.generate_name(dem, 'dir', 'tif')
    else:
        if direction_path.split('.')[-1].lower() != 'tif':
            dirpath = direction_path + '.tif'
        else:
            dirpath = direction_path

    # Run grass command
    with GrassSession(dem):
        from grass.pygrass.modules.shortcuts import raster as graster
        from grass.script import core as grass
        graster.external(input=dem, output='surface')
        grass.run_command('r.watershed', elevation='surface', drainage='fd', accumulation='fa', flags=flags)
        graster.out_gdal('fd', format="GTiff", output=dirpath)
        graster.out_gdal('fa', format="GTiff", output=accupath)

    # Return raster instances
    return raster(dirpath), raster(accupath)


def stream_extract(dem, minimum_contributing_area):
    """
    Extract streams
    :param dem:
    :param minimum_contributing_area:
    :return:
    """
    # Ensure input raster is valid and in a gdal format
    dem = external(dem)

    # Compute threshold using minimum contributing area
    r = raster(dem)
    threshold = minimum_contributing_area / (r.csx * r.csy)

    stream_path = util.generate_name(dem, 'streams', 'tif')

    # Run grass command
    with GrassSession(dem):
        from grass.pygrass.modules.shortcuts import raster as graster
        from grass.script import core as grass
        graster.external(input=dem, output='dem')

        grass.run_command('r.stream.extract', elevation='dem',
                          threshold=threshold, stream_raster='streams')
        graster.out_gdal('streams', format="GTiff", output=stream_path)

    # Return raster instances
    return raster(stream_path)


def stream_order(dem, minimum_contributing_area, stream_order_path=None, method='strahler'):
    """
    Calculated stream order from a DEM using the prescribed method
    :param dem: Digital Elevation Model raster
    :param minimum_contributing_area: Minimum contributing area to constitute a stream
    :param stream_order_path: Path to output stream order raster dataset if desired
    :param method: Output stream order type.  Use one of 'strahler', 'horton', 'hack', or 'shreve'
    :return: Raster instance with output stream order values
    """
    # Ensure input raster is valid and in a gdal format
    dem = external(dem)

    # Compute threshold using minimum contributing area
    r = raster(dem)
    threshold = minimum_contributing_area / (r.csx * r.csy)

    # Parse output path
    if stream_order_path is None:
        orderpath = util.generate_name(dem, 'order', 'tif')
    else:
        if stream_order_path.split('.')[-1].lower() != 'tif':
            orderpath = stream_order_path + '.tif'
        else:
            orderpath = stream_order_path

    # Run grass command
    with GrassSession(dem):
        from grass.pygrass.modules.shortcuts import raster as graster
        from grass.script import core as grass
        graster.external(input=dem, output='dem')

        grass.run_command('r.stream.extract', elevation='dem',
                          threshold=threshold, stream_raster='streams',
                          direction='fd')

        if method.lower() == 'strahler':
            grass.run_command('r.stream.order', stream_rast='streams',
                              direction='fd', strahler="order")
        elif method.lower() == 'horton':
            grass.run_command('r.stream.order', stream_rast='streams',
                              direction='fd', horton="order")
        elif method.lower() == 'shreve':
            grass.run_command('r.stream.order', stream_rast='streams',
                              direction='fd', shreve="order")
        elif method.lower() == 'hack':
            grass.run_command('r.stream.order', stream_rast='streams',
                              direction='fd', hack="order")
        graster.out_gdal('order', format="GTiff", output=orderpath)

    # Return raster instances
    return raster(orderpath)


def water_outlet(coordinates, dem=None, direction=None,  basin_path=None):
    """
    Delineate basins from a list of points
    :param coordinates: vector or list of coordinate tuples in the form [(x1, y1), (x2, y2),...(xn, yn)]
    :param dem: digital elevation model raster (if no flow direction surface is available)
    :param direction: flow direction surface (if available)
    :param basin_path: path for output basin raster
    :return: raster instance with enumerated basins
    """
    # Check coordinates
    if isinstance(coordinates, basestring) or isinstance(coordinates, vector):
        coordinates = vector(coordinates).transform(raster(dem).projection).vertices[:, [0, 1]]

    # Use dem if fd not specified
    if direction is not None:
        fd = external(direction)
    else:
        try:
            fd, _ = watershed(dem)
        except RasterError:
            raise BlueGrassError('If a flow direction raster is not specified, a valid DEM must be specified')
        fd = fd.path

    csx, csy = raster(fd).csx, raster(fd).csy

    with GrassSession(fd):
        from grass.pygrass.modules.shortcuts import raster as graster
        from grass.script import core as grass
        import grass.script.array as garray
        graster.external(input=fd, output="fd")
        # Iterate points and populate output rasters
        areas = []
        index = []
        for i, coord in enumerate(coordinates):
            grass.run_command('r.water.outlet', input="fd",
                              output="b%i" % (i), coordinates=tuple(coord))
            a = garray.array()
            a.read("b%i" % (i))
            m = numpy.where(a == 1)
            area = m[0].shape[0] * (csx * csy)
            areas.append(area)
            print "Basin {} area: {}".format(i + 1, area)
            index.append(m)

    # Allocate output
    outrast = raster(fd).astype('uint32')
    outrast.nodataValues = [0]

    # Write rasters to single dataset
    output = numpy.zeros(shape=outrast.shape, dtype='uint32')
    areas = numpy.array(areas)
    for c, i in enumerate(numpy.arange(areas.shape[0])[numpy.argsort(areas)][::-1]):
        output[index[i]] = c + 1
    outrast[:] = output

    # If an output path is specified, save the output
    if basin_path is not None:
        outrast.save(basin_path)
        outrast = raster(basin_path)

    return outrast


def watershed_basin(dem, basin_area, basin_path=None, flow_direction='SFD', half_basins=False):
    """
    Delineate basins throughout the entire DEM, using an input basin area to control the number of basins
    :param dem: Digital Elevation Model
    :param basin_area: Minimum basin area used to delineate sub-basins.  Note, output basin areas will vary.
    :param basin_path: Path to output basin dataset if desired
    :param flow_direction: Method of flow direction calculation. Use 'SFD' for single flow direction (D8), or 'MFD'
    for multiple flow direction (D-inf)
    :param half_basins: Split basins into halves along streams if desired.
    :return: Raster instance of enumerated basins
    """
    # Ensure input raster is valid and in a gdal format
    dem = external(dem)

    # Ensure the minimum basin area makes sense
    try:
        minarea = float(basin_area)
    except ValueError:
        raise BlueGrassError('Expected a number for the input basin area, not {}'.format(type(basin_area).__name__))
    if minarea <= 0:
        raise BlueGrassError('Basin area must be greater than 0')

    flags = ''
    if flow_direction.lower() == 'sfd':
        flags += 's'

    # Parse output path
    if basin_path is None:
        basinpath = util.generate_name(dem, 'basin', 'tif')
    else:
        if basin_path.split('.')[-1].lower() != 'tif':
            basinpath = basin_path + '.tif'
        else:
            basinpath = basin_path

    with GrassSession(dem):
        from grass.pygrass.modules.shortcuts import raster as graster
        from grass.script import core as grass
        import grass.script.array as garray
        graster.external(input=dem, output="dem")
        if half_basins:
            grass.run_command('r.watershed', elevation="dem",
                              threshold=minarea, half_basin="b0",
                              flags=flags)
        else:
            grass.run_command('r.watershed', elevation="dem",
                              threshold=minarea, basin="b0",
                              flags=flags)
        graster.out_gdal('b0', format="GTiff", output=basinpath)

    return raster(basinpath)


def gwflow(phead, status, hc_x, hc_y, s, top, bottom, **kwargs):
    # Collect kwargs
    type = kwargs.get('type', 'confined')
    dtime = kwargs.get('dtime', 1)
    output_head = kwargs.get('output_head', None)
    output_budget =kwargs.get('output_budget', None)
    maxit = kwargs.get('maxit', 10000)

    # Ensure all input rasters match
    phead = raster(phead)
    status, hc_x, hc_y, s, top, bottom = [external(rast.match_raster(phead)) for rast in
                                          map(raster, [status, hc_x, hc_y, s, top, bottom])]
    phead = external(phead)

    # Parse output paths
    if output_head is None:
        out_head = util.generate_name(phead, 'gwhead', 'tif')
    else:
        if output_head.split('.')[-1].lower() != 'tif':
            out_head = output_head + '.tif'
        else:
            out_head = output_head
    if output_budget is None:
        out_budget = util.generate_name(phead, 'gwbudget', 'tif')
    else:
        if output_budget.split('.')[-1].lower() != 'tif':
            out_budget = output_budget + '.tif'
        else:
            out_budget = output_budget

    with GrassSession(phead) as gs:
        from grass.pygrass.modules.shortcuts import raster as graster
        from grass.script import core as grass
        for name, rast in zip(['phead', 'status', 'hc_x', 'hc_y', 's', 'top', 'bottom'],
                              [phead, status, hc_x, hc_y, s, top, bottom]):
            graster.external(input=rast, output=name)
        grass.run_command('r.gwflow', phead='phead', status='status', hc_x='hc_x', hc_y='hc_y',
                          s='s', top='top', bottom='bottom', type=type, dtime=dtime, output='head',
                          budget='budget', maxit=maxit)
        graster.out_gdal('head', format="GTiff", output=out_head)
        graster.out_gdal('budget', format='GTiff', output=out_budget)

    return raster(out_head), raster(out_budget)


def sun(elevation, day, step=1):
    """
    Calculate global total solar radiation for a given dey (1-365)
    :param dem: Digital elevation model
    :param day: Day (int, 1-365)
    :param step: Time step when computing all-day radiation sums (decimal hours)
    :return: None
    """
    out_sun = util.generate_name(elevation, 'sun', 'tif')
    dem = external(elevation)

    with GrassSession(dem) as gs:
        from grass.pygrass.modules.shortcuts import raster as graster
        from grass.script import core as grass
        graster.external(dem, output='dem')
        # Calculate slope and aspect first
        grass.run_command('r.slope.aspect', elevation='dem', aspect='aspect', slope='slope')
        grass.run_command('r.sun', aspect='aspect', slope='slope', elevation='dem', glob_rad='rad', day=day, step=step)
        graster.out_gdal('rad', format="GTiff", output=out_sun)

    return raster(out_sun)


def lidar(las_file, las_srs_epsg, output_raster, resolution=1, return_type='min'):
    if return_type == 'min':
        return_filter='last'  # DTM
    else:
        return_filter='first'  #DSM
    with GrassSession(las_srs_epsg) as gs:
        from grass.pygrass.modules.shortcuts import raster as graster
        from grass.script import core as grass
        grass.run_command('r.in.lidar', input=las_file, output='outrast',
                          method=return_type, resolution=resolution, return_filter=return_filter, flags='e')
        graster.out_gdal('outrast', format="GTiff", output=output_raster)

    return raster(output_raster)
