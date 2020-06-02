#! /usr/bin/env python

import subprocess
import sys
import os
from tempfile import gettempdir
from shutil import rmtree
from grass_session import Session
from grass.pygrass.modules.shortcuts import raster as graster
from grass.script import core as grass
import grass.script.array as garray
from . import util
from .spatial import *


# Global temporary directory
TEMP_DIR = None


class BlueGrassError(Exception):
    pass


# r. functions
def watershed(dem, flow_direction='SFD', accumulation_path=None, direction_path=None, positive_fd=True,
              change_nodata=True, memory_manage=False):
    """
    Calculate hydrologic routing networks
    :param dem: Digital Elevation Model Raster
    :param flow_direction: Specify one of 'SFD' for single flow direction (D8) or 'MFD' for multiple flow direction (D-inf)
    :param accumulation_path: Path of output flow accumulation dataset if desired
    :param direction_path: Path of output flow direction dataset if desired
    :param positive_fd: Return positive flow direction values only
    :return: Raster instances of flow direction, and flow accumulation, respectively
    """
    # Ensure input Raster is valid and in a gdal format
    dem, garbage = force_gdal(dem)

    # Write flags using args
    flags = ''
    if positive_fd:
        flags += 'a'
    if flow_direction.lower() == 'sfd':
        flags += 's'
    if memory_manage:
        flags += 'm'

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
    with Session(gisdb=TEMP_DIR or gettempdir(), location="location", create_opts=dem):
        graster.external(input=dem, output='surface')
        grass.run_command('r.watershed', elevation='surface', drainage='fd', accumulation='fa', flags=flags)
        graster.out_gdal('fd', format="GTiff", output=dirpath)
        graster.out_gdal('fa', format="GTiff", output=accupath)
    rmtree(os.path.join(TEMP_DIR, 'location'))

    if garbage:
        try:
            os.remove(dem)
        except:
            pass

    # Fix no data values in fa
    if change_nodata:
        fa = Raster(accupath, mode='r+')
        for a, s in fa.iterchunks():
            a[numpy.isnan(a) | numpy.isinf(a) | (a == fa.nodata)] = numpy.finfo('float32').min
            fa[s] = a
        fa.nodataValues = [numpy.finfo('float32').min]
    else:
        fa = Raster(accupath)

    # Return Raster instances
    fd = Raster(dirpath)
    fd.garbage = {'path': dirpath, 'num': 1}
    fa.garbage = {'path': accupath, 'num': 1}
    return fd, fa


def stream_extract(dem, minimum_contributing_area, stream_length=0, accumulation=None):
    """
    Extract streams
    :param dem:
    :param minimum_contributing_area:
    :return:
    """
    # Ensure input Raster is valid and in a gdal format
    dem, dem_garbage = force_gdal(dem)
    if accumulation is not None:
        accumulation, accu_garbage = force_gdal(accumulation)
    else:
        accu_garbage = False

    # Compute threshold using minimum contributing area
    r = Raster(dem)
    threshold = minimum_contributing_area / (r.csx * r.csy)

    stream_path = util.generate_name(dem, 'streams', 'tif')

    # Run grass command
    with Session(gisdb=TEMP_DIR or gettempdir(), location="location", create_opts=dem):
        graster.external(input=dem, output='dem')

        if accumulation is not None:
            graster.external(input=accumulation, output='accumulation')
            grass.run_command('r.stream.extract', elevation='dem',
                              threshold=threshold, stream_raster='streams', stream_length=stream_length,
                              accumulation='accumulation')
        else:
            grass.run_command('r.stream.extract', elevation='dem',
                              threshold=threshold, stream_raster='streams', stream_length=stream_length)
        graster.out_gdal('streams', format="GTiff", output=stream_path)
    rmtree(os.path.join(TEMP_DIR, 'location'))

    if dem_garbage:
        try:
            os.remove(dem)
        except:
            pass

    if accu_garbage:
        try:
            os.remove(accumulation)
        except:
            pass

    # Return Raster instances
    streams = Raster(stream_path)
    streams.garbage = {'path': stream_path, 'num': 1}
    return streams


def stream_order(dem, minimum_contributing_area, stream_order_path=None, method='strahler'):
    """
    Calculated stream order from a DEM using the prescribed method
    :param dem: Digital Elevation Model Raster
    :param minimum_contributing_area: Minimum contributing area to constitute a stream
    :param stream_order_path: Path to output stream order Raster dataset if desired
    :param method: Output stream order type.  Use one of 'strahler', 'horton', 'hack', or 'shreve'
    :return: Raster instance with output stream order values
    """
    # Ensure input Raster is valid and in a gdal format
    dem, garbage = force_gdal(dem)

    # Compute threshold using minimum contributing area
    r = Raster(dem)
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
    with Session(gisdb=TEMP_DIR or gettempdir(), location="location", create_opts=dem):
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
    rmtree(os.path.join(TEMP_DIR, 'location'))

    if garbage:
        try:
            os.remove(dem)
        except:
            pass

    # Return Raster instances
    order = Raster(orderpath)
    order.garbage = {'path': orderpath, 'num': 1}
    return order


def water_outlet(coordinates, dem, direction=None,  basin_path=None, id=None, vectors=False):
    """
    Delineate basins from a list of points
    :param coordinates: Vector or list of coordinate tuples in the form [(x1, y1), (x2, y2),...(xn, yn)]
    :param dem: digital elevation model Raster (if no flow direction surface is available)
    :param direction: flow direction surface (if available)
    :param basin_path: path for output basin Raster
    :return: Raster instance with enumerated basins
    """
    # Check coordinates
    if isinstance(coordinates, str) or isinstance(coordinates, Vector):
        input_vect = Vector(coordinates).transform(Raster(dem).projection)
        coordinates = input_vect.vertices[:, [0, 1]]
        if id is not None:
            try:
                id_print = numpy.uint16(input_vect[id])
                assert numpy.unique(id_print).size == id_print.size
            except:
                raise ValueError('Input ID field must be present,'
                                 ' be numeric,'
                                 ' non-negative,'
                                 ' and have entirely unique values')
        else:
            id_print = numpy.arange(coordinates.shape[0]).astype('uint16') + 1
    else:
        id_print = numpy.arange(len(coordinates)) + 1

    # Use dem if fd not specified
    garbage = False
    if direction is not None:
        fd, garbage = force_gdal(direction)
        fd = Raster(fd)
    else:
        try:
            fd = watershed(dem)[0]
        except RasterError:
            raise BlueGrassError('If a flow direction Raster is not specified, a valid DEM must be specified')

    csx, csy = fd.csx, fd.csy

    with Session(gisdb=TEMP_DIR or gettempdir(), location="location", create_opts=fd.path):
        graster.external(input=fd.path, output="fd")
        # Iterate points and populate output rasters
        areas = []
        index = []
        for _i, coord in enumerate(coordinates):
            i = id_print[_i]
            grass.run_command('r.water.outlet', input="fd",
                              output="b%i" % (i), coordinates=tuple(coord))
            a = garray.array()
            a.read("b%i" % (i))
            m = numpy.where(a == 1)
            area = m[0].shape[0] * (csx * csy)
            areas.append(area)
            print("Basin {} area: {}".format(i, area))
            index.append(m)
    rmtree(os.path.join(TEMP_DIR, 'location'))

    # Allocate output
    outrast = fd.astype('uint16')
    outrast.nodataValues = [numpy.iinfo('uint16').max]

    if garbage:
        try:
            os.remove(fd.path)
        except:
            pass

    areas = numpy.array(areas)
    output = numpy.full(outrast.shape, numpy.iinfo('uint16').max, 'uint16')

    if vectors:
        out_vectors = []
        for i in numpy.arange(areas.shape[0]):
            output[index[i]] = id_print[i]
            outrast[:] = output
            out_vectors.append(outrast.vectorize())
            output = numpy.full(outrast.shape, numpy.iinfo('uint16').max, 'uint16')

        return out_vectors
    else:
        # Write rasters to single dataset
        write_index = numpy.arange(areas.shape[0])[numpy.argsort(areas)][::-1]
        for i in write_index:
            output[index[i]] = id_print[i]
        outrast[:] = output

        # If an output path is specified, save the output
        if basin_path is not None:
            outrast.save(basin_path)
            outrast = Raster(basin_path)

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
    csx, csy = Raster(dem).csx, Raster(dem).csy
    # Ensure input Raster is valid and in a gdal format
    dem, garbage = force_gdal(dem)

    # Ensure the minimum basin area makes sense
    try:
        minarea = float(basin_area) / (csx * csy)  # Number of cells
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

    with Session(gisdb=TEMP_DIR or gettempdir(), location="location", create_opts=dem):
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
    rmtree(os.path.join(TEMP_DIR, 'location'))

    if garbage:
        try:
            os.remove(dem)
        except:
            pass

    basins = Raster(basinpath)
    basins.garbage = {'path': basinpath, 'num': 1}

    return basins


def gwflow(phead, status, hc_x, hc_y, s, top, bottom, **kwargs):
    # Collect kwargs
    type = kwargs.get('type', 'confined')
    dtime = kwargs.get('dtime', 1)
    output_head = kwargs.get('output_head', None)
    output_budget = kwargs.get('output_budget', None)
    maxit = kwargs.get('maxit', 10000)

    # Ensure all input rasters match
    phead = Raster(phead)
    status, hc_x, hc_y, s, top, bottom = [force_gdal(rast.match_raster(phead)) for rast in
                                          map(Raster, [status, hc_x, hc_y, s, top, bottom])]
    phead = force_gdal(phead)

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

    with Session(gisdb=TEMP_DIR or gettempdir(), location="location", create_opts=phead) as gs:
        for name, rast in zip(['phead', 'status', 'hc_x', 'hc_y', 's', 'top', 'bottom'],
                              [phead, status, hc_x, hc_y, s, top, bottom]):
            graster.external(input=rast, output=name)
        grass.run_command('r.gwflow', phead='phead', status='status', hc_x='hc_x', hc_y='hc_y',
                          s='s', top='top', bottom='bottom', type=type, dtime=dtime, output='head',
                          budget='budget', maxit=maxit)
        graster.out_gdal('head', format="GTiff", output=out_head)
        graster.out_gdal('budget', format='GTiff', output=out_budget)
    rmtree(os.path.join(TEMP_DIR, 'location'))

    return Raster(out_head), Raster(out_budget)


def slope_aspect(elevation):
    """
    Calculate slope and aspect
    :param elevation: elevation raster
    :return: slope, aspect rasters
    """
    dem, dem_garbage = force_gdal(elevation)

    slope = util.generate_name(elevation, 'slope', 'tif')
    aspect = util.generate_name(elevation, 'aspect', 'tif')

    with Session(gisdb=TEMP_DIR or gettempdir(), location="location", create_opts=dem) as gs:
        graster.external(dem, output='dem')
        # Calculate slope and aspect
        grass.run_command('r.slope.aspect', elevation='dem', aspect='aspect', slope='slope')
        graster.out_gdal('slope', format="GTiff", output=slope)
        graster.out_gdal('aspect', format="GTiff", output=aspect)
    rmtree(os.path.join(TEMP_DIR, 'location'))

    if dem_garbage:
        try:
            os.remove(dem)
        except:
            pass

    out_slope = Raster(slope)
    out_slope.garbage = {'path': slope, 'num': 1}

    out_aspect = Raster(aspect)
    out_aspect.garbage = {'path': aspect, 'num': 1}

    return out_slope, out_aspect


def sun(elevation, day, step=1, slope=None, aspect=None):
    """
    Calculate global total solar radiation for a given dey (1-365)
    :param dem: Digital elevation model
    :param day: Day (int, 1-365)
    :param step: Time step when computing all-day radiation sums (decimal hours)
    :return: None
    """
    out_sun = util.generate_name(elevation, 'sun', 'tif')
    dem, dem_garbage = force_gdal(elevation)
    slope_garbage = aspect_garbage = False
    if slope is not None:
        slope, slope_garbage = force_gdal(slope)
    if aspect is not None:
        aspect, aspect_garbage = force_gdal(aspect)

    with Session(gisdb=TEMP_DIR or gettempdir(), location="location", create_opts=dem) as gs:
        graster.external(dem, output='dem')
        # Calculate slope and aspect first
        if slope is None or aspect is None:
            grass.run_command('r.slope.aspect', elevation='dem', aspect='aspect', slope='slope')
        else:
            graster.external(aspect, output='aspect')
            graster.external(slope, output='slope')
        grass.run_command('r.sun', aspect='aspect', slope='slope', elevation='dem', glob_rad='rad', day=day, step=step)
        graster.out_gdal('rad', format="GTiff", output=out_sun)
    rmtree(os.path.join(TEMP_DIR, 'location'))

    if dem_garbage:
        try:
            os.remove(dem)
        except:
            pass
    if slope_garbage:
        try:
            os.remove(slope)
        except:
            pass
    if aspect_garbage:
        try:
            os.remove(aspect)
        except:
            pass

    sun = Raster(out_sun)
    sun.garbage = {'path': out_sun, 'num': 1}
    return sun


def lidar(las_file, las_srs_epsg, output_raster, resolution=1, return_type='min'):
    if return_type == 'min':
        return_filter = 'last'  # DTM
    else:
        return_filter = 'first'  # DSM
    with Session(gisdb=TEMP_DIR or gettempdir(), location="location", create_opts=las_srs_epsg) as gs:
        grass.run_command('r.in.lidar', input=las_file, output='outrast',
                          method=return_type, resolution=resolution, return_filter=return_filter, flags='e')
        graster.out_gdal('outrast', format="GTiff", output=output_raster)
    rmtree(os.path.join(TEMP_DIR, 'location'))

    elev = Raster(output_raster)
    elev.garbage = {'path': output_raster, 'num': 1}
    return elev
