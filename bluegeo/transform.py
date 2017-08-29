import pyproj as pj
from raster import *


def bilinear(input_raster, template_raster):
    """
    Transform a raster to match the template raster using bilinear interpolation
    :param input_raster: Raster to be transformed
    :param template_raster: Template raster
    :return: raster instance
    """
    # Read rasters
    inrast, template = raster(input_raster), raster(template_raster)

    # Check if coordinate systems match
    insr = osr.SpatialReference()
    outsr = osr.SpatialReference()
    insr.ImportFromWkt(inrast.projection)
    outsr.ImportFromWkt(template.projection)
    if not insr.IsSame(outsr):
        # Transform the grid coordinates of the template raster
        inpyproj = pj.Proj(insr.ExportToProj4())
        outpyproj = pj.Proj(outsr.ExportToProj4())
        grid = template.mgrid
        grid = pj.transform(outpyproj, inpyproj, grid[1].ravel(), grid[0].ravel())
    else:
        grid = template.mgrid
        grid = (grid[1].ravel(), grid[0].ravel())

    # Convert coordinates to intersected grid coordinates
    x, y = ((grid[0] - inrast.left) / inrast.csx,
            (inrast.top - grid[1]) / inrast.csy)
    del grid

    # Load data values from input raster
    im = inrast.array

    # Allocate output
    outrast = template.empty()

    # Grid lookup as integers for fancy indexing
    x0 = numpy.floor(x).astype(int)
    x1 = x0 + 1
    y0 = numpy.floor(y).astype(int)
    y1 = y0 + 1

    # Ensure no coordinates outside bounds
    x0 = numpy.clip(x0, 0, im.shape[1] - 1)
    x1 = numpy.clip(x1, 0, im.shape[1] - 1)
    y0 = numpy.clip(y0, 0, im.shape[0] - 1)
    y1 = numpy.clip(y1, 0, im.shape[0] - 1)

    # Lookup slices
    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    # Mask to avoid nodata values in calculation
    nd = inrast.nodata
    outnd = template.nodata
    mask = '(Ia!=nd)&(Ib!=nd)&(Ic!=nd)&(Id!=nd)'

    # Perform interpolation and write to output
    expr = '((x1-x)*(y1-y)*Ia)+((x1-x)*(y-y0)*Ib)+((x-x0)*(y1-y)*Ic)+((x-x0)*(y-y0)*Id)'
    outrast[:] = ne.evaluate(
        'where({},{},outnd)'.format(mask, expr)
    ).reshape(outrast.shape)

    return outrast
