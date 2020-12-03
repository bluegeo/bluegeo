'''
Terrain and Hydrologic routing analysis

Blue Geosimulation, 2017
'''

from .spatial import *
from . import util
import math
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as DummyPool
from numba.core.decorators import jit
from scipy import ndimage, interpolate


class TopoError(Exception):
    pass


class topo(Raster):
    '''
    Topographic analysis using a continuous surface (child of Raster)
    '''
    def __init__(self, surface):
        # Open and change to float if not already
        if 'float' not in Raster(surface).dtype:
            surface = Raster(surface).astype('float32')
            super(topo, self).__init__(surface)
        else:
            super(topo, self).__init__(surface)
        # Change interpolation method unless otherwise specified
        self.interpolationMethod = 'bilinear'

    def slope(self, units='degrees', exaggeration=None):
        '''
        Compute topographic slope over a 9-cell neighbourhood.
        For "units" use "degrees" or "percent rise"
        '''
        def eval_slope(a):
            # Create a mask of nodata values
            nd = self.nodata
            mask = a != nd
            # Apply exaggeration
            if exaggeration is not None:
                a = ne.evaluate('where(mask,a*exaggeration,nd)')

            # Add mask to local dictionary
            local_dict = util.window_local_dict(
                util.get_window_views(mask, (3, 3)), 'm'
            )
            # Add surface data
            local_dict.update(util.window_local_dict(
                util.get_window_views(a, (3, 3)), 'a')
            )
            # Add other variables
            local_dict.update({'csx': self.csx, 'csy': self.csy, 'nd': nd,
                               'pi': math.pi})

            # Compute slope
            bool_exp = '&'.join([key for key in list(local_dict.keys())
                                 if 'm' in key])
            if units == 'degrees':
                calc_exp = '''
                    arctan(sqrt(
                                ((((a0_0+(2*a1_0)+a2_0)-
                                   (a2_2+(2*a1_2)+a0_2))/(8*csx))**2)+
                                ((((a0_0+(2*a0_1)+a0_2)-
                                   (a2_0+(2*a2_1)+a2_2))/(8*csy))**2)
                                )
                           )*(180./pi)
                '''
            else:
                calc_exp = '''
                    sqrt(((((a0_0+(2*a1_0)+a2_0)-
                            (a2_2+(2*a1_2)+a0_2))/(8*csx))**2)+
                         ((((a0_0+(2*a0_1)+a0_2)-
                            (a2_0+(2*a2_1)+a2_2))/(8*csy))**2)
                        )*100
                '''
            return ne.evaluate('where(%s,%s,nd)' % (bool_exp.replace(' ', ''),
                                                    calc_exp.replace(' ', '')),
                               local_dict=local_dict)

        # Allocate output
        slope_raster = self.empty()
        if self.useChunks:
            # Iterate chunks and calculate slope
            for a, s in self.iterchunks(expand=(3, 3)):
                s_ = util.truncate_slice(s, (3, 3))
                slope_raster[s_] = eval_slope(a)
        else:
            # Calculate over all data
            slope_raster[1:-1, 1:-1] = eval_slope(self.array)

        # Change outer rows/cols to nodata
        slope_raster[0, :] = self.nodata
        slope_raster[-1, :] = self.nodata
        slope_raster[:, 0] = self.nodata
        slope_raster[:, -1] = self.nodata
        return topo(slope_raster)


    def aspect(self):
        '''
        Compute aspect as a compass (360 > a >= 0)
        or pi (pi > a >= -pi)
        '''
        def eval_aspect(a):
            # Create a mask of nodata values
            nd = self.nodata
            mask = a != nd

            # Add mask to local dictionary
            local_dict = util.window_local_dict(
                util.get_window_views(mask, (3, 3)), 'm'
            )
            # Add surface data
            local_dict.update(util.window_local_dict(
                util.get_window_views(a, (3, 3)), 'a')
            )
            # Add other variables
            local_dict.update({'csx': self.csx, 'csy': self.csy, 'nd': nd,
                               'pi': math.pi})

            # Compute slope
            bool_exp = '&'.join([key for key in list(local_dict.keys())
                                 if 'm' in key])
            calc_exp = '''
                arctan2((((a0_0+(2*a1_0)+a2_0)-
                          (a2_2+(2*a1_2)+a0_2))/(8*csx)),
                        (((a0_0+(2*a0_1)+a0_2)-
                          (a2_0+(2*a2_1)+a2_2))/(8*csy)))*(-180./pi)+180
            '''
            return ne.evaluate('where(%s,%s,nd)' % (bool_exp, calc_exp),
                               local_dict=local_dict)

        # Allocate output
        aspect_raster = self.empty()
        if self.useChunks:
            # Iterate chunks and calculate aspect
            for a, s in self.iterchunks(expand=(3, 3)):
                s_ = util.truncate_slice(s, (3, 3))
                aspect_raster[s_] = eval_aspect(a)
        else:
            # Calculate over all data
            aspect_raster[1:-1, 1:-1] = eval_aspect(self.array)

        # Change outer rows/cols to nodata
        aspect_raster[0, :] = self.nodata
        aspect_raster[-1, :] = self.nodata
        aspect_raster[:, 0] = self.nodata
        aspect_raster[:, -1] = self.nodata
        return topo(aspect_raster)

    def surface_roughness(self, size=(3, 3)):
        """
        Compute the roughness of a surface.
        Methods are:
        "std-elev": standard deviation of locally normalized elevation
        """
        def eval_roughness(a):
            # Generate nodata mask and get views
            view = util.get_window_views(a, size)
            ic, jc = int((size[0] - 1) // 2), int((size[1] - 1) // 2)
            nd = self.nodata
            mask = view[ic][jc] == nd

            # Normalize elevation over neighborhood
            max_ = numpy.zeros(shape=view[0][0].shape, dtype='float32')
            min_ = numpy.zeros(shape=view[0][0].shape, dtype='float32')
            try:
                min_[~mask] = numpy.max(view[ic][jc][~mask])
            except ValueError:
                return
            # Max/Min filter
            for i in range(size[0]):
                for j in range(size[1]):
                    m = (view[i][j] != nd) & (view[i][j] > max_)
                    max_[m] = view[i][j][m]
                    m = (view[i][j] != nd) & (view[i][j] < min_)
                    min_[m] = view[i][j][m]

            # Calculate mean over normalized elevations
            rge = numpy.zeros(shape=view[ic][jc].shape, dtype='float32')
            rge[~mask] = max_[~mask] - min_[~mask]
            del max_
            mean = numpy.zeros(shape=view[ic][jc].shape, dtype='float32')
            modal = numpy.zeros(shape=view[ic][jc].shape, dtype='int8')
            rgemask = rge != 0
            for i in range(size[0]):
                for j in range(size[1]):
                    m = (view[i][j] != nd) & rgemask
                    mean[m] +=\
                        (view[i][j][m] -
                         min_[m]) / rge[m]
                    modal[m] += 1
            repl = ~mask & (modal != 0)
            mean[repl] /= modal[repl]

            # Calculate standard deviation over normalized elevations
            std = numpy.zeros(shape=view[ic][jc].shape, dtype='float32')
            for i in range(size[0]):
                for j in range(size[1]):
                    m = (view[i][j] != nd) & rgemask
                    std[m] += (((view[i][j][m] - min_[m]) / rge[m]) -
                               mean[m])**2
            std[repl] /= modal[repl]
            std[repl] = numpy.sqrt(std[repl])
            std[~rgemask] = 0
            std[mask] = nd
            return std

        # Allocate output
        surf_rough = self.empty()
        if self.useChunks:
            # Iterate chunks and calculate convergence
            for a, s in self.iterchunks(expand=size):
                s_ = util.truncate_slice(s, size)
                _a = eval_roughness(a)
                if _a is None:
                    a.fill(surf_rough.nodata)
                    _a = a[1:-1, 1:-1]
                surf_rough[s_] = _a
        else:
            # Calculate over all data
            surf_rough[:] = eval_roughness(self.array)

        return topo(surf_rough)

    def align(self, input_raster, interpolation='idw', tolerance=1E-06, max_iter=5000, assume_same=False):
        """
        Align two grids, and correct the z-value using difference in overlapping areas
        :param input_raster: Raster to align with self
        :return: Aligned dataset with coverage from self, and input_raster
        """
        if not assume_same:
            print("Matching rasters")
            # Get Extent of both rasters
            inrast = Raster(input_raster)

            # Grab Extent from input_rast in coordinate system of self
            inrastBbox = util.transform_points([(inrast.left, inrast.top), (inrast.right, inrast.top),
                                                (inrast.left, inrast.bottom), (inrast.right, inrast.bottom)],
                                            inrast.projection, self.projection)
            bbox = (max(self.top, inrastBbox[0][1], inrastBbox[1][1]),
                    min(self.bottom, inrastBbox[2][1], inrastBbox[3][1]),
                    min(self.left, inrastBbox[0][0], inrastBbox[2][0]),
                    max(self.right, inrastBbox[1][0], inrastBbox[3][0]))
            selfChangeExtent = self.clip(bbox)  # Need Extent of both rasters
            inrast = topo(input_raster).match_raster(selfChangeExtent)
        else:
            selfChangeExtent = self
            inrast = topo(input_raster)

        # Allocate output
        outrast = selfChangeExtent.empty()

        def read_data(input, second):
            print("Reading raster data")
            selfData = input.array
            targetData = second.array

            print("Generating Masks")
            m = targetData != second.nodata
            points = m & (selfData != input.nodata)
            xi = numpy.where(m & (selfData == input.nodata))

            print("Creating grids")
            # Only include regions on the edge
            points = numpy.where(
                ~ndimage.binary_erosion(points, structure=numpy.ones(shape=(3, 3), dtype='bool')) & points
            )
            if points[0].size == 0:
                raise TopoError("No overlapping regions found during align")
            # Points in form ((x, y), (x, y)...)
            pointGrid = numpy.fliplr(
                numpy.array(util.indices_to_coords(points, input.top,
                                                    input.left, input.csx,
                                                    input.csy)).T
            )
            # Interpolation grid in form ((x, y), (x, y)...)
            xGrid = numpy.fliplr(
                numpy.array(util.indices_to_coords(xi, input.top, input.left,
                                                    input.csx, input.csy)).T
            )
            grad = selfData[points] - targetData[points]

            return selfData, targetData, xi, pointGrid, xGrid, grad

        def nearest(points, missing):
            distance = numpy.ones(shape=selfChangeExtent.shape, dtype='bool')
            distance[points] = 0
            distance = ndimage.distance_transform_edt(distance, (selfChangeExtent.csy, selfChangeExtent.csx),
                                                        return_indices=True,
                                                        return_distances=False)
            distance = (distance[0][missing], distance[1][missing])
            return distance

        def inverse_distance(x, y, z, pred_x, pred_y):
            # Compute chunk size from memory specification and neighbours
            from multiprocessing import Pool, cpu_count
            chunkSize = int(round(pred_x.shape[0] / (cpu_count() * 4)))
            if chunkSize < 1:
                chunkSize = 1
            chunkRange = list(range(0, pred_x.shape[0] + chunkSize, chunkSize))

            iterator = []
            totalCalcs = 0
            for fr, to in zip(chunkRange[:-1], chunkRange[1:-1] + [pred_x.shape[0]]):
                xChunk = pred_x[fr:to]
                totalCalcs += x.shape[0] * xChunk.shape[0]
                iterator.append( (x, y, z, pred_x, pred_y, numpy.zeros(shape=(to - fr,), dtype='float32'), (fr, to)) )
            print('IDW requires {} calculations'.format(totalCalcs))

            import time
            now = time.time()
            # p = DummyPool(cpu_count())
            p = Pool(cpu_count())
            try:
                iterator = list(p.imap_unordered(idw, iterator))
            except Exception as e:
                import sys
                p.close()
                p.join()
                raise e.with_traceback(sys.exc_info()[2])
            else:
                p.close()
                p.join()
            print("Completed interpolation in %s minutes" % (round((time.time() - now) / 60, 3)))
            return iterator

        if interpolation == 'nearest':
            print("Reading data and generating masks")
            selfData = selfChangeExtent.array
            targetData = inrast.array
            targetDataMask = targetData != inrast.nodata
            selfDataMask = selfData != selfChangeExtent.nodata
            points = selfDataMask & targetDataMask
            xi = numpy.where(targetDataMask & ~selfDataMask)
            if points.sum() == 0:
                raise TopoError("No overlapping regions found during align")
            grad = (selfData - targetData)[nearest(points, xi)]
            selfData[xi] = targetData[xi] + grad

        elif interpolation == 'RectBivariateSpline':
            selfData = selfChangeExtent.array
            targetData = inrast.array

            m = targetData != inrast.nodata
            points = m & (selfData != selfChangeExtent.nodata)
            xi = numpy.where(m & (selfData == selfChangeExtent.nodata))

            # Only include regions on the edge
            block = ~ndimage.binary_erosion(points, structure=numpy.ones(shape=(3, 3), dtype='bool')) & points
            points = numpy.where(block)
            if points[0].size == 0:
                raise TopoError("No overlapping regions found during align")
            x = numpy.arange(0, (points[0].max() + 1) - points[0].min())
            y = numpy.arange(0, (points[1].max() + 1) - points[1].min())
            z = selfData - targetData
            z[~block] = numpy.nan
            z = z[points[0].min():points[0].max() + 1, points[1].min():points[1].max() + 1]

            rbs = interpolate.RectBivariateSpline(x, y, z)
            interp = rbs(xi[0], xi[1])

            selfData[xi] = targetData[xi] + interp

        elif interpolation == 'linear':
            selfData, targetData, xi, pointGrid, xGrid, grad = read_data(selfChangeExtent, inrast)
            interp = interpolate.griddata(pointGrid, grad, xGrid)

            selfData[xi] = targetData[xi] + interp
            
        elif interpolation == 'rbf':
            selfData, targetData, xi, pointGrid, xGrid, grad = read_data(selfChangeExtent, inrast)
            rbfi = interpolate.Rbf(pointGrid[:, 0], pointGrid[:, 1], grad)
            interp = rbfi(xGrid[:, 0], xGrid[:, 1])

            selfData[xi] = targetData[xi] + interp

        elif interpolation == 'idw':
            selfData = selfChangeExtent.array
            targetData = inrast.array

            m = targetData != inrast.nodata
            points = m & (selfData != selfChangeExtent.nodata)
            # Only include regions on the edge
            y, x = numpy.where(~ndimage.binary_erosion(points, structure=numpy.ones(shape=(3, 3), dtype='bool')) & points)
            y, x = y.astype('uint32'), x.astype('uint32')
            z = selfData[(y, x)] - targetData[(y, x)]

            pred_y, pred_x = numpy.where(m & (selfData == selfChangeExtent.nodata))
            pred_y, pred_x = pred_y.astype('uint32'), pred_x.astype('uint32')
            iterator = inverse_distance(x, y, z, pred_x, pred_y)

            # Add output to selfData
            output = numpy.zeros(shape=pred_x.shape, dtype='float32')
            for i in iterator:
                output[i[1][0]:i[1][1]] = i[0]

            selfData[(pred_y, pred_x)] = targetData[(pred_y, pred_x)] + output

        elif interpolation == 'progressive':
            print("Reading data and generating masks")
            selfData = selfChangeExtent.array
            targetData = inrast.array
            targetDataMask = targetData != inrast.nodata
            selfDataMask = selfData != selfChangeExtent.nodata
            points = selfDataMask & targetDataMask
            xi = numpy.where(targetDataMask & ~selfDataMask)

            def mean_filter(a, mask):
                # Add mask to local dictionary
                local_dict = util.window_local_dict(
                    util.get_window_views(mask, (3, 3)), 'm'
                )
                # Add surface data
                local_dict.update(util.window_local_dict(
                    util.get_window_views(a, (3, 3)), 'a')
                )
                # Add other variables
                local_dict.update({'csx': selfChangeExtent.csx, 'csy': selfChangeExtent.csy,
                                    'diag': numpy.sqrt((selfChangeExtent.csx**2) + (selfChangeExtent.csy**2))})

                # Compute mean filter
                bool_exp = '&'.join([key for key in list(local_dict.keys())
                                        if 'm' in key])
                calc_exp = '(a0_0+a0_1+a0_2+a1_0+a1_2+a2_0+a2_1+a2_2)/8'
                return ne.evaluate('where(%s,%s,a1_1)' % (bool_exp.replace(' ', ''),
                                                        calc_exp.replace(' ', '')),
                                    local_dict=local_dict)

            if points.sum() == 0:
                raise TopoError("No overlapping regions found during align")
            grad = selfData - targetData
            grad[xi] = grad[nearest(points, xi)]
            pointReplace = numpy.copy(grad[points])
            mask = (~ndimage.binary_erosion(points, structure=numpy.ones(shape=(3, 3), dtype='bool')) &
                    targetDataMask)
            resid = tolerance + 1
            cnt = 0
            print("Iteratively filtering gradient")
            completed = True
            while resid > tolerance:
                cnt += 1
                prv = numpy.copy(grad)
                grad[1:-1, 1:-1] = mean_filter(grad, mask)
                grad[points] = pointReplace
                resid = numpy.abs(grad - prv).max()
                if cnt == max_iter:
                    print("Maximum iterations reached with a tolerance of %s" % (resid))
                    completed = False
                    break
            if completed:
                print("Completed iterative filter in %s iterations with a residual of %s" % (cnt, resid))

            selfData[xi] = targetData[xi] + grad[xi]

        outrast[:] = selfData
        return outrast


@jit(nopython=True, nogil=True)
def idw(args):
    points, xi, values, output, mask = args
    i_shape = xi.shape[0]
    point_shape = points.shape[0]
    for i in range(i_shape):
        num = 0.0
        denom = 0.0
        for j in range(point_shape):
            w = 1 / numpy.sqrt(
                ((points[j, 0] - xi[i, 0]) ** 2) + ((points[j, 1] - xi[i, 1]) ** 2)
            ) ** 2
            denom += w
            num += w * values[j]
        if denom != 0:
            output[i] = num / denom
    return output, mask


def inverse_distance(pointGrid, xGrid, values):
    # Compute chunk size from memory specification and neighbours
    from multiprocessing import Pool, cpu_count
    chunkSize = int(round(xGrid.shape[0] / (cpu_count() * 4)))
    if chunkSize < 1:
        chunkSize = 1
    chunkRange = list(range(0, xGrid.shape[0] + chunkSize, chunkSize))

    iterator = []
    totalCalcs = 0
    for fr, to in zip(chunkRange[:-1], chunkRange[1:-1] + [xGrid.shape[0]]):
        xChunk = xGrid[fr:to]
        totalCalcs += pointGrid.shape[0] * xChunk.shape[0]
        iterator.append(
            (pointGrid, xChunk, values, numpy.zeros(shape=(to - fr,), dtype='float32'), (fr, to))
        )
    print("Requires {} calculations".format(totalCalcs))

    import time
    now = time.time()
    print("Interpolating")
    p = Pool(cpu_count())
    try:
        iterator = list(p.imap_unordered(idw, iterator))
    except Exception as e:
        import sys
        p.close()
        p.join()
        raise e.with_traceback(sys.exc_info()[2])
    else:
        p.close()
        p.join()
    print("Completed interpolation in %s minutes" % (round((time.time() - now) / 60, 3)))
    return iterator


def correct_surface(surface, points, field):
    """
    Correct a surface to align with a z value from a set of points based on their difference
    :param surface: input surface to correct
    :param points: points used to correct surface
    :param field: field with the z-information
    :return: Raster instance
    """
    # Load datasets
    points = Vector(points)
    surface = Raster(surface)
    # Project points to same spatial reference as the surface
    points = points.transform(surface.projection)
    z = points[field]
    vertices = points.vertices

    # Get indices of aligned cells
    alignCells = util.coords_to_indices((vertices[:, 0], vertices[:, 1]),
                                   surface.top, surface.left, surface.csx, surface.csy, surface.shape)
    # Remove points that are not within the surface Extent
    m = util.intersect_mask((vertices[:, 0], vertices[:, 1]),
                       surface.top, surface.left, surface.csx, surface.csy, surface.shape)
    z = z[m]

    # Create a difference surface using IDW
    dif = numpy.squeeze([surface[i, j] for i, j in zip(alignCells[0], alignCells[1])])  # Slow for lots of points...
    dif = z - dif
    grid = surface.mgrid
    grid = numpy.vstack([grid[1].ravel(), grid[0].ravel()]).T
    iterator = inverse_distance(vertices[:, :2], grid, dif)

    # Create an output Raster and correct with interpolated difference
    out = surface.copy()
    grid = surface.array.ravel()
    for i in iterator:
        a = grid[i[1][0]:i[1][1]]
        m = a != out.nodata
        a[m] += i[0][m]
        grid[i[1][0]:i[1][1]] = a
    out[:] = grid.reshape(out.shape)
    return out


def bare_earth(surface, max_area=65., slope_threshold=50.):
    """
    Create a bare-earth representation of a surface model by removing objects
    :param surface:
    :param max_area:
    :return:
    """
    # Create slope surface to work with
    surface = topo(surface)
    print("Computing gradients and identifying objects")
    with surface.slope() as slopeData:
        # Reclassify high-slope regions
        slopeArray = slopeData.array
        regions = numpy.ones(shape=slopeArray.shape, dtype='bool')
        regions[(slopeArray > slope_threshold) & (slopeArray != slopeData.nodata)] = 0
        del slopeArray

    # Label benches and create index
    bench_labels, _ = ndimage.measurements.label(regions, numpy.ones(shape=(3, 3), dtype='bool'))
    bench_labels = bench_labels.ravel()
    indices = numpy.argsort(bench_labels)
    bins = numpy.bincount(bench_labels)
    bench_labels = numpy.split(indices, numpy.cumsum(bins[bins > 0][:-1]))

    # Label steep regions and create index as dict
    steep_labels, _ = ndimage.measurements.label(~regions, numpy.ones(shape=(3, 3), dtype='bool'))
    steep_index = steep_labels.ravel()
    indices = numpy.argsort(steep_index)
    bins = numpy.bincount(steep_index)
    steep_index = dict(list(zip(numpy.unique(steep_index), numpy.split(indices, numpy.cumsum(bins[bins > 0][:-1])))))

    #  Filter labels by area
    print("Filtering objects")
    regions.fill(0)
    regions = regions.ravel()
    for inds in bench_labels:
        if inds.shape[0] * slopeData.csx * slopeData.csy <= max_area:
            regions[inds] = 1
    regions = regions.reshape(surface.shape)

    # Dilate benches to intersect steep regions
    print("Indexing objects")
    steep_intersect = numpy.unique(steep_labels[
        ~regions & ndimage.binary_dilation(regions, numpy.ones(shape=(3, 3), dtype='bool'))
    ])

    # Add steep regions to regions
    regions = regions.ravel()
    for i in steep_intersect:
        regions[steep_index[i]] = 1
    del steep_labels, steep_index, bench_labels
    regions = regions.reshape(surface.shape)

    # Relabel regions and interpolate grid
    print("Interpolating over objects")
    labels, _ = ndimage.measurements.label(regions, numpy.ones(shape=(3, 3), dtype='bool'))
    labels = labels.ravel()
    indices = numpy.argsort(labels)
    bins = numpy.bincount(labels)
    truncate = False
    if numpy.any(labels == 0):
        truncate = True
    labels = numpy.split(indices, numpy.cumsum(bins[bins > 0][:-1]))
    if truncate:
        del labels[0]

    # Interpolate grid over objects
    new_surface = numpy.pad(surface.array, 1, 'edge')
    iterable = []
    reinsert = []
    for enum, inds in enumerate(labels):
        i, j = numpy.unravel_index(inds, dims=surface.shape)
        i_min, j_min = i.min(), j.min()
        _i, _j = i - i_min + 1, j - j_min + 1
        points = numpy.zeros(shape=(_i.max() + 2, _j.max() + 2), dtype='bool')
        xi = points.copy()
        xi[_i, _j] = 1
        points[~xi & ndimage.binary_dilation(xi, numpy.ones(shape=(3, 3), dtype='bool'))] = 1
        values = new_surface[i_min:i.max() + 3, j_min:j.max() + 3][points]
        m, b = numpy.linalg.solve([[0, 1.], [points.shape[0] - 1, 1.]],
                                  [surface.csy * (points.shape[0] - 1), 0])
        points = numpy.vstack(numpy.where(points)[::-1]).T.astype('float32')
        points[:, 0] *= surface.csx
        points[:, 1] = (points[:, 1] * m) + b
        wxi = numpy.where(xi)
        xi = (wxi[1] * surface.csx, (wxi[0] * m) + b)
        reinsert.append(wxi)
        iterable.append((points, xi, values, surface.nodata, (i_min, i.max() + 3, j_min, j.max() + 3, enum)))

    def perform_interpolation(args):
        points, xi, values, nodata, flowthrough = args
        return interpolate.griddata(points, values, xi, fill_value=nodata), flowthrough

    from multiprocessing.dummy import Pool, cpu_count
    p = Pool(cpu_count())
    try:
        ret = p.imap_unordered(perform_interpolation, iterable)
    except Exception as e:
        import sys
        p.close()
        p.join()
        raise e.with_traceback(sys.exc_info()[2])
    else:
        p.close()
        p.join()

    print("Applying changes to output Raster")
    for values, inds in ret:
        new_surface[inds[0]:inds[1], inds[2]:inds[3]][reinsert[inds[4]]] = values

    # Return new Raster
    out = surface.copy()
    out[:] = new_surface[1:-1, 1:-1]
    return out


@jit(nopython=True, nogil=True)
def idw(args):
    x, y, z, pred_x, pred_y, output, mask = args
    i_shape = pred_x.shape[0]
    point_shape = x.shape[0]
    for i in range(i_shape):
        num = 0.0
        denom = 0.0
        for j in range(point_shape):
            w = 1 / numpy.sqrt(
                ((x[j] - pred_x[i]) ** 2.) + ((y[j] - pred_y[i]) ** 2.)
            ) ** 2
            denom += w
            num += w * z[j]
        output[i] = num / denom
    return output, mask
