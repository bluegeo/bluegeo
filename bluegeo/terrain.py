'''
Terrain and Hydrologic routing analysis

Blue Geosimulation, 2016
'''

from raster import *
import util
import numpy.ma as ma
import math
import sys
from scipy import ndimage


# Put a handle on the mower
if sys.platform.startswith('linux'):
    from mower import GrassSession
    grassbin = '/usr/local/bin/grass70'
# elif sys.platform.startswith('win'):
#     # Ad hoc adaptation of mower for testing in a win env
#     from mower_win import GrassSession
#     grassbin = r'C:\Program Files\GRASS GIS 7.0.5\grass70.bat'


class watershed(raster):
    '''
    Topographic routing and watershed delineation primarily using grass
    '''
    def __init__(self, surface):
        # Open and change to float if not already
        if isinstance(surface, raster):
            self.__dict__.update(surface.__dict__)
        else:
            super(watershed, self).__init__(surface)
        # Change interpolation method unless otherwise specified
        self.interpolationMethod = 'bilinear'

    def create_external_datasource(self):
        path = self.generate_name('copy', 'tif', True)
        self.save_gdal_raster(path)
        return path

    def route(self):
        '''
        Return a single (D8) flow direction from 1 to 8, and positive flow
        accumulation surface
        '''
        # Save a new raster if the data source format is not gdal
        if self.format != 'gdal':
            external = self.create_external_datasource()
        else:
            external = self.path

        # Get paths for the outputs
        fd_outpath = self.generate_name('fl_dir', 'tif', True)
        fa_outpath = self.generate_name('fl_acc', 'tif', True)

        # Perform analysis using grass session
        with GrassSession(external, grassbin=grassbin, persist=False) as gs:
            from grass.pygrass.modules.shortcuts import raster as graster
            from grass.script import core as grass
            graster.external(input=external, output='surface')
            grass.run_command('r.watershed', elevation='surface',
                              drainage='fd', accumulation='fa', flags="s")
            graster.out_gdal('fd', format="GTiff", output=fd_outpath)
            graster.out_gdal('fa', format="GTiff", output=fa_outpath)

        return watershed(fd_outpath), watershed(fa_outpath)

    def convergence(self, size=(11, 11)):
        '''
        Compute the relative convergence of flow vectors (uses directions 1 to
        8, which are derived from flow direction)
        '''
        def eval_conv(a):
            nd = self.nodata
            mask = (a > 0) & (a != nd)

            # Convert a into angles
            x, y = numpy.mgrid[0:self.csy * 2:3j, 0:self.csx * 2:3j]
            ang = (numpy.arctan2(y - self.csy, x - self.csx) * -1) + numpy.pi
            a = ne.evaluate('where(mask,a-1,0)')
            a = ang[(0, 0, 0, 1, 2, 2, 2, 1), (2, 1, 0, 0, 0, 1, 2, 2)][a]
            a[~mask] = nd

            # Get neighbours as views and create output
            b = window_local_dict(get_window_views(a, size), 'a')
            x, y = numpy.mgrid[0:(a.shape[0] - 1) * self.csy:a.shape[0] * 1j,
                               0:(a.shape[1] - 1) * self.csx:a.shape[1] * 1j]
            b.update(window_local_dict(get_window_views(x, size), 'x'))
            b.update(window_local_dict(get_window_views(y, size), 'y'))
            pi = numpy.pi
            b.update({'pi': pi, 'nd': nd})
            c = '%s_%s' % ((size[0] - 1) / 2, (size[1] - 1) / 2)
            conv = numpy.zeros(shape=b['a%s' % c].shape, dtype='float32')

            # Iterate neighbours and compute convergence
            size_scale = (size[0] * size[1]) - 1
            for i in range(size[0]):
                for j in range(size[1]):
                    if i == int(c[0]) and j == int(c[2]):
                        continue
                    at2 = ne.evaluate('where(a%i_%i!=nd,abs(((arctan2(y%i_%i-'
                                      'y%s,x%i_%i-x%s)*-1)+pi)-a%i_%i),nd)' %
                                      (i, j, i, j, c, i, j, c, i, j),
                                      local_dict=b)
                    conv = ne.evaluate('where(at2!=nd,conv+((where(at2>pi,(2*'
                                       'pi)-at2,at2)/pi)/size_scale),conv)')
            conv[b['a%s' % c] == nd] = nd
            return conv

        # Allocate output
        conv = raster(self, output_descriptor='conv')
        conv = conv.astype('float32')
        if self.useChunks:
            # Iterate chunks and calculate convergence
            for a, s in self.iterchunks(expand=size):
                s_ = truncate_slice(s, size)
                conv[s_] = eval_conv(a)
        else:
            # Calculate over all data
            conv[:] = eval_conv(self.array)

        return watershed(conv)

    def stream_slope(self, dem, min_contrib_area=1E6, iterations=1):
        '''
        Compute the slope from cell to cell in streams with a minimum
        contributing area.  Increase iterations to evaluate higher order
        change. Seeded by flow accumulation, and requires a dem.
        '''
        output = raster(streams)
        a = output.load('r')
        m = a != streams.nodata[0]
        output.changeDataType('float32')
        a = output.load('r+')
        elev = dem.load('r')
        # Compute stream slope
        inds = numpy.where(m)
        diag = math.sqrt(streams.csx**2 + streams.csy**2)
        run = numpy.array([[diag, streams.csy, diag],
                           [streams.csx, 1, streams.csx],
                           [diag, streams.csy, diag]])

        def compute(i, j):
            i_, _i, j_, _j = util.getSlice(i, j, elev.shape)
            base = elev[i, j]
            rise = numpy.abs(base - elev[i_:_i, j_:_j][m[i_:_i, j_:_j]])
            run_ = run[m[i_:_i, j_:_j]]
            run_ = run_[rise != 0]
            rise = rise[rise != 0]
            if run_.size == 0:
                return 0
            else:
                return numpy.mean(numpy.degrees(numpy.arctan(rise / run_)))
        slopefill = [compute(inds[0][i], inds[1][i])
                     for i in range(inds[0].shape[0])]
        a[m] = slopefill
        a.flush()
        del a
        return output

    def alluvium(self, dem, min_contrib_area=1E6, slope_thresh=8,
                 stream_slope_thresh=1.5):
        '''
        Use the derivative of stream slope to determine regions of
        aggradation to predict alluvium deposition.  Streams above the
        "min_contrib_area" will be used to seed the delineation, and
        slopes exceeding "slope_thresh" will be avoided.
        Seeded by flow accumulation and requires a dem.
        '''
        # Reclassify flow accumulation into streams and create seed points
        a = self.array
        contrib = int(min_contrib_area / (self.csx *self.csy))
        streams = a >= contrib
        seeds = set(zip(*numpy.where(a == contrib)))
        track = numpy.zeros(shape=a.shape, dtype='bool')

        # Recursively propagate downstream and delineate alluvium
        # Get some parameters
        ish, jsh = a.shape
        diag = numpy.sqrt(self.csx**2 + self.csy**2)
        run = numpy.array([[diag, self.csy, diag],
                           [self.csx, 1, self.csx],
                           [diag, self.csy, diag]])
        # Compute slope
        with topo(dem).slope() as slope_output:
            slope = slope_output.array
        # Load elevation data into memory
        dem = raster(dem).array
        while True:
            try:
                seed = seeds.pop()
            except:
                break
            s = (slice(max(0, min(seed[0] - 1, ish - 1)),
                       max(0, min(seed[0] + 1, ish - 1))),
                 slice(max(0, min(seed[1] - 1, jsh - 1)),
                       max(0, min(seed[1] + 1, jsh - 1))))
            mask = streams[s] & ~track[s]
            if mask.sum() > 0:
                # Stream exists
                sl = numpy.mean(numpy.abs(dem[seed] - dem[s][mask]) /
                                run[mask])
                # High slope: erosion- directed propagation at higher slopes
                g = dem[seed] - dem[s]
                mask = g == g.max()
                if sl > stream_slope_thresh:
                    mild = False
                    # Create a mask with correct gradient directions
                    mask = ndimage.binary_dilation(mask)
                    mask = ~track[s] & mask & (slope[s] < slope_thresh)
                # Low slope: aggradation- fan outwards at shallower slopes
                else:
                    mild = True
                    mask = ndimage.binary_dilation(
                        mask, structure=numpy.ones(shape=(3, 3), dtype='bool')
                    )
                    mask = ~track[s] & mask & (slope[s] < (slope_thresh / 2))
            else:
                # Outside of stream or already tracked
                g = dem[seed] - dem[s]
                mask = g == g.max()
                if mild:
                    # Create a mask with correct gradient directions
                    mask = ndimage.binary_dilation(mask)
                    mask = ~track[s] & mask & (slope[s] < slope_thresh)
                else:
                    mask = ndimage.binary_dilation(
                        mask, structure=numpy.ones(shape=(3, 3), dtype='bool')
                    )
                    mask = ~track[s] & mask & (slope[s] < (slope_thresh / 2))
            # Update track and stack with result
            track[mask] = 1
            s_i, s_j = numpy.where(mask)
            s_i += s[0].start
            s_j += s[1].start
            seeds.update(zip(s_i, s_j))

        alluv_out = raster(self, output_descriptor='slope').astype('uint8')
        alluv_out[:] = track.astype('uint8')
        alluv_out.nodataValues = [0]
        return watershed(alluv_out)


class topo(raster):
    '''
    Topographic analysis using a continuous surface (child of raster)
    '''
    def __init__(self, surface):
        # Open and change to float if not already
        if isinstance(surface, raster):
            self.__dict__.update(surface.__dict__)
        else:
            super(topo, self).__init__(surface)
        if 'float' not in self.dtype:
            out = self.astype('float32')
            self.__dict__.update(out.__dict__)
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
            local_dict = window_local_dict(
                get_window_views(mask, (3, 3)), 'm'
            )
            # Add surface data
            local_dict.update(window_local_dict(
                get_window_views(a, (3, 3)), 'a')
            )
            # Add other variables
            local_dict.update({'csx': self.csx, 'csy': self.csy, 'nd': nd,
                               'pi': math.pi})

            # Compute slope
            bool_exp = '&'.join([key for key in local_dict.keys()
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
        slope_raster = raster(self, output_descriptor='slope')
        if self.useChunks:
            # Iterate chunks and calculate slope
            for a, s in self.iterchunks(expand=(3, 3)):
                s_ = truncate_slice(s, (3, 3))
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
            local_dict = window_local_dict(
                get_window_views(mask, (3, 3)), 'm'
            )
            # Add surface data
            local_dict.update(window_local_dict(
                get_window_views(a, (3, 3)), 'a')
            )
            # Add other variables
            local_dict.update({'csx': self.csx, 'csy': self.csy, 'nd': nd,
                               'pi': math.pi})

            # Compute slope
            bool_exp = '&'.join([key for key in local_dict.keys()
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
        aspect_raster = raster(self, output_descriptor='aspect')
        if self.useChunks:
            # Iterate chunks and calculate aspect
            for a, s in self.iterchunks(expand=(3, 3)):
                s_ = truncate_slice(s, (3, 3))
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
        '''
        Compute the roughness of a surface.
        Methods are:
        "std-elev": standard deviation of locally normalized elevation
        '''
        def eval_roughness(a):
            # Generate nodata mask and get views
            view = get_window_views(a, size)
            ic, jc = (size[0] - 1) / 2, (size[1] - 1) / 2
            nd = self.nodata
            mask = view[ic][jc] == nd

            # Normalize elevation over neighborhood
            max_ = numpy.zeros(shape=view[0][0].shape, dtype='float32')
            min_ = numpy.zeros(shape=view[0][0].shape, dtype='float32')
            min_[~mask] = numpy.max(view[ic][jc][~mask])
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
        surf_rough = raster(self, output_descriptor='srf_rgh')
        if self.useChunks:
            # Iterate chunks and calculate convergence
            for a, s in self.iterchunks(expand=size):
                s_ = truncate_slice(s, size)
                surf_rough[s_] = eval_roughness(a)
        else:
            # Calculate over all data
            surf_rough[:] = eval_roughness(self.array)

        return topo(surf_rough)


# Supplementary functions
def truncate_slice(s, size):
    i, j = size
    ifr = int(math.ceil((i - 1) / 2.))
    jfr = int(math.ceil((j - 1) / 2.))
    ito = int((i - 1) / 2)
    jto = int((j - 1) / 2)
    s_ = s[0].start + ifr
    s__ = s[1].start + jfr
    _s = s[0].stop - ito
    __s = s[1].stop - jto
    return (slice(s_, _s), slice(s__, __s))

def get_window_views(a, size):
    '''
    Get a "shaped" list of views into "a" to retrieve
    neighbouring cells in the form of "size."
    Note: asymmetrical shapes will be propagated
        up and left.
    '''
    i_offset = (size[0] - 1) * -1
    j_offset = (size[1] - 1) * -1
    output = []
    for i in range(i_offset, 1):
        output.append([])
        _i = abs(i_offset) + i
        if i == 0:
            i = None
        for j in range(j_offset, 1):
            _j = abs(j_offset) + j
            if j == 0:
                j = None
            output[-1].append(a[_i:i, _j:j])
    return output

def window_local_dict(views, prefix='a'):
    '''
    Create a local dictionary with variable names to craft a numexpr
    expression using offsets of a moving window
    '''
    return {'%s%s_%s' % (prefix, i, j): views[i][j]
            for i in range(len(views))
            for j in range(len(views[i]))}
