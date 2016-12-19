'''
Terrain and Hydrologic routing analysis

Blue Geosimulation, 2016
'''

from raster import *
import util
import numpy.ma as ma
import math
import sys


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
        if 'float' not in self.dtype:
            self = self.astype('float32')
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

    def convergence(self, size=(5, 5)):
        '''
        Compute the relative convergence of flow vectors (uses flow
        accumulation)
        '''
        def eval_conv(a):
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

        # Allocate output
        conv = raster(self, output_descriptor='conv')
        if self.useChunks:
            # Iterate chunks and calculate slope
            for a, s in self.iterchunks(expand=size):
                s_ = truncate_slice(s, size)
                conv[s_] = eval_conv(a)
        else:
            # Calculate over all data
            conv[:] = eval_conv(self.array)


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
            self = self.astype('float32')
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


    def surface_roughness(self, method, size):
        '''
        Compute the roughness of a surface.
        Methods are:
        "std-elev": standard deviation of locally normalized elevation
        '''
        def eval_roughness(a):
            nbrhood = size
            nodatamask = a == nodata
            view = util.getWindowViews(a, nbrhood)
            itop, ibot, jleft, jright = util.getViewInsertLocation(nbrhood)
            # Normalize elevation over neighborhood
            max_ = numpy.zeros(shape=a.shape, dtype='float32')
            min_ = numpy.zeros(shape=a.shape, dtype='float32')
            min_[~nodatamask] = numpy.max(a[~nodatamask])
            # Max/Min filter
            for i in range(nbrhood[0]):
                for j in range(nbrhood[1]):
                    m = (view[i][j] != nodata) & (view[i][j] > max_[itop:ibot,
                                                                    jleft:jright])
                    max_[itop:ibot, jleft:jright][m] = view[i][j][m]
                    m = (view[i][j] != nodata) & (view[i][j] < min_[itop:ibot,
                                                                    jleft:jright])
                    min_[itop:ibot, jleft:jright][m] = view[i][j][m]
            # Calculate mean over normalized elevations
            rge = numpy.zeros(shape=a.shape, dtype='float32')
            rge[~nodatamask] = max_[~nodatamask] - min_[~nodatamask]
            del max_
            mean = numpy.zeros(shape=a.shape, dtype='float32')
            modal = numpy.zeros(shape=a.shape, dtype='int8')
            rgemask = rge != 0
            for i in range(nbrhood[0]):
                for j in range(nbrhood[1]):
                    m = (view[i][j] != nodata) & rgemask[itop:ibot, jleft:jright]
                    mean[itop:ibot, jleft:jright][m] +=\
                        (view[i][j][m] -
                         min_[itop:ibot, jleft:jright][m]) / rge[itop:ibot,
                                                                 jleft:jright][m]
                    modal[itop:ibot, jleft:jright][m] += 1
            repl = ~nodatamask & (modal != 0)
            mean[repl] /= modal[repl]
            # Calculate standard deviation over normalized elevations
            std = numpy.zeros(shape=a.shape, dtype='float32')
            for i in range(nbrhood[0]):
                for j in range(nbrhood[1]):
                    m = (view[i][j] != nodata) & rgemask[itop:ibot, jleft:jright]
                    std[itop:ibot, jleft:jright][m] +=\
                        (((view[i][j][m] - min_[itop:ibot, jleft:jright][m]) /
                          rge[itop:ibot, jleft:jright][m]) - mean[itop:ibot,
                                                                  jleft:jright][m])**2
            std[repl] /= modal[repl]
            std[repl] = numpy.sqrt(std[repl])
            std[~rgemask] = 0
            std[nodatamask] = nodata
            mma[yoff:yoff + win_ysize, xoff:xoff + win_xsize] =\
                std[insislice, insjslice]
            mma.flush()

        # Allocate output
        self.roughness_raster = raster(self.data)
        if self.data.useChunks:
            # Iterate chunks and calculate aspect
            for a, s in self.data.iterchunks(expand=size):
                s_ = truncate_slice(s, size)
                self.roughness_raster[s_] = eval_roughness(a)
        else:
            # Calculate over all data
            self.roughness_raster[1:-1, 1:-1] = eval_roughness(self.data.array)

        # Change outer rows/cols to nodata
        # self.roughness_raster[0, :] = self.nodata
        # self.roughness_raster[-1, :] = self.nodata
        # self.roughness_raster[:, 0] = self.nodata
        # self.roughness_raster[:, -1] = self.nodata


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
