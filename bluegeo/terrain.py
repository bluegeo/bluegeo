'''
Terrain and Hydrologic routing analysis

Blue Geosimulation, 2016
'''

from raster import *
import util
import math
from scipy import ndimage

try:
    from bluegrass import GrassSession
except ImportError:
    print "Warning: Grass functions not available"


class WatershedError(Exception):
    pass


class TopoError(Exception):
    pass


class watershed(raster):
    '''
    Topographic routing and watershed delineation primarily using grass
    '''
    def __init__(self, surface, tempdir=None, grassdiskswap=False):
        # Open and change to float if not already
        self.tempdir = tempdir
        self.useSwap = grassdiskswap
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
        fd_outpath = self.generate_name('fl_dir', 'tif')
        fa_outpath = self.generate_name('fl_acc', 'tif')

        # Perform analysis using grass session
        with GrassSession(external, temp=self.tempdir):
            from grass.pygrass.modules.shortcuts import raster as graster
            from grass.script import core as grass
            graster.external(input=external, output='surface')
            flags = "s"
            if self.useSwap:
                flags += "m"
            grass.run_command('r.watershed', elevation='surface',
                              drainage='fd', accumulation='fa', flags=flags)
            graster.out_gdal('fd', format="GTiff", output=fd_outpath)
            graster.out_gdal('fa', format="GTiff", output=fa_outpath)

        return watershed(fd_outpath, tempdir=self.tempdir), watershed(fa_outpath, tempdir=self.tempdir)

    def convergence(self, size=(11, 11), fd=None):
        '''
        Compute the relative convergence of flow vectors (uses directions 1 to
        8, which are derived from flow direction)
        '''
        def eval_conv(a):
            nd = fd.nodata
            mask = (a > 0) & (a != nd)

            # Convert a into angles
            x, y = numpy.mgrid[0:self.csy * 2:3j, 0:self.csx * 2:3j]
            ang = (numpy.arctan2(y - self.csy, x - self.csx) * -1) + numpy.pi
            a = ne.evaluate('where(mask,a-1,0)')
            a = ang[(0, 0, 0, 1, 2, 2, 2, 1), (2, 1, 0, 0, 0, 1, 2, 2)][a]
            a[~mask] = nd

            # Get neighbours as views and create output
            b = util.window_local_dict(util.get_window_views(a, size), 'a')
            x, y = numpy.mgrid[0:(a.shape[0] - 1) * self.csy:a.shape[0] * 1j,
                               0:(a.shape[1] - 1) * self.csx:a.shape[1] * 1j]
            b.update(util.window_local_dict(util.get_window_views(x, size), 'x'))
            b.update(util.window_local_dict(util.get_window_views(y, size), 'y'))
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

        # Calculate fa if not specified
        if fd is None:
            fa, fd = self.route()
        else:
            fd = raster(fd)
            if 'int' not in fd.dtype:
                fd = fd.astype('int32')
        # Allocate output
        conv = self.copy(True, 'conv')
        if fd.useChunks:
            # Iterate chunks and calculate convergence
            for a, s in fd.iterchunks(expand=size):
                s_ = util.truncate_slice(s, size)
                conv[s_] = eval_conv(a).astype('float32')
        else:
            # Calculate over all data
            conv[:] = eval_conv(fd.array)

        return watershed(conv, tempdir=self.tempdir)

    def stream_order(self, min_contrib_area):
        '''
        Return streams with a contributing area greate than the specified
        threshold.  The resulting dataset includes strahler order for each
        stream
        '''
        # Save a new raster if the data source format is not gdal
        if self.format != 'gdal':
            external = self.create_external_datasource()
        else:
            external = self.path
        # Create output path
        str_path = self.generate_name('streams', 'tif')

        with GrassSession(external, temp=self.tempdir):
            from grass.pygrass.modules.shortcuts import raster as graster
            from grass.script import core as grass
            graster.external(input=external, output='dem')

            # Compute flow accumulation threshold based on min area
            thresh = min_contrib_area / (self.csx * self.csy)
            grass.run_command('r.stream.extract', elevation='dem',
                              threshold=thresh, stream_raster='streams',
                              direction='fd')
            grass.run_command('r.stream.order', stream_rast='streams',
                              direction='fd', strahler="strlr")
            graster.out_gdal('strlr', format="GTiff", output=str_path)

        return watershed(str_path, tempdir=self.tempdir)

    def stream_reclass(self, fa, min_contrib_area):
        '''
        Reclassify a flow accumulation dataset
        '''
        fa_rast = raster(fa)
        st = fa_rast.astype('uint8')
        st[:] = (fa_rast.array > (min_contrib_area /
                                  (st.csx * st.csy))).astype('uint8')
        st.nodataValues = [0]
        return st

    def stream_slope(self, iterations=1, streams=None, min_contrib_area=None,
                     fa=None, units='degrees'):
        '''
        Compute the slope from cell to cell in streams with a minimum
        contributing area.  If streams are specified, they will not be
        computed.
        '''
        if streams is not None:
            with self.match_raster(streams) as dem:
                elev = dem.array
            strms = raster(streams)
            m = strms.array != strms.nodata
        else:
            if min_contrib_area is None:
                raise WatershedError('min_contrib_area must be specified if no'
                                     ' stream layer is provided')
            if fa is not None:
                with watershed(
                    self.path, tempdir=self.tempdir
                ).stream_reclass(fa, min_contrib_area) as ws:
                    m = ws.array != ws.nodata
            else:
                with watershed(self.path, tempdir=self.tempdir).stream_order(min_contrib_area) as ws:
                    m = ws.array != ws.nodata
            elev = self.array
        # Compute stream slope
        inds = numpy.where(m)
        diag = math.sqrt(self.csx**2 + self.csy**2)
        run = numpy.array([[diag, self.csy, diag],
                           [self.csx, 1, self.csx],
                           [diag, self.csy, diag]])
        ish, jsh = self.shape

        def compute(i, j):
            s = (slice(max([0, i - 1]),
                       min([i + 2, ish])),
                 slice(max([0, j - 1]),
                       min([j + 2, jsh])))
            base = elev[i, j]
            rise = numpy.abs(base - elev[s][m[s]])
            run_ = run[m[s]]
            run_ = run_[rise != 0]
            rise = rise[rise != 0]
            if run_.size == 0:
                return 0
            else:
                if units == 'degrees':
                    return numpy.mean(numpy.degrees(numpy.arctan(rise / run_)))
                else:
                    return numpy.mean(rise / run_) * 100

        output = self.copy(True, 'str_slope')
        a = numpy.full(output.shape, output.nodata, output.dtype)
        for _iter in range(iterations):
            slopefill = [compute(inds[0][i], inds[1][i])
                         for i in range(inds[0].shape[0])]
            a[m] = slopefill
        output[:] = a
        return output

    def alluvium(self, slope_thresh=6, stream_slope_thresh=5, **kwargs):
        '''
        Use the derivative of stream slope to determine regions of
        aggradation to predict alluvium deposition.  The input slope threshold
        is used as a cutoff for region delineation, which the stream slope
        threshold is the required stream slope to initiate deposition.
        Uses the dem as an input
        surface, and accepts (or they will be derived):

        streams: a streams raster
        min_contrib_area: minimum contributing area to define streams
        slope: a slope surface used to control the region delineation
        '''
        # Get or compute necessary datasets
        streams = kwargs.get('streams', None)
        min_contrib_area = kwargs.get('min_contrib_area', 1E04)
        slope = kwargs.get('slope', None)
        fa = kwargs.get('fa', None)
        if streams is None:
            strslo = self.stream_slope(2, min_contrib_area=min_contrib_area,
                                       fa=fa)
        else:
            strslo = self.stream_slope(2, streams)
        seeds = set(zip(*numpy.where(strslo.array != strslo.nodata)))
        if slope is None:
            slope = topo(self.path).slope().array

        # Create output tracking array
        track = numpy.zeros(shape=self.shape, dtype='uint8')

        # Recursively propagate downstream and delineate alluvium
        # Load elevation data into memory
        ish, jsh = self.shape
        dem = self.array
        dem_m = dem != self.nodata
        streams = strslo.array
        while True:
            try:
                seed = seeds.pop()
            except:
                break
            s = (slice(max(0, seed[0] - 1),
                       min(seed[0] + 2, ish)),
                 slice(max(0, seed[1] - 1),
                       min(seed[1] + 2, jsh)))
            str_mask = streams[s] != strslo.nodata
            if (streams[seed] != strslo.nodata) & (track[seed] == 0):
                # If stream exists check slope and initiate growth
                if streams[seed] > stream_slope_thresh:
                    track[seed] = 2
                else:
                    track[seed] = 1
            # High slope: erosion- directed propagation at higher slopes
            g = (dem[seed] - dem[s]).ravel()
            mask = numpy.argsort(g)
            if track[seed] == 2:
                # Create a mask with correct gradient directions
                mask = (mask > 5).reshape(str_mask.shape)
                mask = mask & (slope[s] < slope_thresh)
                track_add = 2
            # Low slope: aggradation- fan outwards at shallower slopes
            else:
                mask = (mask > 3).reshape(str_mask.shape)
                mask = mask & (slope[s] < (slope_thresh / 2))
                track_add = 1

            # Update track with non-stream cells
            mask = mask & ~str_mask & (track[s] == 0) & dem_m[s]
            s_i, s_j = numpy.where(mask)
            s_i = s_i + s[0].start
            s_j = s_j + s[1].start
            track[(s_i, s_j)] = track_add

            # Update the stack with new stream and other cells
            mask[str_mask & (track[s] == 0)] = 1
            s_i, s_j = numpy.where(mask)
            s_i = s_i + s[0].start
            s_j = s_j + s[1].start
            seeds.update(zip(s_i, s_j))

        alluv_out = self.copy(True, 'alluv').astype('uint8')
        alluv_out[:] = track
        alluv_out.nodataValues = [0]
        return watershed(alluv_out, tempdir=self.tempdir)

    def alluvium2(self, min_dep=0, max_dep=5, benches=1, **kwargs):
        '''
        Stuff
        '''
        min_contrib_area = kwargs.get('min_contrib_area', 1E04)
        slope = kwargs.get('slope', None)
        if slope is None:
            with topo(self.path).slope() as slope:
                sl = slope.array
                slnd = slope.nodata
        else:
            slope = raster(slope)
            sl = slope.array
            slnd = slope.nodata
        streams = kwargs.get('streams', None)
        if streams is None:
            with self.stream_slope(
                min_contrib_area=min_contrib_area
            ) as strslo:
                sa = strslo.array
                sand = strslo.nodata
        else:
            with self.stream_slope(streams=streams) as strslo:
                sa = strslo.array
                sand = strslo.nodata
        elev = self.array

        # First phase
        sa_m = sa != sand
        sl_m = sl != slnd
        alluv = sa_m.astype('uint8')
        slope_m = sl_m & (sl >= min_dep) & (sl <= max_dep)
        labels, num = ndimage.label(slope_m, numpy.ones(shape=(3, 3),
                                                        dtype='bool'))
        # Find mean stream slopes within regions
        str_labels = numpy.copy(labels)
        str_labels[~sa_m] = 0
        un_lbl = numpy.unique(str_labels[str_labels != 0])
        min_slopes = ndimage.minimum(sa, str_labels, un_lbl)
        max_slopes = ndimage.maximum(sa, str_labels, un_lbl)
        # Find max elevation of streams within regions
        max_elev = ndimage.maximum(elev, str_labels, un_lbl)

        # Iterate stream labels and assign regions based on elevation and slope
        for i, reg in enumerate(un_lbl):
            if reg == 0:
                continue
            # Modify region to reflect stream slope variance and max elevation
            m = ((labels == reg) & (sl >= min_slopes[i]) &
                 (sl <= max_slopes[i]) & (elev < max_elev[i]))
            alluv[m] = 1

        # Remove regions not attached to a stream
        labels, num = ndimage.label(alluv, numpy.ones(shape=(3, 3),
                                                      dtype='bool'))
        alluv = numpy.zeros(shape=alluv.shape, dtype='uint8')
        for reg in numpy.unique(labels[sa_m]):
            alluv[labels == reg] = 1

        alluv_out = self.copy(True, 'alluv').astype('uint8')
        alluv_out.nodataValues = [0]
        alluv_out[:] = alluv
        return watershed(alluv_out, tempdir=self.tempdir)


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
        slope_raster = self.copy(True, 'slope')
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
        aspect_raster = self.copy(True, 'aspect')
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
        '''
        Compute the roughness of a surface.
        Methods are:
        "std-elev": standard deviation of locally normalized elevation
        '''
        def eval_roughness(a):
            # Generate nodata mask and get views
            view = util.get_window_views(a, size)
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
        surf_rough = self.copy(True, 'srf_rgh')
        if self.useChunks:
            # Iterate chunks and calculate convergence
            for a, s in self.iterchunks(expand=size):
                s_ = util.truncate_slice(s, size)
                surf_rough[s_] = eval_roughness(a)
        else:
            # Calculate over all data
            surf_rough[:] = eval_roughness(self.array)

        return topo(surf_rough)
