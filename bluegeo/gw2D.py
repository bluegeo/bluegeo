from raster import *
import matplotlib.pyplot as plt


class domain(object):
    '''
    Groundwater flow calculation domain.
    k is a hydraulic conductivity surface, b is a thickness surface
    '''
    def __init__(self, k, b, head_fill=0, length_units='m', time_units='s'):
        # Create a head surface
        self.nodata = -99999
        if k.shape != b.shape:
            raise ValueError('k and b have different shapes.')
        self.shape = b.shape
        self.head = numpy.full(self.shape, head_fill, dtype='float32')

        print "Computing transmissivity"
        knodata = k.nodata
        bnodata = b.nodata
        self.template = raster(k)
        k = numpy.pad(k.array, 1, 'constant', constant_values=0)
        b = numpy.pad(b.array, 1, 'constant', constant_values=0)
        self.domain = (k != knodata) & (b != bnodata) & (k != 0) & (b != 0)
        # Ensure no-flow boundary created where K is zero or nodata
        self.head[~self.domain[1:-1, 1:-1]] = self.nodata
        # Compute transmissivity in all directions
        k_0, b_0 = k[1:-1, 1:-1], b[1:-1, 1:-1]
        # +y
        slicemask = self.domain[1:-1, 1:-1] & self.domain[:-2, 1:-1]
        k_1, b_1 = k[:-2, 1:-1], b[:-2, 1:-1]
        self.tup = ne.evaluate(
            'where(slicemask,(2/((1/b_1)+(1/b_0)))*(2/((1/k_1)+(1/k_0))),0)'
        )
        # -y
        slicemask = self.domain[1:-1, 1:-1] & self.domain[2:, 1:-1]
        k_1, b_1 = k[2:, 1:-1], b[2:, 1:-1]
        self.tdown = ne.evaluate(
            'where(slicemask,(2/((1/b_1)+(1/b_0)))*(2/((1/k_1)+(1/k_0))),0)'
        )
        # -x
        slicemask = self.domain[1:-1, 1:-1] & self.domain[1:-1, :-2]
        k_1, b_1 = k[1:-1, :-2], b[1:-1, :-2]
        self.tleft = ne.evaluate(
            'where(slicemask,(2/((1/b_1)+(1/b_0)))*(2/((1/k_1)+(1/k_0))),0)'
        )
        # +x
        slicemask = self.domain[1:-1, 1:-1] & self.domain[1:-1, 2:]
        k_1, b_1 = k[1:-1, 2:], b[1:-1, 2:]
        self.tright = ne.evaluate(
            'where(slicemask,(2/((1/b_1)+(1/b_0)))*(2/((1/k_1)+(1/k_0))),0)'
        )

    def addBoundary(self, indices, type='no-flow', values=None):
        '''
        Add a boundary condition to a set of cell indices.
        Available types are "no-flow", "constant", "head-dependent", "flux".
        If any of "constant, "head-dependent", or "flux", the corresponding
        "values" must be specified (and match the indices size).
        '''
        if type == 'no-flow':
            self.head[indices] = self.nodata # Need to fix this
        elif type == 'constant':
            intcon = numpy.full(self.shape, self.nodata, dtype='float32')
            if hasattr(self, 'constantmask'):
                intcon[self.constantmask] = self.constantvalues
            intcon[indices] = values
            intcon[~self.domain[1:-1, 1:-1]] = self.nodata
            self.constantmask = intcon != self.nodata
            self.constantvalues = intcon[self.constantmask]
        elif type == 'flux':
            if not hasattr(self, 'fluxmask'):
                self.fluxmask = numpy.zeros(shape=self.shape, dtype='bool')
                self.fluxmask[indices] = 1
                self.fluxvalues = values
            else:
                vals = numpy.zeros(shape=self.shape, dtype='float32')
                vals[self.fluxmask] = self.fluxvalues
                vals[indices] = values
                self.fluxmask = vals != 0
                self.fluxvalues = vals[self.fluxmask]
        elif type == 'head-dependent':
            pass

    def solve(self, max_iter=100, tolerance=1E-6, show_convergence=False):
        '''Calculate a head surface using current parameter set'''
        # Apply constant head
        if hasattr(self, 'constantmask'):
            self.head[self.constantmask] = self.constantvalues

        # Create a copy of head for calculation
        head = numpy.pad(self.head, 1, 'constant', constant_values=self.nodata)
        # Create iterator for neighbourhood calculations
        nbrs = [('up', slice(0, -2), slice(1, -1)),
                ('down', slice(2, None), slice(1, -1)),
                ('left', slice(1, -1), slice(0, -2)),
                ('right', slice(1, -1), slice(2, None))]
        m = head != self.nodata
        # Iterate and implicitly compute head using derivation:
        # http://inside.mines.edu/~epoeter/583/06/discussion/fdspreadsheet.htm
        resid = tolerance + 0.1
        iters = 0
        convergence = []
        # Create calculation set
        flux = numpy.zeros(shape=self.shape, dtype='float32')
        hdd = numpy.zeros(shape=self.shape, dtype='float32')
        if hasattr(self, 'fluxmask'):
            flux[self.fluxmask] = self.fluxvalues
        if hasattr(self, 'hddpmask'):
            hdd[self.hddpmask] = self.hddpvalues
        calc_set = {'f': flux, 'tup': self.tup, 'tdown': self.tdown,
                    'tleft': self.tleft, 'tright': self.tright,
                    'hdd': hdd}
        den = ne.evaluate('tup+tdown+tleft+tright', local_dict=calc_set)
        calc_set.update({'den': den})
        while resid > tolerance:
            iters += 1
            # Copy head from previous iteration
            h = numpy.copy(head)

            # Add head views to calculation set
            for nbr in nbrs:
                name, i_slice, j_slice = nbr
                calc_set['h%s' % name] = head[i_slice, j_slice]

            # Calculate head at output
            h[1:-1, 1:-1] = ne.evaluate(
                'where(den>0,((hup*tup)+(hdown*tdown)+(hleft*tleft)+'
                '(hright*tright)+f+hdd)/den,0)',
                local_dict=calc_set
            )

            # Replace constant head values
            if hasattr(self, 'constantmask'):
                h[1:-1, 1:-1][self.constantmask] = self.constantvalues

            # Compute residual
            resid = numpy.max(numpy.abs(head - h))
            convergence.append(resid)
            if iters == max_iter:
                print "No convergence, with a residual of %s" % (resid)
                break

            # Update heads
            head = h
        self.head = head[1:-1, 1:-1]

        print ("%i iterations completed with a residual of %s" %
               (iters, resid))

        if show_convergence:
            plt.plot(convergence)
            plt.ylabel('Residual')
            plt.xlabel('Iteration')
            plt.show()

    def calculate_Q(self):
        '''Use the head surface to calculate the steady-state flux'''
        nbrs = [(self.t_y,
                 (slice(0, -1), slice(None, None)),
                 (slice(1, None), slice(None, None))),
                (self.t_y,
                 (slice(1, None), slice(None, None)),
                 (slice(0, -1), slice(None, None))),
                (self.t_x,
                 (slice(None, None), slice(0, -1)),
                 (slice(None, None), slice(1, None))),
                (self.t_x,
                 (slice(None, None), slice(1, None)),
                 (slice(None, None), slice(0, -1)))]
        m = self.head != self.nodata
        for nbr in nbrs:
            t, take, place = nbr
            mask = m[take[0], take[1]] & m[place[0], place[1]]
            t = t[mask]
            calcset = den[place[0], place[1]][mask]
            den[place[0], place[1]][mask] = ne.evaluate('calcset+t')
            headset = self.head[take[0], take[1]][mask]
            calcset = num[place[0], place[1]][mask]
            num[place[0],
                place[1]][mask] = ne.evaluate('calcset+(t*headset)')
        self.q = q

    def view(self, attr='head'):
        '''View the current head surface'''
        fig, ax = plt.subplots()
        a = numpy.copy(self.__dict__[attr])
        a[a == self.nodata] = numpy.nan
        im = ax.imshow(a, cmap='terrain')
        fig.colorbar(im)
        plt.show()

    def saveAsRaster(self, outpath, attr='head', nodata=-99999):
        '''Save to a raster file'''
        self.template.nodataValues = [nodata]
        self.template.save_gdal_raster(outpath)
        temp = raster(outpath, mode='r+')
        if attr == 'Q':
            # Calculate Q
            self.calculate_Q()
            temp[:] = self.q
        else:
            a = self.__dict__[attr]
            a[~self.domain[1:-1, 1:-1]] = nodata
            temp[:] = a
