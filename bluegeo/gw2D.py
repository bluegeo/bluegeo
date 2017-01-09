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

        # Create bi-directional transmissivity surfaces
        #   using harmonic means of K and b
        print "Computing transmissivity"
        knodata = k.nodata
        bnodata = b.nodata
        self.template = raster(k)
        k = k.array
        b = b.array
        self.domain = (k != knodata) & (b != bnodata) & (k != 0) & (b != 0)
        # Ensure no-flow boundary created where K is zero or nodata
        self.head[~self.domain] = self.nodata
        # Compute in y-direction
        slicemask = self.domain[:-1, :] & self.domain[1:, :]
        b_up, k_up = b[:-1, :][slicemask], k[:-1, :][slicemask]
        tup = ne.evaluate('b_up*k_up')
        del b_up, k_up
        b_down, k_down = b[1:, :][slicemask], k[1:, :][slicemask]
        tdown = ne.evaluate('b_down*k_down')
        del b_down, k_down
        self.t_y = numpy.zeros(shape=b[1:, :].shape, dtype='float32')
        self.t_y[slicemask] = ne.evaluate('2/((1/tup)+(1/tdown))')
        # Compute in x-direction
        slicemask = self.domain[:, 1:] & self.domain[:, :-1]
        b_left, k_left = b[:, 1:][slicemask], k[:, 1:][slicemask]
        tleft = ne.evaluate('b_left*k_left')
        del b_left, k_left
        b_right, k_right = b[:, :-1][slicemask], k[:, :-1][slicemask]
        tright = ne.evaluate('b_right*k_right')
        del b_right, k_right
        self.t_x = numpy.zeros(shape=b[:, 1:].shape, dtype='float32')
        self.t_x[slicemask] = ne.evaluate('2/((1/tleft)+(1/tright))')

    def addBoundary(self, indices, type='no-flow', values=None):
        '''
        Add a boundary condition to a set of cell indices.
        Available types are "no-flow", "constant", "head-dependent", "flux".
        If any of "constant, "head-dependent", or "flux", the corresponding
        "values" must be specified (and match the indices size).
        '''
        if type == 'no-flow':
            self.head[indices] = self.nodata
        elif type == 'constant':
            intcon = numpy.full(self.shape, self.nodata, dtype='float32')
            if hasattr(self, 'constantmask'):
                intcon[self.constantmask] = self.constantvalues
            intcon[indices] = values
            intcon[~self.domain] = self.nodata
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
        # Create iterator for neighbourhood calculations
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
        # Iterate and implicitly compute head using derivation:
        # http://inside.mines.edu/~epoeter/583/06/discussion/fdspreadsheet.htm
        resid = tolerance + 0.1
        iters = 0
        convergence = []
        while resid > tolerance:
            iters += 1
            # Copy head from previous iteration
            h = numpy.copy(self.head)

            # Create numerator and denominator surfaces
            num = numpy.zeros(shape=self.shape, dtype='float32')
            den = numpy.zeros(shape=self.shape, dtype='float32')

            # Calculate numerator and denominators
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

            # Add boundary conditions to numerator
            if hasattr(self, 'hddpmask'):
                pass
            if hasattr(self, 'fluxmask'):
                calcset = num[self.fluxmask]
                values = self.fluxvalues
                num[self.fluxmask] = ne.evaluate('calcset+values')

            # Isolated and stagnant cells maintain previous values
            stagnant = m & (den == 0) & (num == 0)

            # Calculate head at output
            num, den = num[m], den[m]
            h[m] = ne.evaluate('num/den')

            # Replace constant head values
            if hasattr(self, 'constantmask'):
                h[self.constantmask] = self.constantvalues

            # Replace stagnant values
            h[stagnant] = self.head[stagnant]

            # Compute residual
            resid = numpy.max(numpy.abs(self.head - h))
            convergence.append(resid)
            if iters == max_iter:
                print "No convergence, with a residual of %s" % (resid)
                break

            # Update heads
            self.head = h

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
            temp[:] = self.__dict__[attr]
