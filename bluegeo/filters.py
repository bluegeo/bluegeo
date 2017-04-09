from raster import *
import util


class FilterError(Exception):
    pass


class rastfilter(raster):
    """
    Perform a series of filters on a surface
    """

    def __init__(self, surface):
        # Open and change to float if not already
        if isinstance(surface, raster):
            self.__dict__.update(surface.__dict__)
        else:
            super(rastfilter, self).__init__(surface)
        if 'float' not in self.dtype:
            out = self.astype('float32')
            self.__dict__.update(out.__dict__)
        # Change interpolation method unless otherwise specified
        self.interpolationMethod = 'bilinear'


    def most_common(self, size=(3, 3)):
        """
        Update raster with the most frequent value in a window
        """

        def eval_mode(a):
            # Add mask to local dictionary
            a = util.window_on_last_axis(a, size)

            # Compute mode on last axis
            axis = 2
            ndim = 3

            # Sort array
            sort = numpy.sort(a, axis=axis)
            # Create array to transpose along the axis and get padding shape
            transpose = numpy.roll(numpy.arange(ndim)[::-1], axis)
            shape = list(sort.shape)
            shape[axis] = 1
            # Create a boolean array along strides of unique values
            strides = numpy.concatenate([numpy.zeros(shape=shape, dtype='bool'),
                                         numpy.diff(sort, axis=axis) == 0,
                                         numpy.zeros(shape=shape, dtype='bool')],
                                        axis=axis).transpose(transpose).ravel()
            # Count the stride lengths
            counts = numpy.cumsum(strides)
            counts[~strides] = numpy.concatenate([[0], numpy.diff(counts[~strides])])
            counts[strides] = 0
            # Get shape of padded counts and slice to return to the original shape
            shape = numpy.array(sort.shape)
            shape[axis] += 1
            shape = shape[transpose]
            slices = [slice(None)] * ndim
            slices[axis] = slice(1, None)
            # Reshape and compute final counts
            counts = counts.reshape(shape).transpose(transpose)[slices] + 1

            # Find maximum counts and return modals/counts
            slices = [slice(None, i) for i in sort.shape]
            del slices[axis]
            index = numpy.ogrid[slices]
            index.insert(axis, numpy.argmax(counts, axis=axis))
            return numpy.squeeze(sort[index])

        # Allocate output
        mode_raster = self.copy(file_suffix='modefilter', fill=self.nodata)
        if self.useChunks:
            # Iterate chunks and calculate mode (memory-intensive, so don't fill cache)
            for a, s in self.iterchunks(expand=size, fill_cache=False):
                s_ = util.truncate_slice(s, size)
                mode_raster[s_] = eval_mode(a)
        else:
            # Calculate over all data
            # TODO: Change this to match the size, otherwise there will be an error. Occurs in others throughout the library
            mode_raster[1:-1, 1:-1] = eval_mode(self.array)

        return rastfilter(mode_raster)
