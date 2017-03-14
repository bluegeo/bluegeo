from raster import *
import tempfile


"""
Create a QGIS colour map file from a raster to display unique values
Devin Cairns, started March, 2017
"""


def create_map(input_raster, alpha=255, output='temp'):
    """
    Do it
    :param input_raster: Any object supported by bluegeo.raster
    :return: Location to a .txt that may be imported in to QGIS
    """
    nodata = raster(input_raster).nodata
    unique_values = unique(input_raster)
    if output == 'temp':
        with tempfile.NamedTemporaryFile() as f:
            outfile = f.name + '.txt'
    else:
        if output.split('.')[-1] != 'txt':
            outfile = output + '.txt'
        else:
            outfile = output
    with open(outfile, 'w') as f:
        f.write('# QGIS Generated Color Map Export File\n{}\n'.format(
            ','.join(map(str, [nodata, 255, 255, 255, 0, 'Null Values'])))
        )
        f.write('\n'.join(['{},{},{},Value {}'.format(
            val, ','.join(map(str, numpy.random.randint(0, 256, (3,)))), alpha, val) for val in unique_values])
        )
    return outfile
