"""
Climate NA grid specs
"""
from osgeo import osr


class Specs(object):
    # PRISM
    prism_csx = 2.5 / 60
    prism_csy = 2.5 / 60
    prism_shape = (1651, 3038)
    prism_top, prism_left = (83.23, -179.23)
    prism_bottom = prism_top - (prism_shape[0] * prism_csy)
    prism_right = prism_left + (prism_shape[1] * prism_csx)

    # Normals
    normal_csx = 0.5
    normal_csy = 0.5
    normal_shape = (144, 258)
    normal_top, normal_left = (84.994, -179.667)
    normal_bottom = normal_top - (normal_shape[0] * normal_csy)
    normal_right = normal_left + (normal_shape[1] * normal_csx)

    # GCM
    gcm_csx = 1
    gcm_csy = 1
    gcm_shape = (72, 129)
    gcm_top, gcm_left = (84.994, -179.667)
    gcm_bottom = gcm_top - (gcm_shape[0] * gcm_csy)
    gcm_right = gcm_left + (gcm_shape[1] * gcm_csx)

    # Spatial reference (GCS)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4269)
    projection = sr.ExportToWkt()

    test = 'some' \
        'string'

    # Parameters
    prism_parameters = ['Elevation', 'Tmax01', 'Tmax02', 'Tmax03', 'Tmax04', 'Tmax05', 'Tmax06',
                        'Tmax07', 'Tmax08', 'Tmax09', 'Tmax10', 'Tmax11', 'Tmax12', 'Tmin01',
                        'Tmin02', 'Tmin03', 'Tmin04', 'Tmin05', 'Tmin06', 'Tmin07', 'Tmin08',
                        'Tmin09', 'Tmin10', 'Tmin11', 'Tmin12', 'PPT01', 'PPT02', 'PPT03',
                        'PPT04', 'PPT05', 'PPT06', 'PPT07', 'PPT08', 'PPT09', 'PPT10', 'PPT11',
                        'PPT12', 'RAD01', 'RAD02', 'RAD03', 'RAD04', 'RAD05', 'RAD06', 'RAD07',
                        'RAD08', 'RAD09', 'RAD10', 'RAD11', 'RAD12']
