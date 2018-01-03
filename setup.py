#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import pkg_resources
import sys


version = open('bluegeo/VERSION', 'r').read().strip()


def is_installed(name):
    try:
        pkg_resources.get_distribution(name)
        return True
    except:
        return False


requires = ['numpy', 'scipy', 'scikit-image', 'h5py', 'numexpr', 'shapely']

setup(name='bluegeo',
      version=version,
      description='bluegeo is a geospatial analysis library that strives for canonical pythonic '
                  '(particularly numpy-like) syntax',
      url='https://bitbucket.org/bluegeo/bluegeo.git',
      author='Blue Geosimulation',
      install_requires=requires,
      author_email='info@bluegeo.ca',
      license='',
      packages=['bluegeo'],
      zip_safe=False)

print "Installation Complete.  Additional dependencies are required to use certain functionality.  These include:\n" \
      "numba: https://numba.pydata.org/\n" \
      "GDAL (including python-gdal): http://www.gdal.org/\n" \
      "GRASS (including grass-dev): https://grass.osgeo.org/\n"
