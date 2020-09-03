#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import pkg_resources
import sys


version = '0.1'


def is_installed(name):
    try:
        pkg_resources.get_distribution(name)
        return True
    except:
        return False


packages = [
    'bluegeo',
    'bluegeo.climate',
    'bluegeo.grass_session'
]


requires = [
    'networkx',
    'cython',
    'matplotlib',
    'numpy',
    'scipy',
    'scikit-image',
    'h5py',
    'numexpr',
    'shapely',
    'pandas',
    'llvmlite==0.32.0',  # Once LLVM 9+ works, remove (installed implicitly with numba)
    'numba==0.49.1',  # Once LLVM 9+ works, remove version
    'rtree',
    'gdal',
    'dask[complete]',
    # 'grass-session'  Add once PR approved
]

setup(name='bluegeo',
      version=version,
      description='bluegeo is a geospatial analysis library that strives for canonical pythonic '
                  '(particularly numpy-like) syntax',
      url='https://bitbucket.org/bluegeo/bluegeo.git',
      author='Blue Geosimulation',
      install_requires=requires,
      author_email='info@bluegeo.ca',
      license='MIT',
      packages=packages,
      zip_safe=False)
