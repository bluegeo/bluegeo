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


requires = ['numpy', 'scipy', 'scikit-image', 'h5py', 'numexpr']

setup(name='bluegeo',
      version=version,
      description='Bluegeo python library for data manipulation',
      url='https://bitbucket.org/bluegeo/bluegeo.git',
      author='Bluegeo',
      install_requires=requires,
      author_email='info@bluegeo.ca',
      license='',
      packages=['bluegeo'],
      zip_safe=False)
