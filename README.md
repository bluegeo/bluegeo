# bluegeo

bluegeo is a library developed to assist in geospatial development where core scientific python libraries
such as numpy, scipy, scikit-image and scikit-learn can be implemented easily over raster and vector geospatial
data. For example:

```python
import bluegeo as bg
import numpy as np

print np.mean(bg.Raster('elevation.tif')[:])
784.91772
```

# INSTALLATION

bluegeo requires python 3, and a system installation of GDAL and GRASS

## Installation of Python Package

Clone the bluegeo repo

`git clone https://github.com/bluegeo/bluegeo.git`

## Install only Python requirements

from the root:
`pip install .`

## With Docker

Build the image (from the root dir)

`docker build -t bluegeo .`

Start a session in the container

```
docker run --rm -v /home/ubuntu/bluegeo/scratch:/scratch -it bluegeo /bin/bash
```

Note:

- To preserve the container, do not use the `--rm` flag.
- A `scratch` directory (absolute path) is mounted to share files - omit it if necessary.

## Linux and bash provisioning

The easiest way to use bluegeo is on Linux, using the bash script to install the dependencies

Minimum server requirements on linux can be met using `privision.sh` for the user `ubuntu`

example

```bash
bluegeo/provision.sh
```
