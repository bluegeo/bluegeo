# bluegeo

_bluegeo_ is a library developed to assist in geospatial development where core scientific python libraries
such as numpy, scipy, scikit-image and scikit-learn can be implemented easily over raster and vector geospatial
data.

For example:

```python
import bluegeo as bg
import numpy as np

# Virtually read a raster file
r = bg.Raster('elevation.tif')

# Slice the data, returning a numpy array
# Note: Slice-based __getitem__ operations are supported, but fancy indexing is not
a = np.ma.masked_equal(r[:], r.nodata)

# Perform a numpy operation
print(a.mean())
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

The easiest way to use bluegeo is on Linux, using the `privision.sh` script to install the dependencies:

```bash
bluegeo/provision.sh
```

You may need to re-initialize conda after this

```bash
eval "$(~/miniconda/bin/conda shell.bash hook)"
```

Test the installation

```bash
python
```

```python
>>> import bluegeo as bg
>>>
```
