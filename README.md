bluegeo
=======

bluegeo is a library developed to assist in geospatial development where core scientific python libraries
such as numpy, scipy, scikit-image and scikit-learn can be implemented easily over raster and vector geospatial
data. For example:

```python
import bluegeo as bg
import numpy as np

print np.mean(bg.Raster('elevation.tif')[:])
784.91772
```

INSTALLATION
============

bluegeo uses python 2.7 (minimal work is required for upgrade to 3+ as time permits)

With Docker...
--------------
Build the image (from the root dir)

```docker build -t bluegeo .```

Start a session in the container

```
docker run --rm -v /home/ubuntu/bluegeo/scratch:/scratch -it bluegeo /bin/bash
```

Note:
* To preserve the container, do not use the `--rm` flag.
* A `scratch` directory (absolute path) is mounted to share files - omit it if necessary.

Using git...
------------
Clone the bluegeo repo

```git clone https://github.com/bluegeo/bluegeo.git```

## Install only Python requirements
from the root:
```pip install .```

## Install everything
Minimum server requirements on linux can be met using `privision.sh`
Run the script and wait

```
sudo chmod u+x bluegeo/provision.sh
bluegeo/provision.sh
```
