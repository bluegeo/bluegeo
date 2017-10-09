INSTALLATION
==========

Minimum server requirements can be met using:
```bash
# Install Ubuntu GIS ppa
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update

# Install gdal, grass, and python dependencies
sudo apt-get install gdal-bin
# Not this does not include Python 3
sudo apt-get install python-gdal
sudo apt-get install grass
sudo apt-get install grass-dev

# Install git and clone bluegeo repo
sudo apt-get update
sudo apt-get install git
# This will prompt for a password
git clone https://devincairns@bitbucket.org/bluegeo/bluegeo.git

# Install pip
sudo apt-get install python-pip

# Install numba separate from bluegeo libarary
sudo apt install llvm-3.7 libedit-dev
sudo -H LLVM_CONFIG=/usr/bin/llvm-config-2.7 pip install llvmlite numba

# Install development version of bluegeo
cd bluegeo
#If working on a development version and you want any changes to bluegeo reflected when you import it, install bluegeo with the following flag
pip install -e .
#else
pip install .

# Install grass extension(s)
sudo python
# Use the following in interpreter:
#import bluegeo as bg
#with bg.bluegrass.GrassSession(26911) as gs:
#    from grass.script import core as grass
#    grass.run_command('g.extension', extension='r.stream.order', flags='s')
```

References:
https://python-packaging.readthedocs.io/en/latest/minimal.html
