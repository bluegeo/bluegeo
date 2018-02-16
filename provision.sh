#!/usr/bin/env bash

# Install Ubuntu GIS ppa
sudo add-apt-repository -y ppa:ubuntugis/ppa
sudo apt-get update

# Install gdal, grass, and python dependencies
sudo apt-get -y install gdal-bin
# Not this does not include Python 3
sudo apt-get -y install python-gdal
sudo apt-get -y install grass
sudo apt-get -y install grass-dev

# Install pip
sudo apt-get -y install python-pip
pip install --upgrade pip

# Install numba separate from bluegeo libarary
sudo apt -y install llvm-3.7 libedit-dev
sudo -H LLVM_CONFIG=/usr/bin/llvm-config-2.7 pip install llvmlite numba

# Install development version of bluegeo
cd bluegeo
#If working on a development version and you want any changes to bluegeo reflected when you import it, install bluegeo with the following flag
sudo pip install -e .

# Uncomment this section if you wish to install R

#sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
#sudo add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'
#sudo apt-get update
#sudo apt-get -y install r-base
#sudo apt-get install -y libcurl4-openssl-dev