#!/usr/bin/env bash

# Install conda
sudo apt-get update
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
eval "$(~/miniconda/bin/conda shell.bash hook)"

# Install python dependencies
conda create -y -n bluegeo
conda activate bluegeo
conda install -y gdal cython numpy scipy networkx matplotlib rtree scikit-image h5py numexpr shapely pandas numba

# Install Ubuntu GIS ppa & GRASS
sudo add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update
sudo apt-get -y install grass
sudo apt-get -y install grass-dev

# Install bluegeo (ignoring dependencies)
cd bluegeo && pip install --no-deps .
