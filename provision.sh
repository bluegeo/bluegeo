#!/usr/bin/env bash

# Install conda
sudo apt-get update
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"

# Install python dependencies
conda create -n bluegeo
conda activate bluegeo
conda install -y gdal cython numpy scipy networkx matplotlib rtree scikit-image h5py numexpr shapely pandas numba

# Install Ubuntu GIS ppa
sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update
sudo apt-get -y install grass
sudo apt-get -y install grass-dev

# Install bluegeo
cd bluegeo && pip install .
