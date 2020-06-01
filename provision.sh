#!/usr/bin/env bash

# Python
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.6

# Install Ubuntu GIS ppa
sudo add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update

# Install gdal, grass, and python dependencies
sudo apt-get -y install python3-rtree gdal-bin python3-gdal grass grass-dev
sudo apt-get update

# Install pip
sudo apt-get -y install python3-pip

# Clone the repo with the user-provided credentials
git clone https://github.com/bluegeo/bluegeo.git
cd bluegeo && pip3 install .
