FROM ubuntu:18.04

# Python - include numpy to use for GIS build
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends gcc python3.6 python3-setuptools python3.6-dev python3-pip python-rtree && \
    pip3 install numpy && \
    rm -rf /var/lib/apt/lists/*

# GIS
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable && \
    apt-get -y install --no-install-recommends gdal-bin libgdal-dev grass grass-dev && \
    rm -rf /var/lib/apt/lists/*

# bluegeo
# Install dependencies first to avoid re-installing all packages with each subsequent build
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . /bluegeo
RUN cd /bluegeo && pip3 install --no-deps .
