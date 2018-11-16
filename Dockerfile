FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    software-properties-common && add-apt-repository -y \
    ppa:ubuntugis/ubuntugis-experimental && apt-get install -y \
    python-rtree \
    gdal-bin \
    python-gdal \
    grass \
    grass-dev \
    python-pip && \
    python -m pip install pip==9.0.3 --upgrade --force-reinstall

COPY . /bluegeo

RUN cd /bluegeo && python -m pip install -e .

