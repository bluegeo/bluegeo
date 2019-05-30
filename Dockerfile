FROM ubuntu:16.04

RUN apt-get update && apt-get install -y software-properties-common python-pip python-rtree && \
    add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable && \
    apt-get install -y \
    gdal-bin \
    python-gdal \
    grass \
    grass-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . /bluegeo

RUN cd /bluegeo && pip install .

# The conda way

#FROM ubuntu:16.04
#
#ENV PATH /opt/conda/bin:$PATH
#
#RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 \
#    libxrender1 software-properties-common curl python-rtree python-pip && \
#    python -m pip install pip==9.0.3 --upgrade --force-reinstall && \
#    curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
#    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#    rm ~/miniconda.sh && \
#    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#    echo "conda activate base" >> ~/.bashrc && \
#    conda install -y gdal cython numpy scipy scikit-image h5py numexpr shapely pandas numba && \
#    apt-get update && \
#    add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable && \
#    apt-get install -y \
#    grass \
#    grass-dev && \
#    rm -rf /var/lib/apt/lists/*
#
#COPY . /bluegeo
#
#RUN cd /bluegeo && python -m pip install -e .

