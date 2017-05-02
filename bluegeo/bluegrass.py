import subprocess
import sys
import tempfile
import os
import time
import shutil


class GrassSession():
    def __init__(self, src=None, grassbin='grass',
                 persist=False, temp=None):

        # If temp is specified, use a different temporary directory
        if temp is not None:
            self.tempdir = temp
        else:
            self.tempdir = tempfile.gettempdir()
        self.persist = persist

        # if src
        if type(src) == int:
            # Assume epsg code
            self.location_seed = "EPSG:{}".format(src)
        else:
            # Assume georeferenced vector or raster
            self.location_seed = src

        self.grassbin = grassbin
        # TODO assert grassbin is executable and supports what we need

        startcmd = "{} --config path".format(grassbin)

        # Adapted from
        # http://grasswiki.osgeo.org/wiki/Working_with_GRASS_without_starting_it_explicitly#Python:_GRASS_GIS_7_without_existing_location_using_metadata_only
        p = subprocess.Popen(startcmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise Exception("ERROR: Cannot find GRASS GIS 7 start script ({})".format(startcmd))
        self.gisbase = out.strip('\n')

        self.gisdb = os.path.join(self.tempdir, 'mowerdb')
        self.location = "loc_{}".format(str(time.time()).replace(".","_"))
        self.mapset = "PERMANENT"

        os.environ['GISBASE'] = self.gisbase
        os.environ['GISDBASE'] = self.gisdb

    def gsetup(self):
        path = os.path.join(self.gisbase, 'etc', 'python')
        sys.path.append(path)
        os.environ['PYTHONPATH'] = ':'.join(sys.path)

        import grass.script.setup as gsetup
        gsetup.init(self.gisbase, self.gisdb, self.location, self.mapset)



    def create_location(self):
        try:
            os.stat(self.gisdb)
        except OSError:
            os.mkdir(self.gisdb)

        createcmd = "{0} -c {1} -e {2}".format(
            self.grassbin,
            self.location_seed,
            self.location_path)

        p = subprocess.Popen(createcmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise Exception("ERROR: GRASS GIS 7 start script ({})".format(createcmd))

    @property
    def location_path(self):
        return os.path.join(self.gisdb, self.location)

    def cleanup(self):
        if os.path.exists(self.location_path) and not self.persist:
            shutil.rmtree(self.location_path)
        if 'GISRC' in os.environ:
            del os.environ['GISRC']

    def __enter__(self):
        self.create_location()
        self.gsetup()
        return self

    def __exit__(self, type, value, traceback):
        self.cleanup()
