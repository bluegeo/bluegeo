from spatial import *
from ftplib import FTP
import zipfile as zf
import os
import tempfile
import urllib
import re
from collections import defaultdict
import pandas
from csv import reader as csv_reader
import sys
from numba import jit

# for version compatibility
if sys.version_info[0] < 3:
    from urllib import urlretrieve  # Python 2.X
else:
    from urllib.request import urlretrieve  # Python 3.X


def collect_nrcan_dem(nts, tmp_dir=None, ftp_address='ftp.geogratis.gc.ca',
                      ftp_dir='/pub/nrcan_rncan/elevation/cdem_mnec'):
    """
    Download the 20m (approximate) DEM from the NRCAN ftp and merge
    :param nts: (dict) NTS numbers and letters in the form {83: ['a', 'b', 'c'], 73: ['e', 'f'], 81: 'all']}
    :param tmp_dir: (str) temporary directory
    :param ftp_address: (str) address of the nrcan FTP server
    :param ftp_dir: (str) address of the nrcan DEM directory
    :return: Raster instance
    """
    # Create temporary directory
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    tmp_dir = os.path.join(tmp_dir, 'nrcan_dem')
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    # Login to ftp
    ftp = FTP(ftp_address)
    ftp.login()
    ftp.cwd(ftp_dir)

    # Iterate NTS and collect rasters
    for num, lets in nts.iteritems():
        num = str(num)
        if len(num) == 1:
            num = '00{}'.format(num)
        elif len(num) == 2:
            num = '0{}'.format(num)
        try:
            ftp.cwd(num)
        except:
            print "Warning: cannot access {}".format(num)
            continue
        dirs = ftp.nlst()
        if lets == 'all':
            # Include all
            inc = dirs
        else:
            inc = [d for d in dirs if
                   any([let.lower() in d.replace('cdem_dem_', '').replace('_tif', '').replace('.zip', '').lower()
                        for let in lets])]
        if len(inc) == 0:
            print "Warning: None of the desired letters found in {}".format(num)
        for d in inc:
            tmpfile = os.path.join(tmp_dir, d)
            with open(tmpfile, 'wb') as zipf:
                ftp.retrbinary('RETR ' + d, zipf.write)
            z = zf.ZipFile(tmpfile, 'r')
            print 'Downloaded and now extracting {}'.format(d)
            z.extractall(tmp_dir)
            del z
            os.remove(tmpfile)
        ftp.cwd('..')
    ftp.quit()

    # Merge
    print "Merging rasters"
    files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.split('.')[-1] == 'tif']
    outpath = os.path.join(tmp_dir, 'nrcan_dem.tif')

    # Pre-check files to avoid errors at this stage
    _files = []
    for f in files:
        try:
            r = Raster(f)
            _files.append(f)
        except:
            print "Warning: cannot read file {}".format(f)

    if len(_files) > 0:
        command = 'gdalwarp -r cubic -overwrite "{}" "{}"'.format('" "'.join(_files), outpath)
        os.system(command)
    else:
        raise Exception("No files available for DEM merge operation")

    return Raster(outpath)


class Hydat(object):
    """
    Collect Environmenmt Canada Hydrometric Station Data

    Use:
      1. Construct an instance of Hydat, changing the url if the gov. decides to change it
      2. Collect a pandas dataframe with data from the station using the get_station_data method
      3. Clean data in the dataframe using the clean_station_data method (not yet developed)
    """

    def __init__(self, url='http://dd.weather.gc.ca/hydrometric/csv'):
        self.base_url = url  # Current URL for http data file collection

    def get_station_data(self, station_id, time='daily'):
        """
        Collect data from a station
        :param station_id: Station ID
        :param str time: 'daily' or 'hourly'
        :return: pandas dataframe
        """
        prov = self.get_province(station_id, time)

        # Download and read the file into a dataframe, and strip white space from headings
        df = pandas.read_csv(
            urlretrieve(self.build_url(prov, time, station_id))[0]
        ).rename(columns=lambda x: x.strip())

        return df

    def build_url(self, province, time=None, station_id=None):
        """
        Build a url using the given parameters
        :param time: (string) daily or hourly
        :param province: Provincial code (i.e. AB, BC, SK, MB, etc.)
        :param station_id: Station ID
        :return:
        """
        # A full URL should look like this:
        #   http://dd.weather.gc.ca/hydrometric/csv/AB/daily/AB_05AA022_daily_hydrometric.csv
        url = '{0}/{1}/'  # Just base and province
        if time is not None:
            # Add a time
            url += '{2}/'
            time = time.lower()
        if station_id is not None:
            # Add a station ID (full url)
            url += '{1}_{3}_{2}_hydrometric.csv'
            station_id = station_id.upper()

        return url.format(self.base_url, province.upper(), time, station_id)

    def collect_stations(self):
        """
        List all available stations by province and time
        :return: dict
        """
        # First, iterate provinces and build url's
        site = urllib.urlopen(self.base_url)

        # Check that the site is still valid or operating by collecting a list of provinces
        print "Collecting provinces"
        provinces = [s[9:11] for s in re.findall('<a href="../">../</a>', site.read())]

        # Iterate provinces and collect list of available times
        print "Collecting time periods and station ID's"
        self.stations = defaultdict(dict)
        for prov in provinces:
            site = urllib.urlopen(self.build_url(prov))
            expression = '<a href="[hd][a-zA-Z]*/">[hd][a-zA-Z]*/</a>'
            times = [s.split('>')[1].split('<')[0].replace('/', '') for s in re.findall(expression, site.read())]

            # Iterate times and collect the station ID's
            for time in times:
                site = urllib.urlopen(self.build_url(prov, time))
                expression = '<a href="{0}_[a-zA-Z0-9]*_{1}_hydrometric.csv">{0}_[a-zA-Z0-9]*_{1}_hydrometric.csv</a>'
                expression = expression.format(prov.upper(), time.lower())
                stations = [s.split('_')[1] for s in re.findall(expression, site.read())]
                self.stations[prov][time] = stations

    def get_province(self, station_id, time):
        """
        Collect the province using a station ID and time
        :param station_id: Station ID
        :param time: daily or hourly
        :return: province string
        """
        # Make sure the stations have been collected
        if not hasattr(self, 'stations'):
            self.collect_stations()

        keys = self.stations.keys()

        index = numpy.where(
            [any([True for id in self.stations[prov][time] if id == station_id]) for prov in keys]
        )[0]

        if index.size == 0:
            raise Exception('Cannot find the station "{}" with {} data'.format(station_id, time))

        return keys[int(index)]

    @staticmethod
    def clean_station_data(station_df):
        """
        Clean the station dataframe and create a mask for each
        :param station_df: dataframe of raw station data
        :return: pandas dataframe
        """
        # TODO implement data preparation here
        # Fix the datetime field

        # Cast to numeric fields where necessary

        # Interpolate missing data
