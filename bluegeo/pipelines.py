from spatial import *
from ftplib import FTP
import zipfile as zf
import os
import tempfile


def collect_nrcan_dem(nts, tmp_dir=None, ftp_address='ftp.geogratis.gc.ca',
                      ftp_dir='/pub/nrcan_rncan/elevation/cdem_mnec'):
    """
    Download the 20m (approximate) DEM from the NRCAN ftp and merge
    :param nts: (dict) NTS numbers and letters in the form {83: ['a', 'b', 'c'], 73: ['e', 'f'], 81: 'all']}
    :param tmp_dir: (str) temporary directory
    :param ftp_address: (str) address of the nrcan FTP server
    :param ftp_dir: (str) address of the nrcan DEM directory
    :return: raster instance
    """
    # Create temporary directory
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    tmp_dir = os.path.join('tmp_dir', 'nrcan_dem')
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
                   any([let.lower() in d.replace('cdem_dem_', '').replace('_tif', '').lower() for let in lets])]
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
    outpath = os.path.join(tmp_dir, '20m_dem.tif')

    # Pre-check files to avoid errors at this stage
    _files = []
    for f in files:
        try:
            r = raster(f)
            _files.append(f)
        except:
            print "Warning: cannot read file {}".format(f)

    if len(_files) > 0:
        command = 'gdalwarp -r bilinear -overwrite "{}" "{}"'.format('" "'.join(_files), outpath)
        os.system(command)
    else:
        raise Exception("No files available for DEM merge operation")

    return raster(outpath)
