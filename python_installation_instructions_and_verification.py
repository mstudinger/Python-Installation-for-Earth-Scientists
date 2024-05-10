# -*- coding: utf-8 -*-
"""
Created on Thu May  9 22:50:37 2024

@author: Michael Studinger
"""

#%% 1) get computer name and OS
import os
import socket
import platform

width_output = 78 # number of characters for output 

str_title = ' Platform and Operating System '
str_title = str_title.center(width_output, "-")
print("\n" + str_title + "\n")

print(f'Computer:            {socket.gethostname():s}')
print(f'Operating system:    {platform.system():s} {os.name.upper():s} Release: {platform.release():s}')

#%% 2) get Python version

import sys

python_version        = sys.version
python_version_detail = python_version.split(" | ",-1) # -1 gets all occurences. can set max number of splits

str_title = ' Python Version and Installation '
str_title = str_title.center(width_output, "-")
print("\n" + str_title + "\n")

print(f'Python version:      {sys.version_info.major:d}.{sys.version_info.minor:d}.{sys.version_info.micro:d}')
print(f'Python installation: {python_version_detail[0]:s} {python_version_detail[1]:s}')
print(f'                     {python_version_detail[2]:s}')

#%% 3) install geopandas and gdal as well as geodatasets
"""
    Installation: conda install geopandas
    Installation: conda install geodatasets -c conda-forge # used for testing
"""

import warnings
warnings.filterwarnings("ignore", module="gdal")        # suppresses all warnings from gdal module
warnings.filterwarnings("ignore", module="geodatasets") # suppresses all warnings from geodatasets module
warnings.filterwarnings("ignore", category=DeprecationWarning)

import geodatasets
import geopandas
import geopandas as gdp
from   osgeo import gdal
from   geodatasets import get_path


str_title = ' GeoPandas & GDAL/Python™ '
str_title = str_title.center(width_output, "-")
print("\n" + str_title + "\n")
print(f'GeoPandas    version: {gdp.__version__:s}')
print(f'GDAL/Python™ version: {gdal.__version__:s}')
print(f'geodatasets  version: {geodatasets.__version__:s}')

# verify GeoPandas
path_to_data = get_path("nybb")
gdf = geopandas.read_file(path_to_data)

# verify modules
f_name_geotiff = r"." + os.sep + "test_data" + os.sep + "GEOTIFF" + os.sep + "IOCAM1B_2019_GR_NASA_20190506-131614.4217.tif"

cambot = gdal.Open(f_name_geotiff)
cambot_proj = cambot.GetProjection()

if "Polar_Stereographic" in cambot_proj:
    print('GDAL/Python™ verification: GeoTiff projection information contains "Polar_Stereographic"')
else:
    print('GDAL/Python™ verification: ERROR: GeoTiff projection information could not be read')

# verify GeoPandas
if hasattr(gdf, 'area'):
  print('GeoPandas    verification: GeoPandas DataFrame has no attribute "area"')
  print(gdf)
else:
  print('GeoPandas    verification: ERROR: GeoPandas DataFrame has no attribute "area"')
  
#%% 4) OpenCV

"""
    see: https://opencv.org/get-started/
    pip3 install opencv-python\
    -> Successfully installed opencv-python-4.9.0.80
    TODO: unclear if .dll needs to be copied as described in link above 
"""

import cv2 as cv2

f_name_jpg = r"." + os.sep + "test_data" + os.sep + "JPEG" + os.sep + "IOCAM0_2019_GR_NASA_20190506-131614.4217.jpg"
image_bgr  = cv2.imread(f_name_jpg)
# image      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
img_size   = image_bgr.shape
 
str_title = ' OpenCV '
str_title = str_title.center(width_output, "-")
print("\n" + str_title + "\n")
print(f'OpenCV version:      {cv2.__version__:s}')
print(f'Test image:          {img_size[0]:d} × {img_size[1]:d} pixels')

#%% 5) conda install Pytorch Torchvision Torchaudio cpuonly -c pytorch

"""
The SAM Python™ module requires PyTorch and TorchVision. The SAM installation instructions recommend installing
both packages with CUDA support, however, if that causes error messages the solution is often to install
both packages without CUDA support.

    Recommended installation with CUDA support:
    > conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    Installation without CUDA support (CPU only):
    > conda install pytorch torchvision torchaudio cpuonly -c pytorch
    To install the SAM Python™ module use (see below):
    pip install git+https://github.com/facebookresearch/segment-anything.git

"""
# verify the installation and check for CUDA support:

import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

# determine which processing unit to use
if torch.cuda.is_available():
    processing_unit = "cuda" # use graphics processing unit (GPU)
else:    
    processing_unit = "cpu"  # use central processing unit (CPU)
    
str_title = ' PyTorch & TorchVision '
str_title = str_title.center(width_output, "-")
print("\n" + str_title + "\n")
print(f'PyTorch     version: {torch.__version__:s}')
print(f'Torchvision version: {torchvision.__version__:s}')
print(f'Processing  support: {processing_unit.upper():s}')
    

#%% 6) pip install git+https://github.com/facebookresearch/segment-anything.git

"""
    The SAM Python™ module requires PyTorch and TorchVision (see Section 5 above).
    To install the SAM Python™ module use:
    pip install git+https://github.com/facebookresearch/segment-anything.git
"""

import segment_anything
  
str_title = ' Segment Anything Model (SAM) '
str_title = str_title.center(width_output, "-")
print("\n" + str_title + "\n")

modulename = 'segment_anything'
if modulename not in sys.modules:
    print(f'Segment Anything:    ERROR: {modulename:s} module is unavailable')
else:
    print(f'Segment Anything:    {modulename:s} module is imported and in the sys.modules dictionary')

#%% 7) pip install pyproj
"""
    PYPRØJ and shapely are installed as part of geopandas. If geopandas is not installed use:
    pip install pyproj
    Shapely is a Python™ package that uses the GEOS library to perform set-theoretic operations
    on planar features. 
"""

import pyproj
import shapely
import numpy as np
from shapely import Point

# verify PYPRØJ
# EPSG:3413 NSIDC Sea Ice Polar Stereographic North/WGS-84 used for Greenland
# EPSG:4326 WGS84 - World Geodetic System 1984, used in GPS 
geo2xy = pyproj.Transformer.from_crs(4326,3413)
xy2geo = pyproj.Transformer.from_crs(3413,4326)
xy     = geo2xy.transform(70.0, -45.0)
lonlat = xy2geo.transform(xy[0],xy[1])

# verify Shapely
patch = Point(0.0, 0.0).buffer(1.0)
path_area = patch.area # result should be Pi

str_title = ' PYPRØJ & Shapely '
str_title = str_title.center(width_output, "-")
print("\n" + str_title + "\n")

print(f'PYPRØJ       version: {pyproj.proj_version_str:s}')
print(f'Shapely      version: {shapely.__version__:s}')
if lonlat[1] == -45:
    print('PYPRØJ  verification: projected geographic coordinates to polar stereographic and back')
if (np.isfinite(path_area)) & (path_area > 0):
    print('Shapely verification: calculated buffer area around point geometry')

#%% 8) conda install -c conda-forge pycrs

import pycrs 
str_title = ' PyCRS GIS '
str_title = str_title.center(width_output, "-")
print("\n" + str_title + "\n")

print(f'PyCRS       version: {pycrs.__version__:s}')

#%% 9) PyMap3D 

"""
    PyMap3D is a Python™ (optional Numpy) toolbox for 3D geographic coordinate transformations and geodesy.
    It supports various coordinate systems, ellipsoids, and Vincenty functions, and has a similar syntax 
    to the MATLAB®  Mapping Toolbox.
    Installation: conda install -c conda-forge pymap3d
"""
import pymap3d

str_title = ' PyMap3D '
str_title = str_title.center(width_output, "-")
print("\n" + str_title + "\n")
print(f'PyMap3D     version: {pymap3d.__version__:s}') 

#%% 10) cmcrameri
"""
    Python™ implementation of the Scientific Colour Maps version 8.0 (2023-06-14).
    Installation: conda install -c conda-forge cmcrameri
"""
import cmcrameri

str_title = ' Python™ implementation of the Scientific Colour Maps '
str_title = str_title.center(width_output, "-")
print("\n" + str_title + "\n")
print(f'cmcrameri   version: {cmcrameri.__version__:s}') 


