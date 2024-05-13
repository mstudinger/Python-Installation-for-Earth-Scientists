#!/usr/bin/env python
# coding: utf-8

# <font size="7"> <div align="center">Python™ for Earth Scientists</div></font>
# <font size="6"> <div align="center">Package Installation and Verification</div></font>

# This Jupyter notebook contains instructions for installing essential Python™ packages for Earth scientists and test the functionality of the installed packages. This Jupyter notebook and the corresponding Python™ script have been tested under the following environments:  
# * Windows 11 Pro (23H2)        Python™ 3.11.7
# * Windows 10 Enterprise (21H2) Python™ 3.9.13
# * Linux POSIX Release: 5.15.146.1-microsoft-standard-WSL2 Python™ 3.11.7
# * Linux POSIX Release: 5.10.198-187.748.amzn2.x86_64 Python™ 3.11.8 (selected packages tested on the [CryoCloud JupyterHub](https://cryointhecloud.com))

# ## 1. Determine if code is run from a classic Jupyter Notebook, JupyterLab, or Python™

# In[1]:


import psutil

parent_process = psutil.Process().parent().cmdline()[-1]

if 'jupyter-lab' in parent_process:
    env_str = 'JupyterLab'
    ENVIRONMENT = 'Jupyterlab'
elif 'jupyter-notebook' in parent_process:
    env_str = 'Jupyter Classic Notebook'
    ENVIRONMENT = 'JupyterNB'
elif "spyder-script.py" in parent_process:
    env_str = 'Windows-Python™ (Spyder)'
    ENVIRONMENT = 'Windows-Python-Spyder'
elif parent_process == '-bash':
    env_str = 'Linux-Python™ (command line)'
    ENVIRONMENT = 'Linux-Python-cmdline'
elif "8890" in parent_process:
    env_str = 'Linux-Python™ (JupyterLab)'
    ENVIRONMENT = 'Linux-JupyterLab'
elif 'anaconda3' in parent_process:
    env_str = 'Windows-Python™(command line)'
    ENVIRONMENT = 'Windows-Python-cmdline'
elif 'jupyterhub-singleuser' in parent_process:
    env_str = 'CryoCloud-JupyterLab'
    ENVIRONMENT = 'CryoCloud-JupyterLab'
else:
    env_str = 'Python™'
    ENVIRONMENT = "Python"


# ## 2. Get name of computer and operating system information

# In[2]:


# get computer name and OS
import os
import socket
import platform

if platform.system() == "Windows":
    win32 = platform.win32_ver(release='', version='', csd='', ptype='')
    ver_str = f'({win32[1]:s})'
else:
    ver_str = ''

print('\nPlatform and Operating System:\n')
print(f'Computer name     : {socket.gethostname():s}')
print(f'Operating system  : {platform.system():s} {os.name.upper():s}, Release {platform.release():s} {ver_str:s}')
print(f'Python environment: {env_str:s}')


# ## 3. Get Python™ version

# In[3]:


import sys
import jupyterlab

python_version        = sys.version
python_version_detail = python_version.split(" | ",-1) # -1 gets all occurences. can set max number of splits

print('\nPython™ and JupyterLab versions:\n')
print(f'Python™ version:      {sys.version_info.major:d}.{sys.version_info.minor:d}.{sys.version_info.micro:d}')
if len(python_version_detail) == 3:
    print(f'Python™ installation: {python_version_detail[0]:s} {python_version_detail[1]:s}')
    print(f'                      {python_version_detail[2]:s}')
print(f'JupyterLab version:   {jupyterlab.__version__:s}')


# ## 4. Package installation and verification of commonly used modules

# ### 4.1. GeoPandas & GDAL/Python™

# In[4]:


"""
    Installation: conda install geopandas
    Installation: conda install geodatasets -c conda-forge # used for testing
"""
import warnings
warnings.filterwarnings("ignore", module="gdal")        # suppresses all warnings from gdal module
warnings.filterwarnings("ignore", module="geodatasets") # suppresses all warnings from geodatasets module
warnings.filterwarnings("ignore", module="paramiko")    # suppresses all warnings from paramiko module (which would probably be sufficient)
warnings.filterwarnings("ignore", category=DeprecationWarning)

PRINT_GDF = False # will be set to True if a GeoDataFrame is loaded

if (ENVIRONMENT == "Jupyterlab" or ENVIRONMENT == "Jupyternotbook") and (sys.version_info.minor > 9):
    print(" ")
    print(f"WARNING: {env_str:s} and Python™ version {sys.version_info.major:d}.{sys.version_info.minor:d}:")
    print(f"         GeoPandas & GDAL/Python™ installations have been verified to work with Xarray in Python™ version {sys.version_info.major:d}.{sys.version_info.minor:d}.")
    print("         For reasons that are unclear GeoPandas produces a GDAL related errror when executed from JupyterLab or \n         Jupyter Notebook after Xarray is installed, but works fine when called from the Python™ console or Spyder.")
    print(f'         => Skipping GeoPandas & GDAL/Python™ verification since code is executed from {env_str:s} with Python™ version {sys.version_info.major:d}.{sys.version_info.minor:d}.')
    PRINT_GDF = False
    
elif sys.version_info.minor >= 9:
    import geodatasets
    import geopandas
    import geopandas as gdp
    from   osgeo import gdal
    from   geodatasets import get_path
    print('\nGeoPandas & GDAL/Python™:\n')
    print(f'GeoPandas    version:      {gdp.__version__:s}')
    print(f'GDAL/Python™ version:      {gdal.__version__:s}')
    print(f'geodatasets  version:      {geodatasets.__version__:s}')
    
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
      print('GeoPandas    verification: GeoPandas GeoDataFrame has attribute "area":')
      PRINT_GDF = True
    else:
      print('GeoPandas    verification: ERROR: GeoPandas GeoDataFrame has no attribute "area"')
else:
    os.sys.exit("The Python™ computing environment or version are not supported. Abort.")
print(" ")
if PRINT_GDF:
    display(gdf)


# ### 4.2. OpenCV

# In[5]:


"""
    see: https://opencv.org/get-started/
    pip3 install opencv-python
    -> Successfully installed opencv-python-4.9.0.80
    TODO: unclear if .dll needs to be copied as described in link above 
"""
import cv2 as cv2
import pathlib

if platform.system() == "Windows":
    f_name_jpg = pathlib.Path(r"./test_data/JPEG/IOCAM0_2019_GR_NASA_20190506-131614.4217.jpg")
elif platform.system() == 'Linux':
    f_name_jpg = pathlib.Path(r"./test_data/JPEG/IOCAM0_2019_GR_NASA_20190506-131614.4217.jpg")

image_bgr  = cv2.imread(str(f_name_jpg))
# image      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
img_size   = image_bgr.shape
 
print('\nOpenCV:\n')
print(f'OpenCV version: {cv2.__version__:s}')
print(f'Test image:     {img_size[0]:d} × {img_size[1]:d} pixels')


# ### 4.3. Pytorch, Torchvision, and Torchaudio

# In[6]:


"""
    The SAM Python™ module requires PyTorch and TorchVision. The SAM installation instructions recommend installing
    both packages with CUDA support, however, if that causes error messages (particularly on Windows) the solution 
    is often to install both packages without CUDA support.

    Recommended installation with CUDA (GPU) support:
    > conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    Installation without CUDA support (CPU only):
    > conda install pytorch torchvision torchaudio cpuonly -c pytorch
    To install the SAM Python™ module (see below):
    pip install git+https://github.com/facebookresearch/segment-anything.git

"""
# verify the installation and check for CUDA support:
import torch
import torchvision

# determine which processing unit to use
if torch.cuda.is_available():
    processing_unit = "cuda" # use graphics processing unit (GPU)
else:    
    processing_unit = "cpu"  # use central processing unit (CPU)
    
print('\nPyTorch & TorchVision:\n')
print(f'PyTorch     version: {torch.__version__:s}')
print(f'Torchvision version: {torchvision.__version__:s}')
print(f'Processing  support: {processing_unit.upper():s}')


# ### 4.4. Segment Anything Model (SAM)

# In[7]:


"""
    The SAM Python™ module requires PyTorch and TorchVision (see Section 5 above).
    To install the SAM Python™ module use:
    pip install git+https://github.com/facebookresearch/segment-anything.git
"""
import segment_anything
    
print('\nSegment Anything Model (SAM):')

modulename = 'segment_anything'
if modulename not in sys.modules:
    print(f'Segment Anything: ERROR: {modulename:s} module is unavailable')
else:
    print(f'Segment Anything: {modulename:s} module is imported and in the sys.modules dictionary')


# ### 4.5. PYPRØJ

# PYPRØJ is a Python™ interface to PROJ, a cartographic projections and coordinate transformations library

# In[8]:


"""
    PYPRØJ and shapely are installed as part of geopandas. If geopandas is not installed use:
    pip install pyproj
    Shapely is a Python™ package that uses the GEOS library to perform set-theoretic operations
    on planar features. 
"""
import pyproj
import shapely
import numpy as np
from   shapely import Point

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

print('\nPYPRØJ & Shapely:\n')
print(f'PYPRØJ       version: {pyproj.proj_version_str:s}')
print(f'Shapely      version: {shapely.__version__:s}')
if lonlat[1] == -45:
    print('PYPRØJ  verification: projected geographic coordinates to polar stereographic')
if (np.isfinite(path_area)) & (path_area > 0):
    print('Shapely verification: calculated buffer area around point geometry')


# ### 4.6. PyCRS

# PyCRS is a pure Python GIS package for reading, writing, and converting between various common coordinate reference system (CRS) string and data source formats.

# In[9]:


import pycrs 
print(f'\nPyCRS version: {pycrs.__version__:s}')


# ### 4.7. PyMap3D

# PyMap3D is a Python™ (optional Numpy) toolbox for 3D geographic coordinate transformations

# In[10]:


"""
    PyMap3D is a Python™ (optional Numpy) toolbox for 3D geographic coordinate transformations and geodesy.
    It supports various coordinate systems, ellipsoids, and Vincenty functions, and has a similar syntax 
    to the MATLAB®  Mapping Toolbox.
    Installation: conda install -c conda-forge pymap3d
"""
import pymap3d

print(f'\nPyMap3D version: {pymap3d.__version__:s}') 


# ### 4.8. Python™ implementation of Scientific Colour Maps

# In[11]:


"""
    Python™ implementation of the Scientific Colour Maps version 8.0 (2023-06-14).
    Installation: conda install -c conda-forge cmcrameri
"""
import cmcrameri

print('\nPython™ implementation of the Scientific Colour Maps:')
print(f'cmcrameri version: {cmcrameri.__version__:s}')


# ### 4.9. laspy & lazrs-python

# In[12]:


"""
    laspy is a Python™ module that reads and writes LAS and LAZ files, which are common formats for lidar pointcloud and full waveform data.
    Installation: conda install -c conda-forge laspy
    See also: https://github.com/LAStools/LAStools
"""
import laspy

print(f'\nlaspy version: {laspy.__version__:s}') 


# ### 4.10. Xarray

# In[13]:


# Xarray
"""
    Xarray is a Python™ library that provides a common interface for working with n-dimensional arrays and datasets.
    Installation: conda install -c conda-forge xarray dask netCDF4 bottleneck
"""
import xarray

print(f'\nXarray version: {xarray.__version__:s}')


# ## 5.1. Extensions for JupyterLab

# ### 5.11. jupyterlab-git 

# In[14]:


# jupyterlab-git
"""
    jupyterlab-git enables git status support from within JupyterLab.
    Installation: conda install -c conda-forge jupyterlab-git
"""
if platform.system() == "Linux":
    get_ipython().system('conda list | grep jupyterlab-git')
else:
    print('Check if jupyterlab-git is listed when running command "conda list"')


# ### 5.12. jupyterlab-spellchecker

# In[15]:


# jupyterlab-spellchecker
"""
    jupyterlab-spellchecker highlights misspelled words in markdown cells within notebooks and in the text files.
    Installation: conda install -c conda-forge jupyterlab-spellchecker
"""
if platform.system() == "Linux":
    get_ipython().system('conda list | grep jupyterlab-spellchecker')
else:
    print('Check if jupyterlab-spellchecker is listed when running command "conda list"')

