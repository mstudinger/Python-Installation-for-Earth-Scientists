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

#%% 3) install geopandas and gdal
# conda install geopandas


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

#%% 8) conda install -c conda-forge pycrs

#%% 9) conda install -c conda-forge pymap3d

#%% 10) conda install -c conda-forge cmcrameri


