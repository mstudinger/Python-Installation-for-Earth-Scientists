# <div align="center">Python™ for Earth Scientists</div> 
### <div align="center">Package Installation and Verification</div>

Managing Python™ packages can be a daunting task. Often, numerous incompatibilities exist between Python™ packages commonly used in Earth sciences (see example about GDAL and Xarray in Jupyter notebook). Commonly required tasks vary depending on the user's need. Here are some capabilities I need for my workflow that are not covered by the standard [Anaconda](https://www.anaconda.com/) package installation:  
* geodetic 2-D and 3-D coordinate transformations
* geodetic computations
* handling commonly used data formats such as GeoData frames
* image processing using sophisticated tools available in dedicated packages such as OpenCV ([Open Source Computer Vision Library](https://opencv.org/))
* work specific tools, such as AI-based tools e.g., [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
* collaborative code developing and open sharing of code in commonly used repositories such as [GitHub](https://github.com/)
  
This repository contains a Jupyter notebook with instructions for installing essential Python™ packages for Earth scientists and testing the functionality of the installed packages in various computational and operating system environments. The [Jupyter notebook](https://github.com/mstudinger/Python-Test-Tools/blob/main/python_installation_instructions_and_verification.ipynb) has the advantage that individual cells can be run depending on which packages are installed. The Jupyter notebook and corresponding Python™ script have been tested under the following environments:  
* Windows 11 Pro (23H2)        Python™ 3.11.7
* Windows 10 Enterprise (21H2) Python™ 3.9.13
* Linux POSIX Release: 5.15.146.1-microsoft-standard-WSL2 Python™ 3.11.7
* Linux POSIX Release: 5.10.198-187.748.amzn2.x86_64 Python™ 3.11.8 (selected packages tested on the [CryoCloud JupyterHub](https://cryointhecloud.com))
