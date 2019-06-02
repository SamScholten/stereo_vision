# stereo_vision
Stereo Vision Project for The Art of Scientific Computation

+-----------------------------------------------------------------------------

ABOUT:

+-----------------------------------------------------------------------------

The code comprises of 4 parts:
correlation_tools.py:
 - Simple tools for cross correlation in 1 and 2 Dims
calibration_tools.py:
 - Simple tools for calibrating a stereo camera with a grid calibration plate
stereo_camera_tools.py:
 - More complex (read: recursive) tools for image comparison/stereo vision etc.
pattern_detector_tools_1D.py:
 - Simple tools for correlation analysis of large time series datasets.
 - also: quandl_codes.py gives access to the datasets I used.
 - You need a quandl account (free) to access the data, but I've left my access code in the program, feel free to use & abuse as much as you wish.

The stereo_vision.py file acts as a script to represent/exemplify the various possibilities of analysis with these modules.  

+-----------------------------------------------------------------------------

INSTRUCTIONS FOR ENVIRONMENT:

+-----------------------------------------------------------------------------

Sorry, but due to us not finding a Python analog of a function, we decided to pass data through the MATLAB API. If you want to run the program you will need to install it. Thankfully, if you have MATLAB R2016+ installed it is quick. However, on my laptop, the API only worked for Python3.6 (even if the documentation says it should work for Python3.7), so you will need to install my Conda environment (also pretty quick, instructions below). Also, probably better to install the conda environment first, it will isolate your system from anything ungodly I have in my environment.

+-----------------------------------------------------------

CONDA:

+-----------------------------------------------------------

quick and simple Miniconda installation:
    https://docs.conda.io/en/latest/miniconda.html

Then you can import my environment with:
    conda env create --file stereo_vision.yml
{where stereo_vision.yml should be stored in the root directory of whatever I've given you}
Ok from here I haven't tested it good luck. It *should* be downloading/installing the relevant files.

To activate the environment, run: 
    conda activate ENVNAME
I am not sure what ENVNAME should be. On my system it is stereo_vision

You can also just read the .yml file to see the long list of dependencies, but I suspect that the Python3.6 part makes everything a little tricky, and I'm sure you would prefer to go back to your previous system state after purging my stuff.

+-----------------------------------------------------------

MATLAB:

+-----------------------------------------------------------

reference: https://www.scivision.dev/matlab-engine-callable-from-python-how-to-install-and-setup/

0. Install the conda environment first
1. find MATLAB root (MATLAB/R2019a/ or similar):
    matlab -batch matlabroot
2. navigate to \engines\python: 
    cd MATLAB/R2019a/extern/engines/python
3. UNIX:
    python setup.py build --build-base=$(mktemp -d) install
   PC:
    python setup.py build --build-base=%TEMP% install



Cheers had a fun semester :)
