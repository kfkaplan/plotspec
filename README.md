# plotspec
Python code for analyzing reduced 1D and 2D IGRINS spectra

plotspec.py is a python library for viewing and analyzing 1D and 2D spectra for spatially resolved extended objects (ie. nebulae) outputted by the IGRINS Pipeline (PLP) https://github.com/igrins/plp/wiki.
(Note: Currently the version of the PLP that outputs 2D spectra is experimental and not public, please contact Jee-Joon Lee leejjoon@kasi.re.kr if you are interested in obtaining that version of the PLP). 
This code is actively being developed so do not expect everything to work perfectly out of the box right now. 
Please sent all questions and bug reports to kfkaplan@astro.as.utexas.edu.
If you work with IGRINS spectra of extended objects, use this code, and would like to contribute to the development of this code,
let me know and I can add you to the github repository.

* [Requirements](#requirements)
* [Installing](#installing)

## Requirements
* Python (version 2.7)
* DS9 (version 7.2 or later) for displaying 2D spectra
* XPA for scripting DS9
* The following python libraries which can be installed and updated via pip, macports, anaconda, etc.
  * [Astropy] (http://www.astropy.org/) - For handeling fits files
  * [Pylab] (http://wiki.scipy.org/PyLab) - For plotting
  * [Scipy] (http://www.scipy.org/) - For interpolation
  * [Bottleneck] (http://berkeleyanalytics.com/bottleneck/) - Set of fast numerical functions, greatly speeds up the code at various places
  
## Installing
To install simply copy the code off of github and place it in your favorite directory.  Once the code has been copied to your computer
open up plotspec.py and modify the following global variables near the top of the code to match your setup:
* pipeline_path - Set this to the directory of the IGRINS PLP, needed to read in the reduced spectra
* scratch_path - This is a directory where plotspec.py will create and store various temporary files.  Can be set to plotspec's own directory if desired, but you might want to put it somewhere else to avoid clutter.
* data_path - Path to PLP output data directory, you should not need to modify this
* calib_path - Path to PLP  calibration directory, you should not need to modify this
* OH_line_list - Name of OH line list, you should not need to modify this
* default_wave_pivot - Cut in wavelength space between overlapping orders when sitching orders together, 0.0 is blue side, 1.0 is red side, 0.5 is in the middle.  Currently set to 0.625 but you can modify it.
* velocity_range - Positive and negative edge in velocity space (km/s) for interpolating lines onto position-velocity diagrams
* velocity_res - Resolution of velocity grid in km/s, on which all lines are interpolated onto, currently set to 1 km/s
* c - Speed of light, I shouldn't have to say do not modify.
* block - Number of pixels used in the window (or block) for the running median smoothing filter for continuum subtraction, currently set to 300 but you can modify if you wish to tweak the continuum subtraction
* half_block - Half of block, do not modify
