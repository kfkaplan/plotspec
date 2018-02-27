
#This library will eventually be the ultimate IGRINS emission line viewability/analysis code
#
#start as test_new_plotspec.py

#Set matplotlib backend to get around freezing plot windows, first try the one TkAgg
import matplotlib
#matplotlib.use("qt4Agg")

#Import libraries
import os #Import OS library for checking and creating directories
import json #For reading in json files, ie. wavelength solutions given by the PLP not in fits files
from astropy.io import fits #Use astropy for processing fits files
from astropy.modeling import models, fitting #import the astropy model fitting package
import pyregion #For reading in regions from DS9 into python
from pylab import *  #Always import pylab because we use it for everything
from scipy.interpolate import interp1d, UnivariateSpline, griddata #For interpolating
#from scipy.ndimage import zoom #Was used for continuum subtraction at one point, commented out for now
import ds9 #For scripting DS9
#import h2 #For dealing with H2 spectra
import copy #Allow objects to be copied
from scipy.ndimage.filters import median_filter #For cosmic ray removal
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel #For smoothing, not used for now, commented out
from pdb import set_trace as stop #Use stop() for debugging
#ion() #Turn on interactive plotting for matplotlib
from matplotlib.colors import LogNorm #For plotting PV diagrams with imshow
#from numba import jit #Import numba for speeding up some definitions, commented out for now since there is a major error importing numba
from matplotlib.backends.backend_pdf import PdfPages  #For outputting a pdf with multiple pages (or one page)
from pylab import size #For some reason size was not working, so I will import it last
try:  #Try to import bottleneck library, this greatly speeds up things such as nanmedian, nanmax, and nanmin
	from bottleneck import * #Library to speed up some numpy routines
except ImportError:
	print "Bottleneck library not installed.  Code will still run but might be slower.  You can try to bottleneck with 'pip install bottleneck' or 'sudo port install bottleneck' for a speed up."
import matplotlib.gridspec as grd

#Global variables user should set
pipeline_path = '/Volumes/home/plp/'
save_path = '/Volumes/home/results/'
#pipeline_path = '/Volumes/IGRINS_Data/plp/' #Paths for running on linux laptop
#save_path = '/Volumes/IGRINS_Data/results/'
#save_path = '/home/kfkaplan/Desktop/results/'
#pipeline_path = '/Volumes/IGRINS_data_Backup/plp/'
#save_path = '/Volumes/IGRINS_data_Backup/results/' #Define path for saving temporary files'
scratch_path = save_path + 'scratch/' #Define a scratch path for saving some temporary files
if not os.path.exists(scratch_path): #Check if directory exists
	print 'Directory '+ scratch_path + ' does not exist.  Making new directory.'
	os.mkdir(scratch_path) #If path does not exist, make directory
#default_wave_pivot = 0.625 #Scale where overlapping orders (in wavelength space) get stitched (0.0 is blue side, 1.0 is red side, 0.5 is in the middle)
default_wave_pivot = 0.85 #Scale where overlapping orders (in wavelength space) get stitched (0.0 is blue side, 1.0 is red side, 0.5 is in the middle)
set_velocity_range =100.0 # +/- km/s for interpolated velocity grid
set_velocity_res = 1.0 #Resolution of velocity grid
#slit_length = 62 #Number of pixels along slit in both H and K bands
slit_length = 100 #Number of pixels along slit in both H and K bands
block = 750 #Block of pixels used for median smoothing, using iteratively bigger multiples of block
cosmic_horizontal_mask = 3 #Number of pixels to median smooth horizontally (in wavelength space) when searching for cosmics
cosmic_horizontal_limit  = 10.0 #Number of times the data must be above it's own median smoothed self to find cosmic rays
cosmic_s2n_min = 2.5 #Minimum S/N needed to flag a pixel as a cosmic ray

#Global variables, should remain untouched
data_path = pipeline_path + 'outdata/'
calib_path = pipeline_path + 'calib/primary/'
OH_line_list = 'OH_Rousselot_2000.dat' #Read in OH line list
c = 2.99792458e5 #Speed of light in km/s
half_block = block / 2 #Half of the block used for running median smoothing
#slit_length = slit_length - 1 #This is necessary to get the proper indexing


#~~~~~~~~~~~~~~~~~~~~~~~~Make a simple contour plot given three lists~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def contour_plot(x, y, z, nx=100, ny=100, levels=[3.,4.,5.,6.,7.,8.,9.,10.,15.,20.,30.,40.]): #Canned definition to make interpolated contour plots with three lists, based off of http://stackoverflow.com/questions/9008370/python-2d-contour-plot-from-3-lists-x-y-and-rho1
	#if z_range[1] == 0: #Automatically set z range if not provided by user
	#	z_range = [min(z), max(z)]
	xmin, xmax = min(x), max(x)
	ymin, ymax = min(y), max(y)
	#zmin, zmax = min(z), max(z)
	#Set up grid of interpolated points
	xi, yi = linspace(xmin, xmax, nx), linspace(ymin, ymax, ny)
	xi, yi = meshgrid(xi, yi)
	#Interpolate
	#rbf = Rbf(x, y, z, function='linear')
	#zi = rbf(xi, yi)
	zi = griddata((x,y), z, (xi, yi), method='linear')
	#imshow(zi, vmin=z_range[0], vmax=z_range[1], origin='lower',
    #       extent=[xmin, xmax, ymin, ymax], aspect='auto')
	#colorbar()
	scatter(x, y, color='grey', s=7)
	cs = contour(xi,yi,zi, levels=levels, colors='black')
	clabel(cs, inline=1, fontsize=12, fmt='%1.0f')
	#scatter(x, y, color='grey')

#~~~~~~~~~~~~~~~~~~~~~~~~Code for storing information on saved data directories under save_path~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class save_class: #Class stores information on what path to save files to
	def __init__(self):
		self.object = 'scratch' #If no name is set, by default save in the "scratch" directory
		self.set_path()
	def name(self, input): #User can change name of science target in script by saying 'save.name('NGC XXXX')
		self.object = input.replace(' ', '_')
		self.set_path()
	def set_path(self): #Update path to directory to save results in
		self.path = save_path + self.object + '/'
		if not os.path.exists(self.path): #Check if directory exists
			print 'Directory '+ self.path+ ' does not exist.  Making new directory.'
			os.mkdir(self.path) #If path does not exist, make directory
		
save = save_class() #Create object user can change the name to


#~~~~~~~~~~~~~~~Optimized pre-compiled functions ~~~~~~~~~~~~~~~~~~~


#@jit #Fast precompiled function for nanmax for using whole array (no specific axis)
def flat_nanmax(input):
    max = -1e99
    f = input.flat
    for i in f:
        if i > max:
            max = i
    if max==-1e99:
        return nan
    else:
        return max
    
#@jit #Fast precompiled function for nanmin for using whole array (no specific axis)
def flat_nanmin(input):
    min = 1e99
    f = input.flat
    for i in f:
       if i < min:
           min = i
    if min == 1e99:
        return nan
    else:
        return min


#~~~~~~~~~~~~~~~~~~~~~~~~~Code for modifying spectral data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Roll an array (typically an order) an arbitrary number of pixels (via interpolation for fractions of a pixel)
#@jit #Compile Just In Time using numba, for speed up
def roll_interp(array_to_correct, correction, axis=0):
	integer_correction = int(correction) #grab whole number component of correction
	fractional_correction = correction - float(integer_correction) #Grab fractional component of correction (remainder after grabbing whole number out)
	rolled_array =  roll(array_to_correct, integer_correction, axis=axis) #role array the number of pixels matching the integer correction
	if fractional_correction > 0.: #For a positive correction
		rolled_array_plus_one = roll(array_to_correct, integer_correction+1, axis=axis) #Roll array an extra one pixel to the right
	else: #For a negative correction
		rolled_array_plus_one = roll(array_to_correct, integer_correction-1, axis=axis) #Roll array an extra one pixel to the left
	corrected_array = rolled_array*(1.0-abs(fractional_correction)) + rolled_array_plus_one*abs(fractional_correction) #interpolate over the fraction of a pixel
	#stop()
	return corrected_array

#Do a quick preview of a 2D array in DS9
def quicklook(arr, pause=False, close=False): 
		spec_fits = fits.PrimaryHDU(arr) #Create FITS object
		spec_fits.writeto(save.path + 'quicklook.fits', overwrite=True)    #Save temporary fits files for later viewing in DS9
		ds9.open()  #Display spectrum in DS9
		ds9.show(save.path + 'quicklook.fits', new=False)
		ds9.set('zoom to fit')
		ds9.set('scale log') #Set view to log scale
		ds9.set('scale ZScale') #Set scale limits to Zscale, looks okay
		#Pause for viewing if user specified
		if pause:
			wait()
		#Close DS9 after viewing if user specified (pause should be true or else DS9 will open then close)
		if close:
			ds9.close()

#For interpolating an order (or orders) to a new number of pixels in y direction along the slit, use where the H 
##@jit #Fast precompiled function for nanmax for using whole array (no specific axis)
def regrid_slit(ungridded_spectrum, size=slit_length):
	len_y, len_x = shape(ungridded_spectrum) #Get x and y shape of ungridded spectrum
	ungridded_y = arange(len_y) #Get y size of ungridded spectrum
	gridded_y = arange(size) * (float(len_y)/float(size)) #Get size of gridded y
	interp_spectrum = interp1d(ungridded_y, ungridded_spectrum, axis=0, bounds_error=False, kind='nearest') #Create interpolation object
	gridded_spectrum =  interp_spectrum(gridded_y) #Create interpolated specturm (along the slit axis)
	scale = float(len_y) / float(size) #Find factor to scale spectrum down by to account for the fact we have spread it out over a larger area
	gridded_spectrum = gridded_spectrum * scale  #Do the actual scaling
	return gridded_spectrum #Send the now interpolated streatched spectrum back to where it came from



#Roll an array (typically an order) an arbitrary number of pixels to correct flexure
#@jit #Compile Just In Time using numba, for speed up
def flexure(array_to_correct, correction):
	integer_correction = int(correction) #grab whole number component of correction
	fractional_correction = correction - float(integer_correction) #Grab fractional component of correction (remainder after grabbing whole number out)
	rolled_array =  roll(array_to_correct, integer_correction) #role array the number of pixels matching the integer correction
	if fractional_correction > 0.: #For a positive correction
		rolled_array_plus_one = roll(array_to_correct, integer_correction+1) #Roll array an extra one pixel to the right
	else: #For a negative correction
		rolled_array_plus_one = roll(array_to_correct, integer_correction-1) #Roll array an extra one pixel to the left
	corrected_array = rolled_array*(1.0-abs(fractional_correction)) + rolled_array_plus_one*abs(fractional_correction) #interpolate over the fraction of a pixel
	#stop()
	return corrected_array

#Artifically redden a spectrum,
#@jit #Compile Just In Time using numba, for speed up
def redden(B, V, waves, flux):
	alpha = 2.14 #Slope of near-infrared extinction law from Stead & Hoare (2009)
	#alpha = 1.75 #Slope from older literature
	#lambda_H = 1.651 #Effective wavelength of H band determiend from Stead & Hoare (2009)
	#lambda_K = 2.159 #Effective wavelength of K band determiend from Stead & Hoare (2009)
	#lambda_H = 1.662 #Effective wavelength of H band filter given by 2MASS (http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec6_4a.html)
	#lambda_K = 2.159  #Effective wavelength of K band filter given by 2MASS (http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec6_4a.html)
	vega_B = 0.03 #Vega B band mag, from Simbad
	vega_V = 0.03 #Vega V band mag, from simbad
	#vega_H = -0.03 #Vega H band magnitude, from Simbad
	#vega_K = 0.13 #Vega K band magnitude, from Simbaddef
	#E_HK = (H-K) - (vega_H-vega_K)  #Calculate E(H-K) = (H-K)_observed - (H-K)_intrinsic, intrinsic = Vega in this case
	E_BV = (B-V) - (vega_B-vega_V)  #Calculate E(B-V) = (B-V)_observed - (B-V)_intrinsic, intrinsic = Vega in this case
	R = 3.09 #Ratio of total/selective extinction from Rieke & Lebofsky (1985)
	A_V = R * E_BV #Calculate extinction A_V for standard star
	A_lambda = array([ 0.482,  0.282,  0.175,  0.112,  0.058]) #(A_lambda / A_V) extinction curve from Rieke & Lebofsky (1985) Table 3
	l = array([ 0.806,  1.22 ,  1.63 ,  2.19 ,  3.45 ]) #Wavelengths for extinction curve from Rieke & Lebofsky (1985)
	extinction_curve = interp1d(l, A_lambda, kind='quadratic') #Create interpolation object for extinction curve from Rieke & Lebofsky (1985)
	reddened_flux = flux * 10**(-0.4*extinction_curve(waves)*A_V)  #Apply artificial reddening
	#reddened_flux = flux * 10**( -0.4 * (E_HK/(lambda_H**(-alpha)-lambda_K**(-alpha))) * waves**(-alpha) ) #Apply artificial reddening
	#stop()
	return reddened_flux



#Mask Hydrogen absorption lines in A0V standard star continuum, used during relative flux calibration
def mask_hydrogen_lines(wave, flux):
	H_lines = [2.1661, 1.9451, 1.8181, 1.7367, 1.6811, 1.6412, 1.6114, 1.5885, 1.5705, 1.5561, 1.5443, 1.5346, 1.5265, 1.5196] #Wavelengths of H I lines
	d_range = [-0.002 , 0.002] #Wavelength range for masking H I lines 
	for H_wave in H_lines: #For each hydrogen line that might be in the flux array
		mask = (wave > H_wave + d_range[0]) & (wave < H_wave + d_range[1]) #Find pixels in flux array on top of H I line
		flux[mask] = nan #Apply mask
	goodpix = flux > -9e99 #Read in indicies of mask
	#stop()
	#print min(wave), max(wave), len(flux[goodpix]) 
	if len(flux[goodpix]) < 2048: #If any mask is applied (ie. if any H I lines are found in order)
		interpolated_flux = interp1d(wave[goodpix], flux[goodpix], bounds_error = False) #Interpolate over only unmasked pixels
		flux_to_return = interpolated_flux(wave) #Replace masked pixels with a linear interpolation around them
		return flux_to_return #Return now masked pixels
	else:
		return flux #If nothing is masked, return the flux unmodified




#Function normalizes A0V standard star spectrum, for later telluric correction, or relative flux calibration
def telluric_and_flux_calib(sci, std, std_flattened, calibration=[], B=0.0, V=0.0, y_scale=1.0, y_power=1.0, y_sharpen=0., wave_smooth=0.0, delta_v=0.0, quality_cut = False, no_flux = False, savechecks=True, telluric_power=1.0, telluric_spectrum=[], std_shift=0.0):
	# #Read in Vega Data
	std.combine_orders() #Combine orders for standard star specturm for later plotting
	vega_file = pipeline_path + 'master_calib/A0V/vegallpr25.50000resam5' #Directory storing Vega standard spectrum     #Set up reading in Vega spectrum
	vega_wave, vega_flux, vega_cont = loadtxt(vega_file, unpack=True) #Read in Vega spectrum
	vega_wave = (vega_wave / 1e3)*(1.0 + std_shift/c) #convert angstroms to microns and shift wavelengths if a velocity correction is given by the user
	waves = arange(1.4, 2.5, 0.000005) #Array to store HI lines
	HI_line_profiles = ones(len(waves)) #Array to store synthetic (ie. scaled vega) H I lines
	x = [1.4, 1.5, 1.6, 1.62487, 1.66142, 1.7, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5] #Coordinates tracing continuum of Vega, taken between H I lines in the model spectrum vegallpr25.50000resam5
	y = [2493670., 1950210., 1584670., 1512410., 1406170. , 1293900., 854857., 706839., 589023., 494054., 417965., 356822., 306391.]
	interpolate_vega_continuum = interp1d(x, y, kind='cubic', bounds_error=False) #Create interpolation object for Vega continuum defined by coordinates above
	interpolated_vega_continuum = interpolate_vega_continuum(waves) #grab interpolated continuum once so dn't have to interpolate it again
	continuum_normalized_vega_flux = vega_flux / interpolate_vega_continuum(vega_wave) #Normalzie synthetic Vega spectrum by its own continuum
	#interpolate_regular_vega_spectrum = interp1d(vega_wave, 1.0 +  (continuum_normalized_vega_flux-1.0),  bounds_error=False) #Divide out continnum and interpolate H I lines, allowing for the H I lines to scale by y_scale
	if size(y_scale) == 2 or size(y_power) == 2 or size(y_sharpen) == 2 or size(wave_smooth) == 2: #if there are two sets of inputs for modifying the H I lines in the synthetic Vega spectrum, run twice and average the two together
		if size(y_scale) == 1: y_scale = [y_scale, y_scale] #If there is only one input for these parameters, just make two indentical versions of it to run it twice easily
		if size(y_power) == 1: y_power = [y_power, y_power]
		if size(y_scale) == 1: y_sharpen = [y_sharpen, y_sharpen]
		if size(y_scale) == 1: wave_smooth = [wave_smooth, wave_smooth]
		if y_sharpen[0] > 0.: #If user specifies they want to sharpen the H I lines in the synthetic Vega spectrum, smaller sharp numbers will sharpen the lines more, best to start with a very large number
			g = Gaussian1DKernel(stddev = y_sharpen[0]) #Set up gaussian smoothing for Vega I lines, here sharp = std deviation in pixels of gaussian used for smoothing
			smooothed_interpolated_vega_lines	= convolve(continuum_normalized_vega_flux, g) - 1.0 #Smooth the vega lines, subtract one to put continuum for the smoothed liens on the x axis
			intepolate_vega_lines = interp1d(vega_wave, 1.0 + y_scale[0] * ((continuum_normalized_vega_flux-smooothed_interpolated_vega_lines)**y_power[0]-1.0),  bounds_error=False) #Divide out continnum and interpolate H I lines, allowing for the H I lines to scale by y_scale
		else: #Ignore sharpening and just use the old method (most common action taken)
			intepolate_vega_lines = interp1d(vega_wave, 1.0 + y_scale[0] * (continuum_normalized_vega_flux**y_power[0]-1.0),  bounds_error=False) #Divide out continnum and interpolate H I lines, allowing for the H I lines to scale by y_scale
		interpolated_vega_lines_0 =  intepolate_vega_lines(waves) #Grab interpoalted lines once			
		if y_sharpen[1] > 0.: #If user specifies they want to sharpen the H I lines in the synthetic Vega spectrum, smaller sharp numbers will sharpen the lines more, best to start with a very large number
			g = Gaussian1DKernel(stddev = y_sharpen[1]) #Set up gaussian smoothing for Vega I lines, here sharp = std deviation in pixels of gaussian used for smoothing
			smooothed_interpolated_vega_lines	= convolve(continuum_normalized_vega_flux, g) - 1.0 #Smooth the vega lines, subtract one to put conhttp://www.u.arizona.edu/~kfkaplan/hpf/20180129/figures/0004.Slope-20180129T014837_R01.optimal.pngtinuum for the smoothed liens on the x axis
			intepolate_vega_lines = interp1d(vega_wave, 1.0 + y_scale[1] * ((continuum_normalized_vega_flux-smooothed_interpolated_vega_lines)**y_power[1]-1.0),  bounds_error=False) #Divide out continnum and interpolate H I lines, allowing for the H I lines to scale by y_scale
		else: #Ignore sharpening and just use the old method (most common action taken)
			intepolate_vega_lines = interp1d(vega_wave, 1.0 + y_scale[1] * (continuum_normalized_vega_flux**y_power[1]-1.0),  bounds_error=False) #Divide out continnum and interpolate H I lines, allowing for the H I lines to scale by y_scale
		interpolated_vega_lines_1 =  intepolate_vega_lines(waves) #Grab interpoalted lines twice			
		interpolated_vega_lines = interpolated_vega_lines_0 + interpolated_vega_lines_1 - 1.0 #Average two sets of modified Vega H I lines together into one synthetic set of lines
	else: #If there is only one set of inputs for modifying the Vega synetheic spectrum H I lines, just run the single input (This is generally the default you want to do, the thing above is an added complciation)
		if y_sharpen > 0.: #If user specifies they want to sharpen the H I lines in the synthetic Vega spectrum, smaller sharp numbers will sharpen the lines more, best to start with a very large number
			g = Gaussian1DKernel(stddev = y_sharpen) #Set up gaussian smoothing for Vega I lines, here sharp = std deviation in pixels of gaussian used for smoothing
			smooothed_interpolated_vega_lines	= convolve(continuum_normalized_vega_flux, g) - 1.0 #Smooth the vega lines, subtract one to put continuum for the smoothed liens on the x axis
			intepolate_vega_lines = interp1d(vega_wave, 1.0 + y_scale * ((continuum_normalized_vega_flux-smooothed_interpolated_vega_lines)**y_power-1.0),  bounds_error=False) #Divide out continnum and interpolate H I lines, allowing for the H I lines to scale by y_scale
		else: #Ignore sharpening and just use the old method (most common action taken)
			intepolate_vega_lines = interp1d(vega_wave, 1.0 + y_scale * (continuum_normalized_vega_flux**y_power-1.0),  bounds_error=False) #Divide out continnum and interpolate H I lines, allowing for the H I lines to scale by y_scale
		interpolated_vega_lines =  intepolate_vega_lines(waves) #Grab interpoalted lines once
	a0v_synth_cont =  interp1d(waves, redden(B, V, waves, interpolated_vega_continuum), kind='linear', bounds_error=False) #Paint H I line profiles onto Vega continuum to create a synthetic A0V spectrum (not yet reddened)
	#STOP
	if wave_smooth > 0.:  #If user specifies they want to gaussian smooth the synthetic pectrum
		g = Gaussian1DKernel(stddev = wave_smooth) #Set up gaussian smoothing for Vega I lines, here wave_smooth = std deviation in pixels of gaussian used for smoothing
		a0v_synth_spec =  interp1d(waves, redden(B, V, waves, convolve(interpolated_vega_lines*interpolated_vega_continuum, g)), kind='linear', bounds_error=False) #Artifically redden synthetic A0V spectrum to match standard star observed
	else: #If no smoothing 
		a0v_synth_spec =  interp1d(waves, redden(B, V, waves, interpolated_vega_lines*interpolated_vega_continuum), kind='linear', bounds_error=False) #Artificially redden model Vega spectrum to match A0V star observed
	#Onto calibrations...
	num_dimensions = ndim(sci.orders[0].flux) #Store number of dimensions
	if num_dimensions == 2: #If number of dimensions is 2D
		slit_pixel_length = len(sci.orders[0].flux[:,0]) #Height of slit in pixels for this target and band
	if savechecks: #If user specifies saving pdf check files 
		with PdfPages(save.path + 'check_flux_calib.pdf') as pdf: #Load pdf backend for saving multipage pdfs
			#Plot easy preview check of how well the H I lines are being corrected
			clf() #Clear page first
			expected_continuum = copy.deepcopy(std_flattened) #Create object to store the "expected continuum" which will end up being the average of each order's adjacent blaze functions from what the PLP thinks the blaze is for the standard star
			g = Gaussian1DKernel(stddev=5.0) #Do a little bit of smoothing of the blaze functions
			for i in range(2,std.n_orders-2): #Loop through each order
			        adjacent_orders = array([convolve(std.orders[i-1].flux/std_flattened.orders[i-1].flux, g),   #Combine the order before and after the current order, while applying a small amount of smoothing
			                                 convolve(std.orders[i+1].flux/std_flattened.orders[i+1].flux, g),])
			        mean_order = nanmean(adjacent_orders, axis=0) #Smooth the before and after order blazes together to estimate what we think the continuum/blaze should be
			        expected_continuum.orders[i].flux = mean_order #Save the expected continuum
			expected_continuum.combine_orders()#Combine all the orders in the expected continuum
			HI_line_waves = [2.166120, 1.7366850, 1.6811111, 1.5884880] #Wavelengths of H I lines will be previewing
			HI_line_labes = ['Br-gamma','Br-10','Br-11', 'Br-14'] #Names of H I lines we will be previewing
			delta_wave = 0.012 # +/- wavelength range to plot on the xaxis of each line preview
			n_HI_lines = len(HI_line_waves) #Count up how many H I lines we will be plotting
			subplots(nrows=2, ncols=2) #Set up subplots
			figtext(0.02,0.5,r"Flux", fontsize=20,rotation=90) #Set shared y-axis label
			figtext(0.4,0.02,r"Wavelength [$\mu$m]", fontsize=20,rotation=0) #Set shared x-axis label
			figtext(0.05,0.95,r"Check AOV H I line fits (y-scale: "+str(y_scale)+", y-power: "+str(y_power)+", y_sharpen: "+str(y_sharpen)+" wave_smooth: "+str(wave_smooth)+", std_shift: "+str(std_shift)+")", fontsize=12,rotation=0) #Shared title
			waves = std.combospec.wave #Wavelength array to interpolate to
			normalized_HI_lines = a0v_synth_cont(waves)/a0v_synth_spec(waves) #Get normalized lines to the wavelength array
			for i in xrange(n_HI_lines): #Loop through each H I line we want to preview
				subplot(2,2,i+1) #Set up current line's subplot
				#tight_layout(pad=5) #Use tightlayout so things don't overlap
				fig = gcf()#Adjust aspect ratio
				fig.set_size_inches([15,10]) #Adjust aspect ratio
				plot(std.combospec.wave, std.combospec.flux, label='H I Uncorrected', color='gray') #Plot raw A0V spectrum, no H I correction applied
				plot(std.combospec.wave, std.combospec.flux*normalized_HI_lines, label='H I Corrected',color='black') #Plot raw A0V spectrum with H I correction applied
				plot(expected_continuum.combospec.wave, expected_continuum.combospec.flux, label='Expected Continuum', color='blue') #Plot expected continuu, which the average of each order's adjacent A0V continnua
				xlim(HI_line_waves[i]-delta_wave, HI_line_waves[i]+delta_wave) #Set x axis range
				j = (std.combospec.wave > HI_line_waves[i]-delta_wave) & (std.combospec.wave < HI_line_waves[i]+delta_wave) #Find only pixels in window of x-axis range for automatically determining y axis range
				max_flux = nanmax(std.combospec.flux[j]*normalized_HI_lines[j]) #Min y axis range
				min_flux = nanmin(std.combospec.flux[j]*normalized_HI_lines[j]) #Max y axis range
				ylim([0.9*min_flux,1.02*max_flux]) #Set y axis range
				title(HI_line_labes[i]) #Set title
				if i==n_HI_lines-1: #If last line is being plotted
					legend(loc='lower right') #plot the legend
			tight_layout(pad=4)
			pdf.savefig() #Save plots showing how well the H I correciton (scaling H I lines from Vega) fits
			clf() #Plot Vega model spectrum on second page
			plot(vega_wave, vega_flux, '--', color='blue', label='Model Vega Spectrum') #Plot vega model
			premake_a0v_synth_cont = a0v_synth_cont(waves) #Load interpolated synthetic A0V spectrum into memory
			plot(waves,premake_a0v_synth_cont, color='black', label='Synethic A0V Continuum') #Plot synthetic A0V continuum
			xlim([flat_nanmin(waves),flat_nanmax(waves)]) #Set limits on plot
			ylim([0., flat_nanmax(premake_a0v_synth_cont)])
			xlabel(r'Wavelength [$\mu$m]')
			ylabel(r'Relative Flux')
			title('Check A0V Reddening (B='+str(B)+', V='+str(V)+')')
			legend(loc="upper right")
			tight_layout()
			pdf.savefig()  #Save showing synthetic A0V spectrum that the data will be divided by to do relative flux calibration & telluric correction on second page of PDF
	for i in xrange(std.n_orders): #Loop through and plot each order for the observed A0V, along with the corrected H I absorption to see how well the synthetic A0V spectrum fits
		if quality_cut: #Generally we throw out bad pixels, but the user can turn this feature off by setting quality_cut = False
			std.orders[i].flux[std_flattened.orders[i].flux <= .1] = nan #Mask out bad pixels
		waves = std.orders[i].wave #Std wavelengths
		std_flux = std.orders[i].flux #Std flux
		if telluric_spectrum == []: #If user does not specifiy a telluric spectrum directly
			telluric_flux = std_flattened.orders[i].flux #Use the flatteneed standard flux given by the PLP, used for scaling telluric lines
		else: #But if the user does specify a telluric spectrum object
			telluric_flux = telluric_spectrum.orders[i].flux #use that object given by the user instead 
		interpolated_a0v_synth_spec = a0v_synth_spec(waves) #Grab synthetic A0V spectrum across current order
		if calibration != []: #If user specifies they are using their own calibration: WARNING FOR TESTING PURPOSES ONLY
			relative_flux_calibration = calibration.orders[i].flux #Then use the calibration given by the user
		else: #Or else use the default calibration
			#relative_flux_calibration = (std_flux * (telluric_flux**(telluric_power-1.0))/ interpolated_a0v_synth_spec)				
			relative_flux_calibration = std_flux / interpolated_a0v_synth_spec
		#s2n =  1.0/sqrt(sci.orders[i].s2n()**-2 + std.orders[i].s2n()**-2)  #Error propogation after telluric correction, see https://wikis.utexas.edu/display/IGRINS/FAQ or http://chemwiki.ucdavis.edu/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error#Arithmetic_Error_Propagation
		s2n =  1.0/sqrt((1.0/sci.orders[i].s2n()**2) + (1.0/std.orders[i].s2n()**2))  #Error propogation after telluric correction, see https://wikis.utexas.edu/display/IGRINS/FAQ or http://chemwiki.ucdavis.edu/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error#Arithmetic_Error_Propagation
		if not no_flux: #As long as user does not specify doing a flux calibration
			sci.orders[i].flux /= relative_flux_calibration   #Apply telluric correction and flux calibration
		sci.orders[i].noise = sci.orders[i].flux / s2n #It's easiest to just work back the noise from S/N after calculating S/N, plus it is now properly scaled to match the (relative) flux calibrati
	return(sci) #Return the spectrum object (1D or 2D) that is now flux calibrated and telluric corrected


#Class creates, stores, and displays lines as position velocity diagrams, one of the main tools for analysis
class position_velocity:
	def __init__(self, input_spec1d, input_spec2d, line_list, make_1d=False, make_1d_y_range=[0,0], shift_lines='', velocity_range=set_velocity_range, velocity_res=set_velocity_res):
		spec1d = copy.deepcopy(input_spec1d)
		spec2d = copy.deepcopy(input_spec2d)
		slit_pixel_length = shape(spec2d.flux)[0] #Height of slit in pixels for this target and band
		wave_pixels = spec2d.wave #Extract 1D wavelength for each pixel
		x = arange(len(wave_pixels)) + 1.0 #Number of pixels across detector
		interp_velocity = arange(-velocity_range, velocity_range, velocity_res) #Velocity grid to interpolate each line onto
		show_lines = line_list.parse(flat_nanmin(wave_pixels), flat_nanmax(wave_pixels)) #Only grab lines withen the wavelength range of the current order
		n_lines = len(show_lines.wave) #Number of spectral lines
		n_velocity = len(interp_velocity) #Number of velocity points
		flux = empty([n_lines, n_velocity]) #Set up list of arrays to store 1D fluxes
		var1d = empty([n_lines, n_velocity])
		pv = empty([n_lines, slit_pixel_length, n_velocity])
		var2d =  empty([n_lines, slit_pixel_length, n_velocity])
		if shift_lines != '': #If user wants to apply a correction in velocity space to a set of lines, use this file shift_lines
			shift_labels = loadtxt(shift_lines, usecols=[0,], dtype=str, delimiter='\t') #Load line labels to ID each line
			shift_v = loadtxt(shift_lines, usecols=[1,], dtype=float, delimiter='\t') #Load km/s to artifically doppler shift spectral lines
			save_shift_wave = zeros(n_lines) #Create arrays to store shifted wavelengths and velocities
			save_shift_v = zeros(n_lines)
			for i in xrange(len(shift_labels)): #Go through each line and shift it's wavlength
				find_line = show_lines.label == shift_labels[i]
				if any(find_line): #Only run if a line is found
					pre_shfited_wavelength = copy.deepcopy(show_lines.wave[find_line]) #Save pre shifted wavelength
					show_lines.wave[find_line] = show_lines.wave[find_line] * (-(shift_v[i]/c)+1.0)#Artifically doppler shift the line
					save_shift_v[find_line] = shift_v[i] #Save velocity shift
					save_shift_wave[find_line] = show_lines.wave[find_line] - pre_shfited_wavelength #save wavelength shift
				#print shift_labels[i]
		for i in xrange(n_lines): #Label the lines
			pv_velocity = c * ( (wave_pixels / show_lines.wave[i]) - 1.0 ) #Calculate velocity offset for each pixel from c*delta_wave / wave
			pixel_cut = abs(pv_velocity) <= velocity_range #Find only pixels in the velocity range, this is for conserving flux
			ungridded_wavelengths = wave_pixels[pixel_cut]
			ungridded_velocities = pv_velocity[pixel_cut]
			ungridded_flux_1d = spec1d.flux[pixel_cut] #PV diagram ungridded on origional pixels
			ungridded_flux_2d = spec2d.flux[:,pixel_cut] #PV diagram ungridded on origional pixels			
			ungridded_variance_1d = spec1d.noise[pixel_cut]**2 #PV diagram variance ungridded on original pixesl
			ungridded_variance_2d = spec2d.noise[:,pixel_cut]**2 #PV diagram variance ungridded on original pixels
			interp_wave = interp1d(ungridded_velocities, ungridded_wavelengths, kind='linear', bounds_error=False) #Create interp obj for wavelengths
			interp_flux_2d = interp1d(ungridded_velocities, ungridded_flux_2d, kind='linear', bounds_error=False) #Create interp obj for 2D flux
			interp_variance_2d = interp1d(ungridded_velocities, ungridded_variance_2d, kind='linear', bounds_error=False) #Create interp obj for 2D variance
			gridded_wavelengths = interp_wave(interp_velocity) #Get wavelengths as they appear on the velocity grid
			dl_dv = gridded_wavelengths[1:] - gridded_wavelengths[:len(gridded_wavelengths)-1] / set_velocity_res #Calculate scale factor delta-lambda/delta-velocity for conserving flux when interpolating from the wavleength grid to velocity grid
			dl_dv = hstack([dl_dv, dl_dv[len(dl_dv)-1]]) #Add an extra thing at the end of the delta-lambda/delta-velocity array so that it has an equal number of elements as everything else here
 			gridded_flux_2d = interp_flux_2d(interp_velocity) * dl_dv #PV diagram velocity gridded	
			gridded_variance_2d = interp_variance_2d(interp_velocity) * dl_dv**2 #PV diagram variance velocity gridded
			if not make_1d: #By default use the 1D spectrum outputted by the pipeline, but....
				gridded_flux_1d = interp(interp_velocity, ungridded_velocities, ungridded_flux_1d) * dl_dv
				gridded_variance_1d =  interp(interp_velocity, ungridded_velocities, ungridded_variance_1d) * dl_dv**2
			elif make_1d_y_range[1] > 0: #... if user sets make_1d = True, then we will create our own 1D spectrum by collapsing the 2D spectrum
				gridded_flux_1d = nansum(gridded_flux_2d[make_1d_y_range[0]:make_1d_y_range[1],:], 0) * dl_dv #Create 1D spectrum by collapsing 2D spectrum
				gridded_variance_1d = nansum(gridded_variance_2d[make_1d_y_range[0]:make_1d_y_range[1],:], 0) * dl_dv**2 #Create 1D variance spectrum by collapsing 2D variance
			else:
				gridded_flux_1d = nansum(gridded_flux_2d, 0) * dl_dv #Create 1D spectrum by collapsing 2D spectrum
				gridded_variance_1d = nansum(gridded_variance_2d, 0) * dl_dv**2 #Create 1D variance spectrum by collapsing 2D variance
			flux[i,:] = gridded_flux_1d# *  scale_flux_1d #Append 1D flux array with line
			var1d[i,:] = gridded_variance_1d# * scale_flux_1d #Append 1D variacne array with line
			pv[i,:,:] = gridded_flux_2d# * scale_flux_2d #Stack PV spectrum of lines into a datacube
			var2d[i,:,:] = gridded_variance_2d# * scale_variance_2d  #Stack PV variance of lines into a datacube
		if shift_lines != '': #If lines were shifted, save the velocities and wavlengths that were shifted in this PV object
			self.shift_v = save_shift_v
			self.shift_wave = save_shift_wave
		self.flux = flux #Save 1D PV fluxes
		self.var1d = var1d #Save 1D PV variances
		self.pv = pv #Save datacube of stack of 2D PV diagrams for each line
		self.var2d = var2d #Save 2D PV variance
		self.velocity = interp_velocity #Save aray storing velocity grid all lines were interpolated onto
		self.label = show_lines.label #Save line labels
		self.lab_wave = show_lines.lab_wave #Save lab wavelengths for all the lines
		self.wave = show_lines.wave #Save line wavelengths
		self.n_lines = len(self.flux) #Count number of individual spectral lines lines stored in position velocity object
		self.slit_pixel_length = slit_pixel_length #Store number of pixels along slit
		self.velocity_range = velocity_range #Store velocity range of PV diagrams
		self.velocity_res = velocity_res #Store velocity resoultion (km/s)
	def view(self, line='', wave=0.0,  pause = False, close = False, printlines=False, name='pv'): #Function loads 2D PV diagrams in DS9 and plots 1D diagrams
		self.save_fits() #Save a fits file of the pv diagrams for opening in DS9
		ds9.open() #Open DS9
		ds9.show(save.path + name + '.fits', new = False) #Load PV diagrams into DS9
		ds9.set('zoom to fit') #Zoom PV diagram to fit ds9 window
		ds9.set('zoom 0.9') #Zoom out a little bit to see the coordinate grid
		ds9.set('scale log') #Set view to log scale
		ds9.set('scale Zscale') #Set scale limits to Zscale, looks okay
		ds9.set('grid on')  #Turn on coordinate grid to position velocity coordinates
		ds9.set('grid type publication') #Set to the more aesthetically pleasing publication type grid
		ds9.set('grid system wcs') #Set to position vs. velocity coordinates
		ds9.set('grid axes type exterior') #Set grid axes to be exterior
		ds9.set('grid axes style 1') #Set grid axes to be "pulbication" type
		ds9.set('grid numerics type exterior') #Put numbers external to PV diagram
		ds9.set('grid numerics color black') #Make numbers black for easier reading
		if printlines: #If user sets printlines = True, list lines and their index in the command line
			print 'Lines are....'
			for i in xrange(self.n_lines): #Loop through each line
				print i+1, self.label[i] #Print index and label for line in terminal
		if line != '': #If user specifies line name, find index of that line and dispaly it
			self.goline(line)
		if wave != 0.0: #If user specifies wavelength, find nearest wavelength for that line being specified and display it
			self.gowave(wave)       
		#Pause for viewing if user specified
		if pause:
			wait()
		#Close DS9 after viewing if user specified (pause should be true or else DS9 will open then close)
		if close:
			ds9.close()
	def goline(self, line): #Function causes DS9 to display a specified line (PV diagram must be already loaded up using self.view()
		try:
			if line != '': #If user specifies line name, find index of that line
				i = 1 + where(self.label == line)[0][0] #Find instance of line matching the provided name
				self.display_line(i)
			else: #If index not provided
				print 'ERROR: No line label specified'
		except IndexError: #If line is unable to be found (ie. not in current band) catch and print the following error...
			print 'ERROR: Unable to find the specified line in this spectrum.  Please try again.'
	def gowave(self, wave): #Function causes DS9 to display a specified line (PV diagram must be already loaded up using self.view()
		if wave != 0.0: #If user specifies line name, find index of that line
			nearest_wave = abs(self.lab_wave - wave).min() #Grab nearest wavelength
			i = 1 + where(abs(self.lab_wave-wave) == nearest_wave)[0][0] #Grab index for line with nearest wavelength
			self.display_line(i)
		else:
			print 'ERROR: No line wavelength specified'
	def display_line(self, i): #Moves DS9 to display correct 2D PV diagram of line, and also displays 1D line
		label_string = self.label[i-1]
		wave_string = "%12.5f" % self.lab_wave[i-1]
		title = label_string + '   ' + wave_string + ' $\mu$m'
		ds9.set('cube '+str(i)) #Go to line in ds9 specified by user in 
		self.plot_1d_velocity(i-1, title = title)
	def make_1D_postage_stamps(self, pdf_file_name): #Make a PDF showing all 1D lines in a single PDF file
		with PdfPages(save.path + pdf_file_name) as pdf: #Make a multipage pd
			for i in xrange(self.n_lines):
				label_string = self.label[i]
				wave_string = "%12.5f" % self.lab_wave[i]
				title = label_string + '   ' + wave_string + ' $\mu$m'
				self.plot_1d_velocity(i, title=title) #Make 1D plot postage stamp of line
				pdf.savefig() #Save as a page in a PDF file
	def make_2D_postage_stamps(self, pdf_file_name): #Make a PDF showing all 2D lines in a single PDF file
		#figure(figsize=(2,1), frameon=False)
		with PdfPages(save.path + pdf_file_name) as pdf: #Make a multipage pd
			for i in xrange(self.n_lines):
				label_string = self.label[i]
				wave_string = "%12.5f" % self.lab_wave[i]
				title = label_string + '   ' + wave_string + ' $\mu$m'
				#self.plot_1d_velocity(i, title=title) #Make 1D plot postage stamp of line
				frame = gca() #Turn off axis number labels
				frame.axes.get_xaxis().set_ticks([]) #Turn off axis number labels
				frame.axes.get_yaxis().set_ticks([]) #Turn off axis number labels
				ax = subplot(111)
				suptitle(title)
				imshow(self.pv[i,:,:], cmap='gray')
				pdf.savefig() #Save as a page in a PDF file
	def plot_1d_velocity(self, line_index, title='', clear=True, fontsize=18, show_zero=True, show_x_label=True, show_y_label=True, uncertainity_color='red', y_max=0., scale_flux=1.0/1e3, uncertainity_line='solid'): #Plot 1D spectrum in velocity space (corrisponding to a PV Diagram), called when viewing a line
		if clear: #Clear plot space, unless usser sets clear=False
			clf() #Clear plot space
		velocity = self.velocity
		flux = self.flux[line_index] * scale_flux #Scale flux so numbers are not so big
		noise = sqrt(self.var1d[line_index]) * scale_flux
		max_flux = nanmax(flux + noise, axis=0) #Find maximum flux in slice of spectrum
		fill_between(velocity, flux - noise, flux + noise, facecolor = uncertainity_color, linestyle=uncertainity_line) #Fill in space between data and +/- 1 sigma uncertainity
		plot(velocity, flux, color='black') #Plot 1D spectrum slice
		#plot(velocity, flux + noise, ':', color='red') #Plot noise level for 1D spectrum slice
		#plot(velocity, flux - noise, ':', color='red') #Plot noise level for 1D spectrum slice
		if show_zero: #Normally show the zero point line, but if user does not want it, don't plot it
			plot([0,0], [-0.2*max_flux, max_flux], '--', color='black') #Plot velocity zero point
		xlim([-self.velocity_range, self.velocity_range]) #Set xrange to be +/- the velocity range set for the PV diagrams
		if y_max == 0.: #If user specifies no maximum y scale
			ylim([-0.20*max_flux, 1.2*max_flux]) #Set yrange automatically
		else: #If user specifies a maximum y sclae
			ylim([-0.10*y_max, y_max]) #Base y range on what the user set the y_max to be
		if title != '': #Add title to plot showing line name, wavelength, etc.
			suptitle(title, fontsize=20)
		#if label != '' and wave > 0.0:
			#title(label + ' ' + "%12.5f" % wave + '$\mu$m')
		#elif label != '':
			#title(label)
		#elif wave > 0.0:
			#title("%12.5f" % wave + '$\mu$m')
		if show_x_label: #Let user specificy showing x axis
			xlabel('Velocity [km s$^{-1}$]', fontsize=fontsize) #Label x axis
		#if self.s2n:
		#	ylabel('S/N per resolution element (~3.3 pixels)', fontsize=18) #Label y axis as S/N for S/N spectrum
		#else:
		#	ylabel('Relative Flux', fontsize=18) #Or just label y-axis as relative flux
		if show_y_label:
			ylabel('Relative Flux', fontsize=fontsize) #Or just label y-axis as relative flux 
		#draw()
		#show()
	def save_fits(self, name='pv', dim=2, type='flux'): #Save fits file of PV diagrams
		if type == 'flux' and dim == 2: #If user specifies 2D flux (default)
			pv_file = fits.PrimaryHDU(self.pv) #Set up fits file object to hold 2D flux
		elif type == 'var' and dim == 2: #If user specifies 2D variance
			pv_file = fits.PrimaryHDU(self.var2d) #Set up fits file object to hold 2D variance
		elif type == 'flux' and dim == 1: #If user specifies 1D flux
			pv_file = fits.PrimaryHDU(self.flux)
		elif type == 'var' and dim == 1: #If user specifies 1D variance
			pv_file = fits.PrimaryHDU(self.var1d)
		else: #Report error
			print 'ERROR: Type '+ type + ' or dimension' + str(dim) + 'for saving fits file not correctly specified.'
		#Add WCS for linear interpolated velocity
		pv_file.header['CTYPE1'] = 'km/s' #Set unit to "Optical velocity" (I know it's really NIR but whatever...)
		pv_file.header['CRPIX1'] = (self.velocity_range / self.velocity_res) + 1 #Set zero point to where v=0 km/s (middle of stamp)
		pv_file.header['CDELT1'] = self.velocity_res #Set zero point to where v=0 km/s (middle of stamp)
		pv_file.header['CUNIT1'] = 'km/s' #Set label for x axis to be km/s
		if dim == 2:
			pv_file.header['CTYPE2'] = 'Slit Position' #Set unit for slit length to something generic
			pv_file.header['CRPIX2'] = 1 #Set zero point to 0 pixel for slit length
			pv_file.header['CDELT2'] = 1.0 / self.slit_pixel_length #Set slit length to go from 0->1 so user knows what fraction from the bottom they are along the slit
		pv_file.writeto(save.path + name +'.fits', overwrite  = True) #Save fits file
		# 	s2n_file = fits.PrimaryHDU(self.s2n) #Set up fits file object
		# 	#Add WCS for linear interpolated velocity
		# 	s2n_file.header['CTYPE1'] = 'km/s' #Set unit to "Optical velocity" (I know it's really NIR but whatever...)
		# 	s2n_file.header['CRPIX1'] = (self.velocity_range / self.velocity_res) + 1 #Set zero point to where v=0 km/s (middle of stamp)
		# 	s2n_file.header['CDELT1'] = self.velocity_res #Set zero point to where v=0 km/s (middle of stamp)
		# 	s2n_file.header['CUNIT1'] = 'km/s' #Set label for x axis to be km/s
		# 	s2n_file.header['CTYPE2'] = 'Slit Position' #Set unit for slit length to something generic
		# 	s2n_file.header['CRPIX2'] = 1 #Set zero point to 0 pixel for slit length
		# 	s2n_file.header['CDELT2'] = 1.0 / self.slit_pixel_length #Set slit length to go from 0->1 so user knows what fraction from the bottom they are along the slit
		# 	s2n_file.writeto(scratch_path + 'pv_s2n.fits', overwrite  = True) #Save fits file
	def save_var(self, name='pv_var2d'): #Save fits file of PV diagrams variance, OLD!   USE def save_fits above, kept here for compatibility
		pv_file = fits.PrimaryHDU(self.var2d) #Set up fits file object
		#Add WCS for linear interpolated velocity
		pv_file.header['CTYPE1'] = 'km/s' #Set unit to "Optical velocity" (I know it's really NIR but whatever...)
		pv_file.header['CRPIX1'] = (self.velocity_range / self.velocity_res) + 1 #Set zero point to where v=0 km/s (middle of stamp)
		pv_file.header['CDELT1'] = self.velocity_res #Set zero point to where v=0 km/s (middle of stamp)
		pv_file.header['CUNIT1'] = 'km/s' #Set label for x axis to be km/s
		pv_file.header['CTYPE2'] = 'Slit Position' #Set unit for slit length to something generic
		pv_file.header['CRPIX2'] = 1 #Set zero point to 0 pixel for slit length
		pv_file.header['CDELT2'] = 1.0 / self.slit_pixel_length #Set slit length to go from 0->1 so user knows what fraction from the bottom they are along the slit
		pv_file.writeto(save.path + name + '.fits', overwrite  = True) #Save fits file
	def getline(self, line): #Grabs PV diagram for a single line given a line label
		i =  where(self.label == line)[0][0] #Search for line by label
		return self.pv[i] #Return line found
	def getvariance(self,line): #Grabs PV diagram variasnce for a single line given a line label
		i =  where(self.label == line)[0][0] #Search for line by label
		return self.var2d[i] #Return variance of line found
	def getline1d(self, line): #Grabs 1D flux in velocity space for a single line given a line label
		i =  where(self.label == line)[0][0] #Search for line by label
		return self.flux[i] #Return line found
	def getvariance1d(self,line): #Grabs 1D  variance in velocity space for a single line given a line label
		i =  where(self.label == line)[0][0] #Search for line by label
		return self.var1d[i] #Return variance of line found
	def ratio(self, numerator, denominator):  #Returns PV diagram of a line ratio
		return self.getline(numerator) / self.getline(denominator)
	def normalize(self, line): #Normalize all PV diagrams by a single line
		norm_flux_2d = self.getline(line) #Grab flux (in PV space) of line to normalize by
		norm_var_2d =  self.getvariance(line) #Grab variance (in PV space) of line to normalize by
		norm_flux_1d = self.getline1d(line) #Grab 1D flux (in vel. space) of line to normalize by
		norm_var_1d =  self.getvariance1d(line) #Grab 1D variance (in vel. space) of line to normalize by
		self.var2d = (self.pv/norm_flux_2d)**2 * ((self.var2d/self.pv**2) + (norm_var_2d/norm_flux_2d**2)) #Propogate uncertainity and store the new variance after normalizing to the chosen line
		self.var1d = (self.flux/norm_flux_1d)**2 * ((self.var1d/self.flux**2) + (norm_var_1d/norm_flux_1d**2))
		self.pv  = self.pv/ norm_flux_2d #Noramlize all lines to the selected line in 2D PV space
		self.flux  = self.flux/ norm_flux_1d #Normalize all lines to the slected line in 1D velocity space
	def basic_flux(self, x_range, y_range):
		sum_along_x = nansum(self.pv[:, y_range[0]:y_range[1], x_range[0]:x_range[1]], axis=2) #Collapse along velocity space
		total_sum = nansum(sum_along_x, axis=1) #Collapse along slit space
		return(total_sum) #Return the integrated flux found for each line in the box defined by the user
	def inspection(self): #Interactively loop through and view each each line to construct a pared down line list for a target for later reading inded
		#ioff()#  Turn off interactive plotting mode
		self.view() #Load up DS9 and the 1D view of the PV diagrams
		save_text = [] #Set up array for ascii text to save new pared down line list
		for i in xrange(self.n_lines): #Loop through to read in each line
			clf() #Clear figure for looking at 1D
			self.goline(self.label[i]) #View this line
			#show() 
			pause(0.001)
			print 'Line = ', self.label[i] #Print info about line in command line
			print 'Wave = ', self.lab_wave[i]
			answer = raw_input('Include in list? (y/n) ') #Ask user if they want to include this line in the line list
			if answer == 'Y' or answer == 'y':
				save_text.append('%1.10f' % self.lab_wave[i] + '\t' + self.label[i])
		print 'DONE WITH LIST!'
		output_filename = 'line_lists/' + raw_input('Please give a filename for the line list: ')
		savetxt(output_filename, save_text, fmt="%s") #Output line list
		#ion() #Turn interactive plotting mode back on 0
	def calculate_moments(self, vrange=[-set_velocity_range, set_velocity_range], prange=[0,0], s2n_cut=0.0, s2n_smooth=0.): #Calculate (mathematical) moments of the flux in velocity and position space; explicitely calculates  moments 0, 1, 2 = flux, mean, variance
		pv = copy.deepcopy(self.pv)
		if s2n_cut > 0.:
			g = Gaussian2DKernel(stddev=s2n_smooth)
			for i in xrange(len(pv)):
				low_s2n_mask = convolve(pv[i], g) / sqrt(self.var2d[i]) < s2n_cut
				pv[i][low_s2n_mask] = nan
		if prange[0] == 0 and prange[1] == 0: #If user does not specify prange explicitely 
			prange = [0, self.slit_pixel_length] #Set to use the whole slit by default
		use_velocities = (self.velocity >= vrange[0]) & (self.velocity <= vrange[1]) #Find indicies within velocity range specified by the variable vrange and only at those pixels, masking everything outside that range out
		position = arange(self.slit_pixel_length) #Set up an array for position along the slit (in pixel space, not in arcseconds)
		velocity_flux =  nansum(pv[:,:,use_velocities] * self.velocity_res, axis=2) #Calculate moment 0 (the flux) along the velocity axis
		velocity_mean = nansum(pv[:,:,use_velocities] * self.velocity[use_velocities] * self.velocity_res, axis=2) / velocity_flux
		velocity_variance =nansum(pv[:,:,use_velocities] * (self.velocity[newaxis,newaxis,use_velocities] - velocity_mean[:,:,newaxis])**2 * self.velocity_res, axis=2) / velocity_flux
		position_flux = nansum(pv[:,prange[0]:prange[1],:], axis=1)
		position_mean = nansum(pv[:,prange[0]:prange[1],:] *  position[prange[0]:prange[1],newaxis], axis=1) / position_flux
		position_variance = nansum(pv[:,prange[0]:prange[1],:] *  (position[newaxis,prange[0]:prange[1],newaxis]-position_mean[:,newaxis,:])**2, axis=1) / position_flux
		self.velocity_flux = velocity_flux  #Store all the moments in the position_velocity object as these variables for later use
		self.velocity_mean = velocity_mean
		self.velocity_variance = velocity_variance
		self.position_flux = position_flux
		self.position_mean = position_mean
		self.position_variance = position_variance
	def create_moment_mask(self, sigma=1.0): #Create a mask  +/- 3 sigma around the position and velocity moments, used for plotting positions of lines in a simplified way
		velocity_moment_mask = zeros(shape(self.pv)) #Set up array that will hold masks
		position_moment_mask = zeros(shape(self.pv)) #Set up array that will hold masks
		combined_moment_mask = zeros(shape(self.pv)) #Set up array that will hold masks
		position = arange(self.slit_pixel_length)
		for i in xrange(len(self.pv)): #Loop through each line
			for j in xrange(len(self.velocity)): #Loop through each position along the slit
				three_sigma_range = (position < self.position_mean[i,j]+sigma*sqrt(self.position_variance[i,j])) & (position > self.position_mean[i,j]-sigma*sqrt(self.position_variance[i,j])) #Find +/- 3 sigma from mean
				position_moment_mask[i,three_sigma_range,j] = 1.0 #Apply mask
				combined_moment_mask[i,three_sigma_range,j] = 1.0 #Apply mask
			for k in xrange(len(position)): #Loop through each velocity resoultion element
				three_sigma_range = (self.velocity < self.velocity_mean[i,k]+sigma*sqrt(self.velocity_variance[i,k])) &  (self.velocity > self.velocity_mean[i,k]-sigma*sqrt(self.velocity_variance[i,k]))#Find +/- 3 sigma from mean
				velocity_moment_mask[i,k,three_sigma_range] = 1.0 #Apply mask
				combined_moment_mask[i,k,three_sigma_range] = 1.0 #Apply mask
		self.velocity_moment_mask = velocity_moment_mask #Store the moment masks
		self.position_moment_mask = position_moment_mask
		self.combined_moment_mask = combined_moment_mask
	# def get_moment(self, moment, line): #Specify desired moment and line and return the result
	# 	if not hasattr(self, 'moments'): #Check if moments have been calculated yet
	# 		print 'Moments not yet calculated.  Claculating now.'
	# 		self.calculate_moments() #Calculate moments, if not done already
	# 	i =  where(self.label == line)[0][0] #Search for line by label
	# 	return self.moments[moment, i, :] #Return moment 






#@jit #Compile JIT using numba
def fit_mask(mask_contours, data, variance, pixel_range=[-10,10]): #Find optimal position (in velocity space) for mask for extracting 
	smoothed_data = median_filter(data, size=[5,5])
	shift_pixels = arange(pixel_range[0], pixel_range[1]) #Set up array for rolling mask
	s2n = zeros(shape(shift_pixels)) #Set up array to store S/N of each shift
	for i in xrange(len(shift_pixels)):
		shifted_mask_contours = roll(mask_contours, shift_pixels[i], 1) #Shift the mask contours by a certain number of pixels
		shifted_mask = shifted_mask_contours == 1.0 #Create new maskf from shifted mask countours
		flux = nansum(smoothed_data[shifted_mask]) - nanmedian(smoothed_data[~shifted_mask])*size(smoothed_data[shifted_mask]) #Calculate flux from shifted mask, do simple background subtraction
		sigma =  sqrt( nansum(variance[shifted_mask]) ) #Calculate sigma from shifted_mask
		s2n[i] = flux/sigma #Store S/N of mask in this position
	if all(isnan(s2n)): #Check if everything in the s2n array is nan, if so this is a bad part of the spectrum
		return 0 #so return a zero and move along
	else: #Otherwise we got something decent so...
		return shift_pixels[s2n == flat_nanmax(s2n)][0] #Return pixel shift that maximizes the s2n

#@jit  #Compile JIT using numba
def fit_weights(weights, data, variance, pixel_range=[-10,10]): #Find optimal position for an optimal extraction 
	#median_smoothed_data = median_filter(data, [5,5])
	#median_smoothed_variance =  median_filter(variance, [5,5])
	shift_pixels = arange(pixel_range[0], pixel_range[1]) #Set up array for rolling weights
	s2n = zeros(shape(shift_pixels)) #Set up array to store S/N of each shift
	#max_weight = nanmax(weights) #Find maximum of weights 
	#background_weight = max_weight * background_threshold_scale #Set weight below which will be used as background, typicall 1000x less than the peak signal
	for i in xrange(len(shift_pixels)): #Loop through each position in velocity space to test the optimal extraction
		shifted_weights =  roll(weights, shift_pixels[i], 1) #Shift weights by some amount of km/s for searching for the optimal shift
		background = nanmedian(data[shifted_weights == 0.0]) #Calculate typical background per pixel
		flux = nansum((data-background)*shifted_weights) #Calcualte weighted flux
		sigma = sqrt( nansum(variance*shifted_weights**2) ) #Calculate weighted sigma
		#flux = nansum((median_smoothed_data-background)*shifted_weights) #Calcualte weighted flux
		#sigma = sqrt( nansum(median_smoothed_variance*shifted_weights**2) ) #Calculate weighted sigma
		s2n[i] = flux / sigma
	if all(isnan(s2n)): #Check if everything in the s2n array is nan, if so this is a bad part of the spectrum
		return 0 #so return a zero and move along
	else: #Otherwise we got something decent so...
		return shift_pixels[s2n == flat_nanmax(s2n)][0] #Return pixel shift that maximizes the s2n



class region: #Class for reading in a DS9 region file, and applying it to a position_velocity object
	def __init__(self, pv, name='flux', file='', background='', s2n_cut = -99.0, show_regions=True, s2n_mask = 0.0, line='', pixel_range=[-10,10],
			savepdf=True, optimal_extraction=False, weight_threshold=1e-3, systematic_uncertainity=0.0):
		path = save.path + name #Store the path to save files in so it can be passed around, eventually to H2 stuff
		use_background_region = False
		line_labels =  pv.label #Read out line labels
		line_wave = pv.lab_wave #Read out (lab) line wavelengths
		mask_shift =  zeros(len(line_wave)) #Array to store shift (in pixels) of mask for s2n mask fitting
		pv_data = pv.pv #Holder for flux datacube
		pv_variance = pv.var2d #holder for variance datacube
		#bad_data = pv_data < -10000.0  #Mask out bad pixels and cosmic rays that somehow made it through, commented out for now since it doesn't seem to help with anything
		#pv_data[bad_data] = nan
		#pv_variance[bad_data] = nan
		pv_shape = shape(pv_data[0,:,:]) #Read out shape of a 2D slice of the pv diagram cube
		n_lines = len(pv_data[:,0,0]) #Read out number of lines
		velocity_range = [flat_nanmin(pv.velocity), flat_nanmax(pv.velocity)]
		if file == '' and line == '': #If no region file is specified by the user, prompt user for the path to the region file
			file = raw_input('What is the name of the region file? ')
		if background == '': #If no background region file is specified by the user, ask if user wants to specify region, and if so ask for path
			answer = raw_input('Do you want to designate a specific region to measure the median background (y) or just use the whole postage stamp (n)? ')
			if answer == 'y':
				print 'Draw DS9 region around part(s) of line you want to measure the median background for and save it as a .reg file in the scratch directory.'
				background == raw_input('What is the name of the region file? ')
				use_background_region = True
			else:
				use_background_region = False
		if background == 'all': #If user specifies to use whole 
			use_background_region = False
		if optimal_extraction: #If user specifies optimal extraction, we will weight each pixel by the signal of a bright line
			line_for_weighting = line_labels == line #Find index of line to weight by
			signal = copy.deepcopy(pv_data[line_for_weighting,:,:][0])-nanmedian(pv_data[line_for_weighting,:,:][0])
			signal = median_filter(signal, size=[5,5]) #Median filter signal before calculating weights to get rid of noise spikes, cosmics rays, etc.
			signal[signal < nanmax(signal) * weight_threshold] = 0. #Zero out pxiels below the background threshold scale
			weights = signal**2.0 #Grab signal of line to weight by, this signal is what will be used for the optimal extraction
			#weights = abs(signal)
			weights = weights / nansum(weights) #Normalize weights
			#weights[weights < weight_threshold]
		elif s2n_mask == 0.0: #If user specifies to use a region
			on_region = pyregion.open(file)  #Open region file for reading flux
			on_mask = on_region.get_mask(shape = pv_shape) #Make mask around region file
		else: #If user specifies to mask with a specific spectral line's S/N
			line_for_masking = line_labels == line #Find index of line to weight by
			s2n = pv_data[line_for_masking,:,:][0] / sqrt(pv_variance[line_for_masking,:,:][0])
			if any(isfinite(s2n)):
				on_mask = s2n > s2n_mask #Set on mask to be where line is above some s2n threshold
				if nansum(on_mask)>1:
					off_mask = ~on_mask
					#stop()
					mask_contours = zeros(shape(s2n)) #Set up 2D array of 1s and 0s that store the mask, 0 = outside mask, 1 = inside mask
					mask_contours[on_mask] = 1.0
		if use_background_region: #If you want to use another region to designate the background, read it in here
			off_region = pyregion.open(background) #Read in background region file
			off_mask = off_region.get_mask(shape = pv_shape) #Set up mask
		#figure(figsize=(4.0,3.0), frameon=False) #Set up figure check size
		#if weight != '': #If user specifies a line to weight by
		#	g = Gaussian2DKernel(stddev=5) #Set up gaussian smoothing to get rid of any grainyness between pixels
		#	line_for_weighting = line_labels == weight #Find index of line to weight by
		#	weights =  convolve(pv_data[line_for_weighting,:,:][0] / pv_variance[line_for_weighting,:,:][0], g) [on_mask] #Weight by the S/N ratio of that line
		#	
		#else: #If user specifies no weighting shoiuld be used
		#	weights = ones(shape(pv_variance[0,:,:])) #Give everything inside the region equal weight
		line_flux = zeros(n_lines) #Set up array to store line fluxes
		line_s2n = zeros(n_lines) #Set up array to store line S/N, set = 0 if no variance is found
		line_sigma = zeros(n_lines) #Set up array to store 1 sigma uncertainity
		if s2n_mask > 0.0: #If user is using a s2n mask...
			rolled_masks = zeros(shape(pv_data)) #Create array for storing rolled masks for later plotting, to save time computing the roll
		elif optimal_extraction: #If user wants an optimal extraction 
			rolled_weights =  zeros(shape(pv_data)) #Create array for storing rolled weights for later plotting, to save time computing the roll
		for i in xrange(n_lines): #Loop through each line
			if s2n_mask > 0.0: #If user specifies a s2n mask
				#shift_mask_pixels = self.fit_mask(mask_contours, pv_data[i,:,:], pv_variance[i,:,:], pixel_range=pixel_range) #Try to find the best shift in velocity space to maximize S/N
				shift_mask_pixels = fit_mask(mask_contours, pv_data[i,:,:], pv_variance[i,:,:], pixel_range=pixel_range) #Try to find the best shift in velocity space to maximize S/N
				mask_shift[i] == shift_mask_pixels
				try:
					use_mask = roll(on_mask, shift_mask_pixels, 1) #Set mask to be shifted to maximize S/N
					mask_shift[i] = shift_mask_pixels #Store how many pixels the mask has been shifted for later readout
					rolled_masks[i,:,:] = use_mask #store rolled mask for later plotting
				except:
					stop()
			elif optimal_extraction: #If user wantes to use optimal extraction
				shift_weight_pixels = fit_weights(weights, pv_data[i,:,:], pv_variance[i,:,:], pixel_range=pixel_range)
				mask_shift[i] = shift_weight_pixels
				shifted_weights = roll(weights, shift_weight_pixels, 1) #Set mask to be shifted to maximize S/N
				rolled_weights[i,:,:] = shifted_weights  #store shifted weights for later plotting the contours of
			else:
				use_mask = on_mask
			if "use_mask" in locals(): #If mask is valid run the code, otherwise ignore code to skip errors
				on_data = pv_data[i,:,:][use_mask]  #Find data inside the region for grabbing the flux
				on_variance = pv_variance[i,:,:][use_mask]
				if use_background_region: #If a backgorund region is specified
					off_data = pv_data[i,:,:][~use_mask] #Find data in the background region for calculating the background
					background = nanmedian(off_data) * size(on_data) #Calculate backgorund from median of data in region and multiply by area of region used for summing flux
				else: #If no background region is specified by the user, use the whole field 
					background = nanmedian(pv_data[i,:,:]) * size(on_data) #Get background from median of all data in field and multiply by area of region used for summing flux
				line_flux[i] = nansum(on_data) - background #Calculate flux from sum of pixels in region minus the background (which is the median of some region or the whole field, multiplied by the area of the flux region)
				line_sigma[i] =  sqrt( nansum(on_variance) ) #Store 1 sigma uncertainity for line
				line_s2n[i] = line_flux[i] / line_sigma[i] #Calculate the S/N in the region of the line
				#print 'i = ', i
				#print 'nansum(on_data) = ', nansum(on_data)
				#print 'background = ', background
			elif optimal_extraction: #Okay if the user specifies to use optimal extraction now that we know how the weights have been shifted to maximize S/N
				### Horne 1986 optimal extraction method, tests show it doesn't work so well as the weighting scheme below so it's commented out, left here if I wever want to revivie it
				# p = sqrt(shifted_weights)
				# p = p - nanmedian(p)
				# p[p < 0.] = 0.
				# p = p / nansum(p)
				# v = pv_variance[i,:,:]
				# f = pv_data[i,:,:]
				# s = nanmedian(f[p == 0.0])
				# m = ones(pv_shape) 
				# #sigma_clip_bad_pix = (f - s - nansum(f-s)*p)**2 > 10.0**2 * v
				# #m[sigma_clip_bad_pix] = 0.
				# p_squared_divided_by_v = nansum(m * p**2 / v)
				# try:
				# 	line_flux[i] = nansum((m * p * (f-s)) / v) / p_squared_divided_by_v
				# 	line_sigma[i] = sqrt( nansum((m*p)) / p_squared_divided_by_v  )
				# except:
				# 	line_flux[i] = nan
				# 	line_sigma[i] = nan
				### Current version of the extraction, appears to work best
				background = nanmedian(pv_data[i,:,:][shifted_weights == 0.0])  #Find background from all pixels below the background thereshold
				weighted_data =  (pv_data[i,:,:]-background) * shifted_weights #Extract the weighted data, while subtracting the background from each pixel
				weighted_variance  = pv_variance[i,:,:] * shifted_weights**2 #And extract the weighted variance
				line_flux[i] = nansum(weighted_data)#Calculate flux sum of weighted pixels
				line_sigma[i] =  sqrt( nansum(weighted_variance) ) #Store 1 sigma uncertainity for line
				line_s2n[i] = line_flux[i] / line_sigma[i] #Calculate the S/N in the region of the line
		if savepdf:  #If user specifies to save a PDF of the PV diagram + flux results
			with PdfPages(save.path + name + '.pdf') as pdf: #Make a multipage pdf
				figure(figsize=[11.0,8.5])
				for i in xrange(n_lines): #Loop through each line
					#subplot(n_subfigs, n_subfigs, i+1)
					
					clf() #Clear plot field
					gs = grd.GridSpec(2, 1, wspace = 0.2, hspace=0.05, width_ratios=[1], height_ratios=[0.5,0.5]) #Set up a grid of stacked plots for putting the excitation diagrams on
					subplots_adjust(hspace=0.05, left=0.08,right=0.96,bottom=0.08,top=0.93) #Set all plots to have no space between them vertically
					#ax = subplot(211) #Turn on "ax", set first subplot
					fig = gcf()#Adjust aspect ratio
					fig.set_size_inches([11.0,8.5]) #Adjust aspect ratio
					if line_s2n[i] > s2n_cut: #If line is above the set S/N threshold given by s2n_cut, plot it
						ax=subplot(gs[0])
						frame = gca() #Turn off axis number labels
						frame.axes.get_xaxis().set_ticks([]) #Turn off axis number labels
						frame.axes.get_yaxis().set_ticks([]) #Turn off axis number labels
						#if not optimal_extraction: #if not optimal extraction just show the results
						imshow(pv_data[i,:,:]+1e7, cmap='gray', interpolation='Nearest', origin='lower', norm=LogNorm(), aspect='auto') #Save preview of line and region(s)
						suptitle('i = ' + str(i+1) + ',    '+ line_labels[i] +'  '+str(line_wave[i])+',   Flux = ' + '%.3e' % line_flux[i] + ',   $\sigma$ = ' + '%.3e' % line_sigma[i] + ',   S/N = ' + '%.1f' % line_s2n[i] ,fontsize=14)
						#ax[0].set_title('i = ' + str(i+1) + ',    '+ line_labels[i] +'  '+str(line_wave[i])+',   Flux = ' + '%.3e' % line_flux[i] + ',   S/N = ' + '%.1f' % line_s2n[i])
						#xlabel('Velocity [km s$^{-1}$]')
						#ylabel('Along slit')
						ylabel('Position', fontsize=12)
						#xlabel('Velocity [km s$^{-1}$]')
						if show_regions and s2n_mask == 0.0 and not optimal_extraction: #By default show the 
							on_patch_list, on_text_list = on_region.get_mpl_patches_texts() #Do some stuff
							for p in on_patch_list: #Display DS9 regions in matplotlib
								try:
									ax.add_patch(p)
								except:
									print 'Glitch plotting pyregion.  Weird.'
							for t in on_text_list:
								ax.add_artist(t)
							if use_background_region:
								off_patch_list, off_text_list = off_region.get_mpl_patches_texts() #Do some stuff
								for p in off_patch_list:
									ax.add_patch(p)
								for t in off_text_list:
									ax.add_artist(t)
						if s2n_mask > 0.: #Plot s2n mask if user sets it
							try:
								#contour(roll(mask_contours, line_shift[i], 1))
								contour(rolled_masks[i,:,:])
							except:
								stop()
						elif optimal_extraction: #Plot weight contours if user specifies using optimal extraction
							#try:
								contour(sqrt(rolled_weights[i,:,:]), linewidths=0.5) #Plot weight contours
								background_mask = rolled_weights[i,:,:] == 0.0 #Find pixels used for background
								find_background = ones(shape(rolled_weights[i,:,:])) #Set up array to store 1 where backgorund is and 0 where it is not
								find_background[background_mask] = 0.0 #Set background found to 1 for plotting below
								contour(find_background, colors='red', linewidths=0.25) #Plot the backgorund with a dotted line
								#stop()
							#except:
							#	stop
						#ax = subplot(212) #Turn on "ax", set first subplot
						ax = subplot(gs[1])
						pv.plot_1d_velocity(i, clear=False, fontsize=10) #Test plotting 1D spectrum below 2D spectrum
						pdf.savefig() #Add figure as a page in the pdf
			#figure(figsize=(11, 8.5), frameon=False) #Reset figure size
		if systematic_uncertainity > 0.: #If user specifies some fractional systematic uncertainity
			line_sigma = sqrt(line_sigma**2 + (line_flux*systematic_uncertainity)**2) #Combine the statistical uncertainity with the systematic uncertainity
			line_s2n = line_flux / line_sigma #And then recalculate the S/N based on the new value
		self.wave = line_wave #Save wavelength of lines
		self.label = line_labels #Save labels of lines
		self.flux = line_flux #Save line fluxes
		self.s2n = line_s2n #Save line S/N
		self.sigma = line_sigma #Save the 1 sigma limit
		if hasattr(pv, 'shift_v'): #Check if the pv object has a stored shift_v variable
			self.shift_v = pv.shift_v #Carry over velocity shifts from position_velocity (if they exist) into the region object for later tabulation
			self.shift_wave = pv.shift_wave #Carry over wavelength shifts  from position_velocity (if they exist) into the region object for later tabulation
		if s2n_mask: #If user uses masked equal weighted extraction save the following....
			self.mask_contours = mask_contours #store mask contours for later inspection or plotting if needed (for making advnaced 2D figures in papers)
			self.mask_shift = mask_shift #Store mask shift (in pixels) to later recall what the S/N maximization routine found
		elif optimal_extraction: #else if the user uses optimal extraction save the following
			self.weights = weights #Store weights used in extraction
			self.rolled_weights = rolled_weights #Store pixel shifts in weights used for extractionx
			self.mask_shift = mask_shift
		elif s2n_mask == 0.0 and not optimal_extraction: #If user specified region in DS0 is used
			self.on_region = on_region #Store the region data for later plotting if necessary
		self.path = path #Save path to 
	# def fit_mask(self, mask_contours, data, variance, pixel_range=[-10,10]): #Find optimal position (in velocity space) for mask for extracting 
	# 	smoothed_data = median_filter(data, size=[5,5])
	# 	shift_pixels = arange(pixel_range[0], pixel_range[1]) #Set up array for rolling mask
	# 	s2n = zeros(shape(shift_pixels)) #Set up array to store S/N of each shift
	# 	for i in xrange(len(shift_pixels)):
	# 		shifted_mask_contours = roll(mask_contours, shift_pixels[i], 1) #Shift the mask contours by a certain number of pixels
	# 		shifted_mask = shifted_mask_contours == 1.0 #Create new mask from shifted mask countours
	# 		flux = nansum(smoothed_data[shifted_mask]) - nanmedian(smoothed_data[~shifted_mask])*size(smoothed_data[shifted_mask]) #Calculate flux from shifted mask, do simple background subtraction
	# 		sigma =  sqrt( nansum(variance[shifted_mask]) ) #Calculate sigma from shifted_mask
	# 		s2n[i] = flux/sigma #Store S/N of mask in this position
	# 	if all(isnan(s2n)): #Check if everything in the s2n array is nan, if so this is a bad part of the spectrum
	# 		return 0 #so return a zero and move along
	# 	else: #Otherwise we got something decent so...
	# 		return shift_pixels[s2n == nanmax(s2n)][0] #Return pixel shift that maximizes the s2n
	def make_latex_table(self, output_filename, s2n_cut = 3.0, normalize_to='5-3 O(3)'): #Make latex table of line fluxes
   		lines = []
   		#lines.append(r"\begin{table}")  #Set up table header
   		lines.append(r"\begin{longtable}{rrlrr}")
   		lines.append(r"\caption{Line Fluxes}{} \label{tab:fluxes} \\")
   		#lines.append("\begin{scriptsize}")
   		#lines.append(r"\begin{tabular}{cccc}")
   		lines.append(r"\hline")
   		lines.append(r"$\lambda_{\mbox{\tiny vacuum}}$ & $\Delta\lambda$  & Line ID & $\log_{10} \left(F_i / F_{\mbox{\tiny "+normalize_to+r"}}\right)$ & S/N \\")
   		lines.append(r"\hline\hline")
   		lines.append(r"\endfirsthead")
   		lines.append(r"\hline")
   		lines.append(r"$\lambda_{\mbox{\tiny vacuum}}$ & $\Delta\lambda$  & Line ID & $\log_{10} \left(F_i / F_{\mbox{\tiny "+normalize_to+r"}}\right)$ & S/N \\")
   		lines.append(r"\hline\hline")
   		lines.append(r"\endhead")
   		lines.append(r"\hline")
   		lines.append(r"\endfoot")
   		lines.append(r"\hline")
   		lines.append(r"\endlastfoot")
   		flux_norm_to = self.flux[self.label == normalize_to]
   		for i in xrange(len(self.label)):
   			if self.s2n[i] > s2n_cut:
				lines.append(r"%1.6f" % self.wave[i] + " & " + "%1.6f" % self.shift_wave[i]  + " & " + self.label[i] + " & $" + "%1.2f" % log10(self.flux[i]/flux_norm_to) + 
					r"^{+%1.2f" % (-log10(self.flux[i]/flux_norm_to) + log10(self.flux[i]/flux_norm_to+self.sigma[i]/flux_norm_to)) 
				   +r"}_{%1.2f" % (-log10(self.flux[i]/flux_norm_to) + log10(self.flux[i]/flux_norm_to-self.sigma[i]/flux_norm_to)) +r"} $ & %1.1f" % self.s2n[i]  + r" \\") 
   		#lines.append(r"\hline\hline")
		#lines.append(r"\end{tabular}")
		lines.append(r"\end{longtable}")
		#lines.append(r"\end{table}")
		savetxt(output_filename, lines, fmt="%s") #Output table
	def normalize(self, normalize_to):  #Normalize all line fluxes by dividing by this number
		self.flux = self.flux / normalize_to
		self.sigma = self.sigma / normalize_to
	def save_table(self, output_filename, s2n_cut = 3.0): #Output simple text table of wavelength, line label, flux
		lines = []
		lines.append('#Label\tWave [um]\tFlux\t1 sigma uncertainity')
		for i in xrange(len(self.label)):
			lines.append(self.label[i] + "\t%1.5f" % self.wave[i] + "\t%1.3e" % self.flux[i] + "\t%1.3e" % self.sigma[i])
		savetxt(save.path + output_filename, lines, fmt="%s") #Output Table
	#~~~~~~~save line ratios~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	def save_ratios(self, to_line='', factor=1.0):
		if to_line == '': #If user does not specify line to take ratio relative to
			to_line = self.label[0]  #Take ratio relative to the first line in the 
		ratios = factor * self.flux / self.flux[self.label == to_line] #Take ratios, multiply by some factor if comparing to some other table (ie. compared to H-beta as found in Osterbrock & Ferland 2006)
		ratios_sigma = sqrt(ratios**2 *((self.sigma/self.flux)**2+(self.sigma[self.label == to_line]/self.flux[self.label == to_line])**2))
		fname = save.path + 'line_ratios_relative_to_' + to_line + '.dat' #Set up file path name to save
		printme = [] #Array that will hold file ouptut
		for i in xrange(len(self.label)): #Loop thorugh each line
			printme.append(self.label[i] + '/'+ to_line + '\t%1.5f' % ratios[i] + '\t%1.5f' % ratios_sigma[i]) #Save ratio to an array that will be outputted as the .dat file
		savetxt(fname, printme, fmt="%s") #Output Table, and we're done
	# def calculate_HI_extinction(self, s2n_cut=3.0, max_n=16): #Calculate extinction by comparing measured H I line fluxes to theory predicted by pyneb 
	# 	data_file = 'data/predicted_HI_fluxes.dat' #File storing predicted H I line fluxes
	# 	state_i, state_j = loadtxt(data_file, unpack=True, dtype='i', delimiter='\t', usecols=(0,1,)) #read in states
	# 	wave, flux  = loadtxt(data_file, unpack=True, dtype='f', delimiter='\t', usecols=(2,3,)) #Read in wavelength and predicted fluxes
	# 	#construct a list of all found H I lines
	# 	found_wavelengths = []
	# 	found_observed_fluxes = []
	# 	found_predicted_fluxes = []
	# 	found_labels = []
	# 	#loop through each possible H I line in our predicted flux list and see if they exist in this region object, if they do, append the founds lists
	# 	for i in xrange(len(state_is[]	# 		label = 'H I '+str(state_j[i])+'-'+str(state_i[i]) #construct a string to match the line label strings in the H I line list
	# 		find_line = where(self.label == label)[0] #Check if line exists in this region object
	# 		if len(find_line)==1 and state_i[i] < max_n and state_j[i] < max_n: #If it is found
	# 			find_line = find_line[0] #Get found line index out of array and into a simple integer
	# 			if self.s2n[find_line] > s2n_cut:
	# 				found_labels.append(label)
	# 				found_wavelengths.append(self.wave[find_line]) #Store waveneghts...
	# 				found_observed_fluxes.append(self.flux[find_line]) #observed fluxes...
	# 				found_predicted_fluxes.append(flux[i]) #and predicted fluxes for every H I line found
	# 	found_wavelengths = array(found_wavelengths) #Convert everything to numpy arrays for easy processing
	# 	found_observed_fluxes = array(found_observed_fluxes)
	# 	found_predicted_fluxes = array(found_predicted_fluxes)
	# 	#Now wee construct the extinction curve
	# 	A_lambda = array([ 0.482,  0.282,  0.175,  0.112,  0.058]) #(A_lambda / A_V) extinction curve from Rieke & Lebofsky (1985) Table 3
	# 	l = array([ 0.806,  1.22 ,  1.63 ,  2.19 ,  3.45 ]) #Wavelengths for extinction curve from Rieke & Lebofsky (1985)
	# 	extinction_curve = interp1d(l, A_lambda, kind='quadratic') #Create interpolation object for extinction curve from Rieke & Lebofsky (1985)
	# 	AVs = arange(0,20.0,0.1) #Create array of exctinations to test
	# 	n_AVs = len(AVs)
	# 	chisq = zeros(n_AVs)
	# 	br14_brgamma_chisq = zeros(n_AVS)
	# 	br14 = found_labels == 'H I 4-14'
	# 	brgamma = found_labels == 'H I 4-7'
	# 	for i in xrange(n_AVs):
	# 		reddened_predicted_flux = found_predicted_fluxes * 10**(-0.4*extinction_curve(found_wavelengths)*AVs[i])  #Apply artificial def  to predicted line fluxes
	# 		ratio = found_observed_fluxes / reddened_predicted_flux #Calculate ratio to observed to artificially reddened predicted line fluxes
	# 		median_ratio = nanmedian(ratio) #Find the median ratio
	# 		chisq[i] = nansum(log10(ratio/median_ratio)**2) #calculate the chisq'
	# 		chisq_br14_brgamma = nansum(log10((ratio[])/))
	# 	print 'Best A_V for all lines used (minus cuts) = ', AVs[chisq==nanmin(chisq)]
	# 	#return AVs[chisq==nanmin(chisq)] #return the AV that matches the minimum chisq
	def calculate_HI_extinction(self, plot_result=True): #Calculate extinction by comparing measured H I line fluxes to theory predicted by pyneb 
		data_file = 'data/predicted_HI_fluxes.dat' #File storing predicted H I line fluxes
		state_i, state_j = loadtxt(data_file, unpack=True, dtype='i', delimiter='\t', usecols=(0,1,)) #read in states
		wave, flux  = loadtxt(data_file, unpack=True, dtype='f', delimiter='\t', usecols=(2,3,)) #Read in wavelength and predicted fluxes
		#construct a list of all found H I lines
		predicted_br14_flux = (flux[(state_i==14) & (state_j==4)])[0]
		predicted_brgamma_flux =  (flux[(state_i==7) & (state_j==4)])[0]
		#Now wee construct the extinction curve
		A_lambda = array([ 0.482,  0.282,  0.175,  0.112,  0.058]) #(A_lambda / A_V) extinction curve from Rieke & Lebofsky (1985) Table 3
		l = array([ 0.806,  1.22 ,  1.63 ,  2.19 ,  3.45 ]) #Wavelengths for extinction curve from Rieke & Lebofsky (1985)
		extinction_curve = interp1d(l, A_lambda, kind='quadratic') #Create interpolation object for extinction curve from Rieke & Lebofsky (1985)
		AVs = arange(0,20.0,0.01) #Create array of exctinations to test
		n_AVs = len(AVs)
		chisq = zeros(n_AVs)
		observed_br14_brgamma_ratio = (self.flux[self.label=='H I 4-14'] / self.flux[self.label=='H I 4-7'])[0]
		br14_extinction_curve = extinction_curve(1.5884880)
		brgamma_extinction_curve = extinction_curve(2.166120)
		for i in xrange(n_AVs):
			reddened_predicted_br_14_flux =  predicted_br14_flux * 10**(-0.4*br14_extinction_curve*AVs[i])  #Apply artificial reddening to predicted line fluxes
			reddened_predcited_br_gamma_flux = predicted_brgamma_flux * 10**(-0.4*brgamma_extinction_curve*AVs[i])  #Apply artificial reddening to predicted line fluxes
			reddened_predicted_ratio = reddened_predicted_br_14_flux / reddened_predcited_br_gamma_flux #Calculate ratio to observed to artificially reddened predicted line fluxes
			chisq[i] = (observed_br14_brgamma_ratio-reddened_predicted_ratio)**2 #calculate the chisq'
		best_AV = AVs[chisq==nanmin(chisq)] 
		print 'AV = ', best_AV #print the AV that matches the minimum chisq
		if plot_result:
			found_wavelengths = []
			found_observed_fluxes = []
			found_predicted_fluxes = []
			found_labels = []
			#loop through each possible H I line in our predicted flux list and see if they exist in this region object, if they do, append the founds lists
			for i in xrange(len(state_i)):
				label = 'H I '+str(state_j[i])+'-'+str(state_i[i]) #construct a string to match the line label strings in the H I line list
				find_line = where(self.label == label)[0] #Check if line exists in this region object
				if len(find_line)==1 : #If it is found
					find_line = find_line[0] #Get found line index out of array and into a simple integer
					found_labels.append(label)
					found_wavelengths.append(self.wave[find_line]) #Store waveneghts...
					found_observed_fluxes.append(self.flux[find_line]) #observed fluxes...
					found_predicted_fluxes.append(flux[i]) #and predicted fluxes for every H I line found
			found_wavelengths = array(found_wavelengths) #Convert everything to numpy arrays for easy processing
			found_observed_fluxes = array(found_observed_fluxes)
			found_predicted_fluxes = array(found_predicted_fluxes)
			clf()
			plot(found_wavelengths, found_observed_fluxes / found_predicted_fluxes, 'o', label='Reddening Uncorrected')
			plot(found_wavelengths, found_observed_fluxes * 10**(0.4*extinction_curve(found_wavelengths)*best_AV) / found_predicted_fluxes, 'o', label='Reddening Corrected')
			xlabel('Wavelength')
			ylabel('Dereddend fluxes / predicted fluxes')
			suptitle('Br-14/Br-$\gamma$ A$_V$ = '+str(best_AV))




def combine_regions(region_A, region_B, name='combined_region'): #Definition to combine two regions by adding their fluxes and variances together
	combined_region = copy.deepcopy(region_A) #Start by created the combined region
	combined_region.flux += region_B.flux #Add fluxes together
	combined_region.sigma = sqrt(combined_region.sigma**2 + region_B.sigma**2) #Add uncertianity in quadrature
	combined_region.s2n = combined_region.flux / combined_region.sigma #Recalculate new S/N
	combined_region.path = save.path + name #Store the path to save files in so it can be passed around, eventually to H2 stuff
	return(combined_region) #Returned combined region


class extract: #Class for extracting fluxes in 1D from a position_velocity object
	def __init__(self, pv, name='flux_1d', file='', background=True, s2n_cut = -99.0, vrange=[0,0], use2d=False, show_extraction=True, systematic_uncertainity=0.0):
		path = save.path + name #Store the path to save files in so it can be passed around, eventually to H2 stuff
		line_labels =  pv.label #Read out line labels
		line_wave = pv.lab_wave #Read out (lab) line wavelengths
		if use2d: #By default use the 1D spectrum (for ABBA observations), but use 2D for extended objects
			flux = nansum(pv.pv, 1) #Collapse flux along slit
			var = nansum(pv.var2d, 1)#Collapse variance along slit
		else:
			flux = pv.flux #Holder for flux datacube
			var = pv.var1d #holder for variance datacube
		velocity = pv.velocity
		#bad_data = pv_data < -10000.0  #Mask out bad pixels and cosmic rays that somehow made it through
		#pv_data[bad_data] == nan
		#pv_variance[bad_data] == nan
		#pv_shape = shape(pv_data[0,:,:]) #Read out shape of a 2D slice of the pv diagram cube
		n_lines = pv.n_lines #Read out number of lines
		if range == [0,0]: #If user does not specify velocity range, ask for it from user
			low_range = float(raw_input('What is blue velocity limit? '))
			high_range = float(raw_input('What is red velocity limit? '))
		on_target = (velocity > vrange[0]) & (velocity < vrange[1]) #Find points inside user chosen velocity range
		off_target = ~on_target #Find points outside user chosen velocity range
		#figure(figsize=(4.0,3.0), frameon=False) #Set up figure check size
		figure(0)
		with PdfPages(save.path + name + '.pdf') as pdf: #Make a multipage pdf
			line_flux = zeros(n_lines) #Set up array to store line fluxes
			line_s2n = zeros(n_lines) #Set up array to store line S/N, set = 0 if no variance is found
			line_sigma = zeros(n_lines) #Set up array to store 1 sigma uncertainity
			for i in xrange(n_lines): #Loop through each line
				clf() #Clear plot field
				data = flux[i]
				variance = var[i]
				if background: #If user 
					background_level =  nanmedian(data[off_target]) #Calculate level (per pixel) of the background level
				else: #If no background region is specified by the user, use the whole field 
					background_level = 0.0 #Or if you don't want to subtract the background, just make the level per pixel = 0
				line_flux[i] = nansum(data[on_target]) - background_level * size(data[on_target]) #Calculate flux from sum of pixels in region minus the background (which is the median of some region or the whole field, multiplied by the area of the flux region)
				line_sigma[i] =  sqrt( nansum(variance[on_target]) ) #Store 1 sigma uncertainity for line
				line_s2n[i] = line_flux[i] / line_sigma[i] #Calculate the S/N in the region of the line
				if line_s2n[i] > s2n_cut: #If line is above the set S/N threshold given by s2n_cut, plot it
					suptitle('i = ' + str(i+1) + ',    '+ line_labels[i] +'  '+str(line_wave[i])+',   Flux = ' + '%.3e' % line_flux[i] + ',   $\sigma$ = ' + '%.3e' % line_sigma[i] + ',   S/N = ' + '%.1f' % line_s2n[i] ,fontsize=10)
					pv.plot_1d_velocity(i, clear=False, fontsize=16, show_zero=show_extraction) #Test plotting 1D spectrum below 2D spectrum
					if show_extraction: #By default, the extraction velocity limits and background level are shown.  If user sets show_extraction = False, these are not shown
						plot([vrange[0], vrange[0]], [-1e50,1e50], linestyle='--', color = 'blue')  #Plot blueshifted velocity limits
						plot([vrange[1], vrange[1]], [-1e50,1e50], linestyle='--', color = 'blue')  #Plot redshifted velocity limits
						plot([flat_nanmin(velocity), flat_nanmax(velocity)], [background_level/1e3, background_level/1e3], linestyle='--', color = 'blue') #Plot background level
					#print "background_level = ",background_level
					pdf.savefig() #Add figure as a page in the pdf
		#dfigure(figsize=(11, 8.5), frameon=False) #Reset figure size
		if systematic_uncertainity > 0.: #If user specifies some fractional systematic uncertainity
			line_sigma = sqrt(line_sigma**2 + (line_flux*systematic_uncertainity)**2) #Combine the statistical uncertainity with the systematic uncertainity
			line_s2n = line_flux / line_sigma
		self.velocity = velocity #Save velocity grid
		self.wave = line_wave #Save wavelength of lines
		self.label = line_labels #Save labels of lines
		self.flux = line_flux #Save line fluxes
		self.s2n = line_s2n #Save line S/N
		self.sigma = line_sigma #Save the 1 sigma limit
		self.path = path #Save path to 
	def normalize(self, normalize_to):  #Normalize all line fluxes by dividing by this number
		self.flux = self.flux / normalize_to
		self.sigma = self.sigma / normalize_to
	def save_table(self, output_filename, s2n_cut = 3.0): #Output simple text table of wavelength, line label, flux
		lines = []
		lines.append('#Label\tWave [um]\tFlux\t1 sigma uncertainity')
		for i in xrange(len(self.label)):
			lines.append(self.label[i] + "\t%1.5f" % self.wave[i] + "\t%1.3e" % self.flux[i] + "\t%1.3e" % self.sigma[i])
		savetxt(save.path + output_filename, lines, fmt="%s") #Output Table




#~~~~~~~~~~~~~~~~~~~~~~~~~Code for reading in analyzing spectral data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~j

#Convenience function for making a single spectrum object in 1D or 2D that combines both H & K bands while applying telluric correction and flux calibration
#The idea is that the user can call a single line and get a single spectrum ready to go
def getspec(date, waveno, frameno, stdno, oh=0, oh_scale=0.0, oh_flexure=0., B=0.0, V=0.0, y_scale=1.0, wave_smooth=0.0, y_power=1.0, y_sharpen=0.0,
		twodim=True, usestd=True, no_flux=False, make_1d=False, median_1d=False, tellurics=False, savechecks=True, mask_cosmics=False,
		telluric_power=1.0, telluric_spectrum=[], calibration=[], telluric_quality_cut=False, interpolate_slit=False, std_shift=0.0):
	if usestd:
		#Make 1D spectrum object for standard star
		H_std_obj = makespec(date, 'H', waveno, stdno) #Read in H-band
		K_std_obj = makespec(date, 'K', waveno, stdno) #Read in H-band
		std_obj = H_std_obj #Create master object=
		std_obj.orders = K_std_obj.orders + H_std_obj.orders #Combine orders
		std_obj.n_orders = K_std_obj.n_orders + H_std_obj.n_orders #Find new total number of orders
		#Made 1D spectrum for flattened standard star (used for telluric correction)
		H_stdflat_obj =  makespec(date, 'H', waveno, stdno, std=True) #Read in H-band
		K_stdflat_obj =  makespec(date, 'K', waveno, stdno, std=True) #Read in K-band
		stdflat_obj = H_stdflat_obj #Create master object
		stdflat_obj.orders = K_stdflat_obj.orders + H_stdflat_obj.orders #Combine orders
		stdflat_obj.n_orders = K_stdflat_obj.n_orders + H_stdflat_obj.n_orders #Find new total number of orders
	#Make 1D spectrum object
	H_sci1d_obj =  makespec(date, 'H', waveno, frameno) #Read in H-band
	K_sci1d_obj =  makespec(date, 'K', waveno, frameno) #Read in K-band
	sci1d_obj = H_sci1d_obj #Create master object
	sci1d_obj.orders = K_sci1d_obj.orders + H_sci1d_obj.orders #Combine orders
	sci1d_obj.n_orders = K_sci1d_obj.n_orders + H_sci1d_obj.n_orders #Find new total number of orders
	if twodim: #If user specifies also to make a 2D spectrum object
		#Make 2D spectrum object
		H_sci2d_obj =  makespec(date, 'H', waveno, frameno, twodim=True, mask_cosmics=mask_cosmics, interpolate_slit=interpolate_slit) #Read in H-band
		K_sci2d_obj =  makespec(date, 'K', waveno, frameno, twodim=True, mask_cosmics=mask_cosmics, interpolate_slit=interpolate_slit) #Read in K-band
		#if H_sci2d_obj.slit_pixel_length != K_sci2d_obj.slit_pixel_length:
		#print 'H slit length: ', H_sci2d_obj.slit_pixel_length
		#print 'K slit length: ', K_sci2d_obj.slit_pixel_length
		sci2d_obj = H_sci2d_obj #Create master object
		sci2d_obj.orders = K_sci2d_obj.orders + H_sci2d_obj.orders #Combine orders
		sci2d_obj.n_orders = K_sci2d_obj.n_orders + H_sci2d_obj.n_orders #Find new total number of orders
		if make_1d: #If user specifies they want to make a 1D spectrum, we will overwrite the spec1d
			for i in xrange(sci2d_obj.n_orders): #Loop through each order to....
				sci1d_obj.orders[i].flux = nansum(sci2d_obj.orders[i].flux, 0) #Collapse 2D spectrum into 1D
				sci1d_obj.orders[i].noise = sqrt(nansum(sci2d_obj.orders[i].noise**2, 0)) #Collapse 2D noise in 1D
		elif median_1d: #If user specifies they want to make a 1D spectrum by median collapsing, overwrite the old spec1d, for now we calculate uncertainity by summing variance
			for i in xrange(sci2d_obj.n_orders): #Loop through each order to....
				sci1d_obj.orders[i].flux = nanmedian(sci2d_obj.orders[i].flux, 0) #Collapse 2D spectrum into 1D
				sci1d_obj.orders[i].noise = sqrt(nansum(sci2d_obj.orders[i].noise**2, 0)) #Collapse 2D noise in 1D
	#Read in sky difference frame to correct for OH lines, with user interacting to set the scaling
	if oh != 0: #If user specifies a sky correction image number
		oh1d = getspec(date, waveno, oh, oh, usestd=False, median_1d=True, twodim=False) #Create 1D and 2D spectra objects for all orders combining both H and K bands (easy eh?)
		if (oh_scale == 0.0) or (oh_scale == [0.0,0.0]): #If scale is not specified by user find it automatically, along with flexure, independently for H and K bands
			oh1d.combine_orders() #Combine OH sky difference orders so we can examine the entire H and K bands
			sci_obj = copy.deepcopy(sci1d_obj) #Make copy of science 1D object so we don't accidently modify the original data
			sci_obj.combine_orders() #Combine the science data orders
			oh_flux = oh1d.combospec.flux #Grab OH sky difference 1D flux
			wave = oh1d.combospec.wave #Grab wavelength array
			oh_flux[isinf(oh_flux)] = nan #turn infininte oh values into nan to fix errors
			flux = sci_obj.combospec.flux #Grab science 1D flux
			g = Gaussian1DKernel(stddev=20) #Prepare to smooth the OH sky difference data to find where there are OH residuals and where there are none
			oh_smoothed = abs(convolve(oh_flux,g)) #Smooth OH sky difference frame
			oh_smoothed = oh_smoothed / nanmax(oh_smoothed) #Normalize to brightest OH residual
			#oh_mask = oh_smoothed > 0.05 #Find all the smoothed OH sky difference residuals above 1/20th the brightness of the smoothed brightest residual
			oh_mask = zeros(len(oh_smoothed)) #Set up oh mask as an array of numbers
			oh_mask[(oh_smoothed/nanmedian(oh_smoothed) > 10.0)] += 1 #Find all pixels above 3x the median of the smoothed residuals
			OH_lines = lines(OH_line_list, delta_v=0.0) #Load OH line list
			parsed_OH_lines = OH_lines.parse( flat_nanmin(wave), flat_nanmax(wave))
			width=0.00006
			for i in xrange(len(parsed_OH_lines.wave)): #Loop through each line
				oh_mask[abs(wave-parsed_OH_lines.wave[i]) < width] += 1 #Create mask of OH lines...
			g = Gaussian1DKernel(stddev=200) #Prepare to smooth the science data to zero out any continuum
			flux = flux - convolve(flux, g) #Subtract a crude fit to the continuum so that most of the OH residuals start around 0 flux
			flex = arange(-4.0, 4.0, 0.05) #Range of flexure shifts to test
			scales = arange(-4.0,4.0,0.01) #Range of OH scales to test
			in_h_band = (sci_obj.combospec.wave < 1.85) & (oh_mask == 2) #Find only pixels in the H band and near a bright OH residual
			in_k_band = (sci_obj.combospec.wave > 1.85) & (oh_mask == 2) #Find only pixels in the K band and near a bright OH residual
			h_store_chi_sq = zeros([len(scales), len(flex)]) #Array for storing chi-sq for h band
			k_store_chi_sq = zeros([len(scales), len(flex)]) #Array for storing chi-sq for k band
			for i in xrange(len(scales)): #Loop through each possible scaling of the OH residuals
				for k in xrange(len(flex)): #Look through each possible flexure value of the OH residuals
					tweaked_oh =  flexure(oh_flux*scales[i], flex[k]) #Apply the flexure and scaling to the OH sky difference residuals
					diff = (flux - tweaked_oh) #Subtract the tweaked OH sky difference residuals from the residuals in the science flux
					diff[~isfinite(diff)] = nan #Turn all values for diff that are infinite into nans so the nansum doesn't sum to infinity
					h_store_chi_sq[i,k] = nansum((diff[in_h_band])**2) #Calculate chisq for H band
					k_store_chi_sq[i,k] = nansum((diff[in_k_band])**2) #Calculate chisq for K band
			h_store_chi_sq[h_store_chi_sq==0.] = nan #Nan out zeros to correct error in finding the real minimum chisq
			k_store_chi_sq[k_store_chi_sq==0.] = nan
			best_h_band_indicies = where(h_store_chi_sq == flat_nanmin(h_store_chi_sq)) #Find best fit by findinging the minimum chisq in the H band
			best_k_band_indicies = where(k_store_chi_sq == flat_nanmin(k_store_chi_sq)) #Find best fit by findinging the minimum chisq in the K band
			oh_scale = [scales[best_h_band_indicies[0][0]], scales[best_k_band_indicies[0][0]]] #Save OH scaling best fit
			oh_flexure = [flex[best_h_band_indicies[1][0]], flex[best_k_band_indicies[1][0]]] #Save OH flexure best fit
			print 'No oh_scale specified by user, using automated chi-sq rediction routine.'
			print 'OH residual scaling found to be: ', oh_scale
			print 'OH residual flexure found to be: ', oh_flexure
		if oh_flexure != 0.: #If user specifies a flexure correction
			if size(oh_flexure) == 1: #If the correction is only one number, correct all orders
				for i in xrange(sci1d_obj.n_orders): #Loop through each order
					oh1d.orders[i].flux = flexure(oh1d.orders[i].flux, oh_flexure) #Apply flexure correction to 1D array
					#oh2d.orders[i].flux = flexure(oh1d.orders[i].flux, oh_flexure) #Apply flexure correction to 2D array
			else: #Else if correction has two numbers, the first number is the H band and hte second number is the K band
				for i in xrange(sci1d_obj.n_orders):#Loop through each order
					if  oh1d.orders[i].wave[0] < 1.85: #check which band we are in, index=0 is H band, 1 is K band
						flexure_index = 0
					else:
						flexure_index = 1
					oh1d.orders[i].flux = flexure(oh1d.orders[i].flux, oh_flexure[flexure_index]) #Apply flexure correction to 1D array
					#oh2d.orders[i].flux = flexure(oh1d.orders[i].flux, oh_flexure[flexure_index]) #Apply flexure correction to 2D array
		if size(oh_scale) == 1: #if user specifies only one oh scale for the h and k band, use the same scale in both bands, else use the scale for each band seperately if the user provides two oh_scales
			oh_scale = [oh_scale, oh_scale]
		if savechecks: #If user specifies to save checks as a pdf
			with PdfPages(save.path + 'check_OH_correction.pdf') as pdf: #Create PDF showing OH correction for user inspection
				clf()
				for i in xrange(sci1d_obj.n_orders): #Save whole spectrum at once
					plot(oh1d.orders[i].wave, oh1d.orders[i].flux, color='red', label='Differential Sky Subtraction')
					plot(sci1d_obj.orders[i].wave, sci1d_obj.orders[i].flux, ':', color='black', label='Uncorrected Science Data')
					if  oh1d.orders[i].wave[0] < 1.85: #check which band we are in, index=0 is H band, 1 is K band
						plot(oh1d.orders[i].wave, sci1d_obj.orders[i].flux -  oh1d.orders[i].flux*oh_scale[0], color='black', label='OH Corrected Science Data')
					else:
						plot(oh1d.orders[i].wave, sci1d_obj.orders[i].flux -  oh1d.orders[i].flux*oh_scale[1], color='black', label='OH Corrected Science Data')
					if i==0:
						legend(loc='upper right', fontsize=9) #Only plot legend for first set 
				xlabel('$\lambda$ [$\mu$m]')
				ylabel('Relative Flux')
				title('Check whole spectrum')
				tight_layout()
				pdf.savefig()
				for i in xrange(sci1d_obj.n_orders): #Then save each order for closer inspection
					clf()
					#()
					plot(oh1d.orders[i].wave, oh1d.orders[i].flux, color='red', label='Differential Sky Subtraction')
					plot(sci1d_obj.orders[i].wave, sci1d_obj.orders[i].flux, ':', color='black', label='Uncorrected Science Data')
					if  oh1d.orders[i].wave[0] < 1.85: #check which band we are in, index=0 is H band, 1 is K band
						plot(oh1d.orders[i].wave, sci1d_obj.orders[i].flux -  oh1d.orders[i].flux*oh_scale[0], color='black', label='OH Corrected Science Data')
					else:
						plot(oh1d.orders[i].wave, sci1d_obj.orders[i].flux -  oh1d.orders[i].flux*oh_scale[1], color='black', label='OH Corrected Science Data')
					xlabel('$\lambda$ [$\mu$m]')
					ylabel('Relative Flux')
					legend(loc='upper right',fontsize=9)
					title('Check individual orders')
					tight_layout()
					pdf.savefig()
		for i in xrange(sci1d_obj.n_orders):
			if  oh1d.orders[i].wave[0] < 1.85: #check which band we are in, index=0 is H band, 1 is K band
				use_oh_scale = oh_scale[0]
			else:
				use_oh_scale = oh_scale[1]
			sci1d_obj.orders[i].flux -= oh1d.orders[i].flux * use_oh_scale 
			if twodim: #If user specifies a two dimensional object
				sci2d_obj.orders[i].flux -= oh1d.orders[i].flux * use_oh_scale / float(sci2d_obj.slit_pixel_length)
			# if twodim: #If user specifies a two dimensional object
			# 	#sci2d_obj.orders[i].flux = sci2d_obj.orders[i].flux - tile(nanmedian(oh2d.orders[i].flux, 0), [slit_length,1]) * oh_scale
			# 	sci2d_obj.orders[i].flux -= nanmedian(oh2d.orders[i].flux, 0) * use_oh_scale
	#Apply telluric correction & relative flux calibration
	if tellurics: #If user specifies "tellurics", return only flattened standard star spectrum
		return stdflat_obj
	elif usestd: #If user wants to use standard star (True by default)
		spec1d = telluric_and_flux_calib(sci1d_obj, std_obj, stdflat_obj, B=B, V=V, no_flux=no_flux, y_scale=y_scale, y_power=y_power, y_sharpen=y_sharpen, wave_smooth=wave_smooth, savechecks=savechecks,
			telluric_power=telluric_power, telluric_spectrum=telluric_spectrum, calibration=calibration, quality_cut=telluric_quality_cut) #For 1D spectrum
		if twodim: #If user specifies this object has a 2D spectrum
			spec2d = telluric_and_flux_calib(sci2d_obj, std_obj, stdflat_obj,  B=B, V=V, no_flux=no_flux, y_scale=y_scale, y_power=y_power, y_sharpen=y_sharpen, wave_smooth=wave_smooth, savechecks=savechecks, 
				telluric_power=telluric_power, telluric_spectrum=telluric_spectrum, calibration=calibration, quality_cut=telluric_quality_cut) #Run for 2D spectrum
		#Return either 1D and 2D spectra, or just 1D spectrum if no 2D spectrum exists
		if twodim:
			return spec1d, spec2d #Return both 1D and 2D spectra objects
		else:
			return spec1d #Only return 1D spectra object
	else: #If user does not want to use standard star
		if twodim:
			return sci1d_obj, sci2d_obj #Return both 1D and 2D spectra objects
		else:
			return sci1d_obj #Only return 1D spectra object


		



#Wrapper for easily creating a 1D or 2D comprehensive spectrum object of any type, allowing user to import an entire specturm object in one line
def makespec(date, band, waveno, frameno, std=False, twodim=False, mask_cosmics=False, interpolate_slit=False):
	#spec_data = fits_file(date, frameno, band, std=std, twodim=twodim, s2n=s2n) #Read in data from spectrum
	spec_data = fits_file(date, frameno, band, std=std, twodim=twodim)
	try: #Try reading in new wavelength data from A0V
		wave_data = fits_file(date, waveno, band, wave=True) #If 1D, read in data from wavelength solution
	except: #If it does not exist, try reading in wavelength data the old way (from the calib directory)
		try: #Try reading in fits file
			wave_data = fits_file(date, waveno, band, wave_old=True) #If 1D, read in data from wavelength solution
		except: #If no fits file is found, try reading in json file instead
			filename = calib_path+str(date)+'/SDC'+band+'_'+str(date)+'_'+'%.4d' % int(frameno) +'.wvlsol_v0.json' #Set json file name
			with open(filename) as data_file:  #Read in Json file
			    data = json.load(data_file)
			wave_data = data['wvl_sol'] #Splice out the wavelength solution
	if twodim: #If spectrum is 2D but no variance data to be read in
		var_data = fits_file(date, frameno, band, var2d=True) #Grab data for 2D variance cube
		spec_obj = spec2d(wave_data, spec_data, fits_var=var_data, mask_cosmics=mask_cosmics, interpolate_slit=interpolate_slit) #Create 2D spectrum object, with variance data inputted to get S/N
	else: #If spectrum is 1D
		var_data = fits_file(date, frameno, band, var1d=True) 
		spec_obj = spec1d(wave_data, spec_data, var_data) #Create 1D spectrum object
	return(spec_obj) #Return the fresh spectrum object!
	
	

#Class stores information about a fits file that has been reduced by the PLP
class fits_file:
	def __init__(self, date, frameno, band, std=False, wave=False, twodim=False, s2n=False, var1d=False, var2d=False, wave_old=False):
		self.date = '%.4d' % int(date) #Store date of observation
		self.frameno =  '%.4d' % int(frameno) #Store first frame number of observation
		self.band = band #Store band name 'H' or 'K'
		self.std = std #Store if file is a standard star
		self.wave = wave #Store if file is a wavelength solution
		self.wave_old = wave_old #Store if file is the old way wavelength solutions are done
		self.s2n = s2n #Store if file is the S/N spectrum
		self.twodim = twodim #Store if file is of a 2D spectrum instead of a 1D spectrum
		self.var1d = var1d
		self.var2d = var2d #Store if file is a 2D variance map (like twodim but with variance instead of signal)
		self.path = self.filepath() #Determine path and filename for fits file
		fits_container = fits.open(self.path) #Open fits file and put data into memory
		self.data = fits_container[0].data.byteswap().newbyteorder() #Grab data from fits file
		self.n_orders = len(fits_container[0].data[:,0]) #cound number of orders in fits file
		fits_container.close() #Close fits file data
		#self.data = fits.open(self.path) #Open fits file and put data into memory
		#print self.path
	def filepath(self): #Given input variables, determine the path to the target fits file
		prefix =  'SDC' + self.band + '_' + self.date + '_' + self.frameno #Set beginning (prefix) of filename
		if self.std: #If file is for a standard star and you want the flattened spectrum
			postfix = '.spec_flattened.fits'
			master_path = data_path
		elif self.wave: #If file is the 1D wavelength calibration
			#prefix = 'SKY_' + prefix  #Old version reads in arclamp or sky wavelength calibraiton
			#postfix = '.wvlsol_v1.fits'
			#master_path = calib_path
			postfix = '.wave.fits' #New version reads in A0V telluric wavelength solution
			master_path = data_path
		elif self.wave_old: #This is the old way wavelength soultions were read in, use it if the new way doesn't work
			prefix = 'SKY_' + prefix  #Old version reads in arclamp or sky wavelength calibraiton
			postfix = '.wvlsol_v1.fits'
			master_path = calib_path
		elif self.twodim:  #If file is the 2D spectrum
			postfix = '.spec2d.fits'
			master_path = data_path
		elif self.var1d: #If file is 1D variance
			postfix = '.variance.fits'
			master_path = data_path
		elif self.var2d: #If file is 2D variance map
			postfix = '.var2d.fits'
			master_path = data_path
		elif self.s2n: #if the file is the 1D S/N spectrum
			postfix = '.sn.fits'
			master_path = data_path
		else: #If you just want to read in a normal 1D spectrum (including the unnormalized standard star)
			postfix = '.spec.fits'
			master_path = data_path
		return master_path + self.date +'/' + prefix + postfix #Return full path for fits file 
	def get(self):  #Get fits file data with an easy to call definition
		if self.wave:  #If wavelength file, the wavelenghts are stored in nanometers, so we must convert them to um
			return self.data / 1e3
		else: #Else just return what was storted in the FITS file without modifying the data
			return self.data

#Class to store and analyze a 1D spectrumc
class spec1d:
	def __init__(self, fits_wave, fits_spec, fits_var):
		try: #First try to see if the wavelength data is from a fits file
			wavedata = fits_wave.get() #Grab fits data for wavelength out of object, first try as if it were a fits object
		except: #If it is not a fits object, say something read out of a json file..
			wavedata = array(fits_wave) #Just copy over the data and get on with it
		specdata = fits_spec.get() #Grab fits data for flux out of object
		vardata = fits_var.get() #Grab fits data for variance
		orders = [] #Set up empty list for storing each orders
		n_orders = fits_spec.n_orders
		#n_orders = len(specdata[0].data[:,0]) #Count number of orders in spectrum
		#wavedata = wavedata[0].data.byteswap().newbyteorder() #Read out wavelength and flux data from fits files into simpler variables
		#fluxdata = specdata[0].data.byteswap().newbyteorder() #Read out wavelength and flux data from fits files into simpler variables
		#noisedata = sqrt( vardata[0].data.byteswap().newbyteorder() ) #Read out noise from fits file into a simpler variable by taking the square root of the variance
		noisedata = sqrt(vardata)  #Read out noise from fits file into a simpler variable by taking the square root of the variance
		for i in xrange(n_orders): #Loop through to process each order seperately
			orders.append( spectrum(wavedata[i,:], specdata[i,:], noise=noisedata[i,:])  ) #Append order to order list
		self.n_orders = n_orders
		self.orders = orders
	def subtract_continuum(self, show = False, size=0, sizes=[1000,500]): #Subtract continuum and background with an iterative running median
		if size != 0:
			sizes = [size]
		orders = self.orders
		for order in orders: #Apply continuum subtraction to each order seperately
				flux = copy.deepcopy(order.flux)
				whole_order_trace = nanmedian(flux)
				if whole_order_trace == nan: whole_order_trace = 0.
				flux = flux - whole_order_trace #Do an intiial removal of the flux
				nx = len(flux) 
				for size in sizes:
					x_left = arange(nx) - size #Create array to store left side of running median
					x_left[x_left < 0] = 0 #Set pixels beyond edge of order to be nonexistant
					x_right = arange(nx) + size #Create array to store right side of running median
					x_right[x_right > nx] = nx - 1 #Set pixels beyond right edge of order to be nonexistant
					x_size = x_right - x_left #Calculate number of pixels in the x (wavelength) direction			
					unmodified_flux = copy.deepcopy(flux)	
					for i in xrange(nx):
							trace = nanmedian(unmodified_flux[x_left[i]:x_right[i]])
							if trace == nan: trace = 0.
							flux[i] -= trace
				order.flux = flux	
	def old_subtract_continuum(self, show = False, size = half_block, lines=[], vrange=[-10.0,10.0], use_poly=False): #Subtract continuum using robust running median
		if show: #If you want to watch the continuum subtraction
			clf() #Clear interactive plot
			first_order = True #Keep track of where we are in the so we only create the legend on the first order
		for order in self.orders: #Apply continuum subtraction to each order seperately
			old_order = copy.deepcopy(order) #Make copy of flux array so the original is not modified
			if lines != []: #If user supplies a line list
				old_order = mask_lines(old_order, lines, vrange=vrange, ndim=1) #Mask out lines with nan with some velocity range, before applying continuum subtraction
			wave = order.wave #Read n wavelength array
			if use_poly:
				p_init = models.Polynomial1D(degree=4)
				fit_p = fitting.SimplexLSQFitter()
				nx = len(old_order.flux)
			 	x = arange(nx)
				p =  fit_p(p_init, x, old_order.flux)
				subtracted_flux = order.flux - p(x)
			else:
				median_result_1d = robust_median_filter(old_order.flux, size = size) #Take a robust running median along the trace, this is the found continuum
				subtracted_flux = order.flux - median_result_1d #Apply continuum subtraction
			if show: #If you want to watch the continuum subtraction
				if first_order:  #If on the first order, make the legend along with plotting the order
					plot(wave, subtracted_flux, label='Science Target - Continuum Subtracted', color='black')
					plot(wave, old_order.flux, label='Science Target - Continuum Not Subtracted', color='blue')
					plot(wave, median_result_1d, label='Continuum Subtraction', color='green')
					first_order = False #Now that we are done, just plot the ldata for all the other orders without making a long legend
				else: #Else just plot the order
					plot(wave, subtracted_flux, color='black')
					plot(wave, old_order.flux, color='blue')
					plot(wave, median_result_1d, color='green')
			order.flux = subtracted_flux #Replace this order's flux array with one that has been continuum subtracted
		if show: #If you want to watch the continuum subtraction
			legend() #Show the legend in the plot

	def normalize_continuum(self, show = False, size = half_block, lines=[], vrange=[-10.0,10.0], use_poly=False): #Normalize spectrum to continuum using robust running median
		for order in self.orders: #Apply continuum subtraction to each order seperately
			old_order = copy.deepcopy(order) #Make copy of flux array so the original is not modified
			if lines != []: #If user supplies a line list
				old_order = mask_lines(old_order, lines, vrange=vrange, ndim=1) #Mask out lines with nan with some velocity range, before applying continuum subtraction
			wave = order.wave #Read n wavelength array
			median_result_1d = robust_median_filter(old_order.flux, size = size) #Take a robust running median along the trace, this is the found continuum
			normalized_flux = order.flux / median_result_1d #Normalize continuum 
			order.flux = normalized_flux #Replace this order's flux array with one that has been continuum normalized
	def combine_orders(self, wave_pivot = default_wave_pivot): #Sitch orders together into one long spectrum
		combospec = copy.deepcopy(self.orders[0]) #Create a spectrum object to append wavelength and flux to
		order_length =  len(combospec.flux)
		blank = zeros([order_length*self.n_orders])#Create blanks to store new giant spectrum
		combospec.flux = copy.deepcopy(blank) #apply blanks to everything
		combospec.wave = copy.deepcopy(blank)
 		combospec.noise = copy.deepcopy(blank)
 		#combospec.s2n = copy.deepcopy(blank)
		for i in range(self.n_orders-1, -1, -1): #Loop through each order to stitch one and the following one together
			if i == self.n_orders-1: #If first order, simply throw it in
				xl = 0
				xr = order_length
				goodpix_next_order =  self.orders[i].wave > 0.
			else: #Else find the wave pivots
				[low_wave_limit, high_wave_limit]  = [flat_nanmin(self.orders[i].wave), combospec.wave[xr-1]] #Find the wavelength of the edges of the already stitched orders and the order currently being stitched to the rest 
				wave_cut = low_wave_limit + wave_pivot*(high_wave_limit-low_wave_limit) #Find wavelength between stitched orders and order to stitch to be the cut where they are combined, with pivot set by global var wave_pivot
				goodpix_next_order = self.orders[i].wave > wave_cut #Find pixels to the right of the where the order will be cut and stitched to the rest
				if combospec.wave[xr-1] > wave_cut:
					xl = where(combospec.wave > wave_cut)[0][0]-1 #Set left pixel to previous right pixel
				else:
					xl = xr-1
				xr = xl + len(self.orders[i].wave[goodpix_next_order])
			combospec.wave[xl:xr] = self.orders[i].wave[goodpix_next_order] #Stitch wavelength arrays together
			combospec.flux[xl:xr] = self.orders[i].flux[goodpix_next_order]  #Stitch flux arrays together
			combospec.noise[xl:xr] = self.orders[i].noise[goodpix_next_order] #Stitch noise arrays together
			#combospec.s2n[xl:xr] = self.orders[i].s2n[goodpix_next_order]  #Stitch S/N arrays together
		combospec.wave = combospec.wave[0:xr] #Get rid of extra pixels at end of arrays
		combospec.flux = combospec.flux[0:xr]
		combospec.noise = combospec.noise[0:xr]
		#combospec.s2n = combospec.s2n[0:xr]
		self.combospec = combospec #save the orders all stitched together
	#Simple function for plotting a 1D spectrum orders
	def plot(self, combospec=False):
		#clf()
		if combospec: #If user specifies, plot the combined spectrum (stitched together orders)
			plot(self.combospec.wave, self.combospec.flux)
		else: #or else just plot each order seperately (each a different color)
			for order in self.orders: #Plot each order
				plot(order.wave, order.flux)
		xlabel('Wavelength [$\mu$m]')
		ylabel('Relative Flux')
		show()
		#draw()
	#Plot spectrum with lines from line list overplotted
	def plotlines(self, linelist, threshold=0.0, model='', rows=5, ymax=0.0, fontsize=9.5, relative=False):
		if not hasattr(self, 'combospec'): #Check if a combined spectrum exists
			print 'No spectrum of combined orders found.  Createing combined spectrum.'
			self.combine_orders() #Combine spectrum before plotting, if not done already
		#clf() #Clear plot
		min_wave  = flat_nanmin(self.combospec.wave) #Find maximum wavelength
		max_wave  = flat_nanmax(self.combospec.wave) #Find minimum wavelength
		if ymax == 0.0: #If use does not set maximum y, do it automatically
			max_flux = nanmax(self.combospec.flux, axis=0)
		else: #else set it to what the user wants
			max_flux  = ymax / 1.4 
		if relative: #If relative set = true, make scale relative to whatever ymax was set to
			self.combospec.flux = self.combospec.flux / ymax
			ymax = 1.0
			max_flux  = ymax / 1.4 
		total_wave_coverage = max_wave - min_wave #Calculate total wavelength coverage
		if (model != '') and (model != 'none'): #Load model for comparison if needed
			model_wave, model_flux = loadtxt(model, unpack=True) #Read in text file of model with format of two columns with wave <tab> flux
			model_max_flux = nanmax(model_flux[logical_and(model_wave > min_wave, model_wave < max_wave)], axis=0) #find tallest line in model
			normalize_model = max_flux / model_max_flux #normalize tallest line in model to the tallest line in IGRINS data
			model_flux = normalize_model * model_flux #Apply normalization to model spectrum to match IGRINS spectrum
		#fig = figure(figsize=(15,11)) #Set proportions
		for j in xrange(rows): #Loop breaks spectrum figure into multiple rows
			wave_range = [min_wave + total_wave_coverage*(float(j)/float(rows)), #Calculate wavelength range for a single row
						min_wave + total_wave_coverage*(float(j+1)/float(rows))]
			subplot(rows,1,j+1) #split into multiple plots
			sci_in_range = logical_and(self.combospec.wave > wave_range[0], self.combospec.wave < wave_range[1]) #Find portion of spectrum in single row
			sub_linelist = linelist.parse(wave_range[0], wave_range[1]) #Find lines in single row
			wave_to_interp = append(insert(self.combospec.wave, 1.0, 0.0), 3.0) #Interpolate IGRINS spectrum to allow line labels to be placed in correct position in figure
			flux_to_interp = append(insert(self.combospec.flux, 0, 0.0), 0.0)
			sci_flux_interp = interp1d(wave_to_interp, flux_to_interp) #Get interpolation object of science spec.
			sub_linelist.flux = sci_flux_interp(sub_linelist.wave) #Get height of spectrum for each individual line
			for i in xrange(len(sub_linelist.wave)):#Output label for each emission lin
				other_lines = abs(sub_linelist.wave - sub_linelist.wave[i]) < 0.00001 #Window (in microns) to check for regions of higher flux nearby so only the brightest lines (in this given range) are labeled.
				#if sub_linelist.flux[i] > max_flux*threshold and nanmax(sub_linelist.flux[other_lines], axis=0) == sub_linelist.flux[i]: #if line is the highest of all surrounding lines within some window
					#if sub_linelist.label[i] == '{OH}': #If OH lines appear in line list.....
						#mask_these_pixels = abs(self.combospec.wave-sub_linelist.wave[i]) < 0.00006 #Create mask of OH lines...
						#self.combospec.flux[mask_these_pixels] = nan #Turn all pixels with OH lines into numpy nans so the OH lines don't get plotted
						#plot([linelist_wave[i], linelist_wave[i]], [linelist_flux[i]+max_flux*0.025, max_flux*0.92], ':', color='gray')
						#text(linelist_wave[i], linelist_flux[i]+max_flux*0.02, '$\oplus$', rotation=90, fontsize=9, verticalalignment='bottom', horizontalalignment='center', color='black') 
					#else:   #If no OH lines found, plot lines on figure
						#plot([sub_linelist.wave[i], sub_linelist.wave[i]], [sub_linelist.flux[i], sub_linelist.flux[i] + max_flux*0.065], ':', color='black') #Plot location of line as a dotted line a little bit above the spectrum
						#text(sub_linelist.wave[i], sub_linelist.flux[i] +  max_flux*0.073, sub_linelist.label[i], rotation=90, fontsize=fontsize, verticalalignment='bottom', horizontalalignment='center', color='black')  #Label line with text
			plot([sub_linelist.wave[i], sub_linelist.wave[i]], [sub_linelist.flux[i], sub_linelist.flux[i] + max_flux*0.065], ':', color='black') #Plot location of line as a dotted line a little bit above the spectrum
			text(sub_linelist.wave[i], sub_linelist.flux[i] +  max_flux*0.073, sub_linelist.label[i], rotation=90, fontsize=fontsize, verticalalignment='bottom', horizontalalignment='center', color='black')  #Label line with text
			plot(self.combospec.wave[sci_in_range], self.combospec.flux[sci_in_range], color='black') #Plot actual spectrum
			if (model != '') and (model != 'none'): #Load model for comparison if needed
				model_in_range = logical_and(model_wave > wave_range[0], model_wave < wave_range[1]) #Find portion of model spectrum in a given row 
				plot(model_wave[model_in_range], model_flux[model_in_range], color='red') #Plot model spectrum
			ylim([-0.05*max_flux, 1.4*max_flux]) #Set y axis range to show spectrum but also allow user to vew lines that are labeled
			xlim=(wave_range) #set x axis range
			ylabel('Relative Flux') #Set y axis label
			if j == rows-1: #only put x label on final plot
				xlabel('Wavelength [$\mu$m]') #Set x-axis label
			minorticks_on() #Show minor tick marks
			gca().set_autoscale_on(False) #Turn off autoscaling
		show() #Show spectrum
	def mask_OH(self, width=0.00006, input_linelist=OH_line_list): #Mask OH lines, use only after processing and combinging spectrum to make a cleaner 1D spectrum
		OH_lines = lines(input_linelist, delta_v=0.0) #Load OH line list
		parsed_OH_lines = OH_lines.parse( flat_nanmin(self.combospec.wave), flat_nanmax(self.combospec.wave))
		for i in xrange(len(parsed_OH_lines.wave)): #Loop through each line
			mask_these_pixels = abs(self.combospec.wave-parsed_OH_lines.wave[i]) < width #Create mask of OH lines...
			self.combospec.flux[mask_these_pixels] = nan #Turn all pixels with OH lines into numpy nans so the OH lines don't get plotted
	def savespec(self, name='1d_spectrum.dat'): #Save 1D spectrum, set 'name' to be the filename yo uwant
		if not hasattr(self, 'combospec'): #Check if a combined spectrum exists
			print 'No spectrum of combined orders found.  Createing combined spectrum.'
			self.combine_orders() #Combine spectrum before plotting, if not done already
		savetxt(save.path + name, transpose([self.combospec.wave, self.combospec.flux, self.combospec.noise])) #Save 1D spectrum as simple .dat file with wavelength, flux, and noise in seperate columns
	def fitgauss(self,line_list, v_range=[-30.0,30.0]): #Fit 1D gaussians to the 1D spectra and plot results
		self.fwhm = zeros(len(line_list.lab_wave)) #Store FWHM of all found lines
		fit_g = fitting.LevMarLSQFitter() #Initialize minimization algorithim for fitting gaussian
		all_fwhm = array([])
		all_wave = array([])
		all_x_pixels = array([])
		order_count = 0 
		interp_velocity_grid = arange(v_range[0], v_range[1], 0.1) #Velocity grid to interpolate line profiles onto
		with PdfPages(save.path + 'check_line_widths.pdf') as pdf:
			for order in self.orders:
				parsed_line_list = line_list.parse(flat_nanmin(order.wave), flat_nanmax(order.wave))
				finite_pixels = isfinite(order.flux) #store which pixels are finite
				n_lines = len(parsed_line_list.label) #Number of spectral lines
				fwhm = zeros(n_lines) #Create array to store FWHM of gaussian fits for each line
				x_pixels = zeros(n_lines) #Create array to store which x pixel the line is centered on
				for i in xrange(n_lines): #Loop through each individual line
					line_wave = parsed_line_list.wave[i]
					x_pixels[i] = where(order.wave >= line_wave)[0][0]
					all_velocity = c * ( (order.wave - line_wave) /  line_wave )
					goodpix = finite_pixels & (all_velocity > v_range[0]) & (all_velocity < v_range[1])
					flux = order.flux[goodpix]
					velocity =  all_velocity[goodpix]
					if len(flux) > 2:
						g_init = models.Gaussian1D(amplitude=max(flux), mean=0.0, stddev=8.0) #Initialize gaussian model for this specific line, centered at 0 km/s with a first guess at the dispersion to be the spectral resolution
						g = fit_g(g_init, velocity, flux) #Fit gaussian to line
						g_mean = g.mean.value #Grab mean of gaussian fit
						g_stddev = g.stddev.value
						g_fwhm = g_stddev * 2.355
						self.fwhm[i] = g_fwhm
						g_flux = g(velocity) #Grab 
						g_residuals = flux - g_flux
						#check_pixels = abs(velocity - g_mean) <= quality_check_window
						#if g_fwhm > 0.0 and g_fwhm < 50.0 and sum(abs(g_residuals[check_pixels]))/sum(flux[check_pixels]) < 0.10: #Quality check, gausdsian fit should (mostly) get most of the residual line flux
						if abs(g_mean) < 3.0 and g_fwhm > 6.0 and g_fwhm < 10.0 and nansum(abs(g_residuals))/abs(nansum(flux)) < 0.3 and nansum(g_flux) > 0.: #Quality check, gausdsian fit should (mostly) get most of the residual line flux
							#self.plot_1d_velocity(i+1) #Plot 1D spectrum in velocity space (corrisponding to a PV Diagram), called when viewing a line
							#fwhm.append(g_fwhm)
							fwhm[i] = g_fwhm
							clf() #Clear plot
							title(parsed_line_list.label[i] + ', FWHM='+str(g_fwhm) + ', Wavelength=' + str(line_wave))
							plot(velocity, flux, color = 'blue', label='Flux')
							plot(velocity, g(velocity), color = 'red', label='Gaussian')
							plot(velocity, g_residuals, color = 'green', label='Residuals')
							interpolate_line_profile = interp1d(velocity, flux, kind='cubic', bounds_error=False) #Interpolate line profile
							plot(interp_velocity_grid, interpolate_line_profile(interp_velocity_grid), color='blue', label='Interpolation')
							legend()
							pdf.savefig()
							#print 'mean = ', g_mean
							#print 'stddev = ', g_stddev
							#print 'FWHM = ', g_fwhm
				goodfit = fwhm > 0.
				clf()
				plot(parsed_line_list.wave[goodfit], fwhm[goodfit], 'o')
				title('Order = '+str(order_count))
				xlabel('Wavelength')
				ylabel('FWHM')
				xlim([flat_nanmin(order.wave), flat_nanmax(order.wave)])
				pdf.savefig()
				clf()
				plot(x_pixels[goodfit], fwhm[goodfit], 'o')
				xlim([0, len(order.wave)])
				title('Order = '+str(order_count))
				xlabel('x pixel')
				ylabel('FWHM')
				pdf.savefig()
				all_x_pixels = concatenate([all_x_pixels, x_pixels[goodfit]])
				all_wave = concatenate([all_wave, parsed_line_list.wave[goodfit]])
				all_fwhm = concatenate([all_fwhm, fwhm[goodfit]])
				order_count = order_count + 1
			clf()
			plot(all_wave, all_fwhm, 'o')
			title('ALL ORDERS')
			xlabel('Wavelength')
			ylabel('FWHM')
			pdf.savefig()
			clf()
			plot(all_x_pixels, all_fwhm, 'o')
			title('ALL ORDERS')
			xlabel('x pixel')
			ylabel('FWHM')
			pdf.savefig()
			print 'Number of lines with decent Gaussian fits = ', len(all_fwhm)
			print 'All Lines Median FWHM = ', median(all_fwhm)
			print 'All Lines Mean FWHM = ', mean(all_fwhm)
			print 'All Lines std-dev FWHM = ', std(all_fwhm)
	def c_deredden(self, c_value): #Deredden spectrum with a value of "c" measured for H-beta from the literature, while assuming the extinction law of Rieke & Lebofsky (1985)		
		#A_lambda = array([1.531, 1.324, 1.000, 0.748, 0.482,  0.282,  0.175,  0.112,  0.058]) #(A_lambda / A_V) extinction curve from Rieke & Lebofsky (1985) Table 3
		#l = array([ 0.365, 0.445, 0.551, 0.658, 0.806,  1.22 ,  1.63 , 2.19 , 3.45 ]) #Wavelengths for extinction curve from Rieke & Lebofsky (1985)
		#extinction_curve = interp1d(l, A_lambda, kind='quadratic') #Create interpolation object for extinction curve from Rieke & Lebofsky (1985)
		A_V = 0.83446 * 2.5 * c_value #Calcualte A_V from c(h-beta), use linearly interolated A_V/A_hbeta from Rieke & Lebofsky (1985)
		A_K = 0.118 * A_V #Convert A_V to A_K from Fitspatrick (1998)
		#a = 2.14 #extinction curve in the form of a power law from Stead and Hoare (2009)
		a = 1.8
		A_lambda = A_K * self.combospec.wave**(-a) / 2.19**(-a) #Calculate an extinction correction
		#h.F *= 10**(0.4*A_lambda) #Apply extinction correction
		#dereddening = 10**(0.4*extinction_curve(self.combospec.wave)*A_V) #Calculate dereddening as a function of wavelength
		#self.combospec.flux = self.combospec.flux * dereddening #Apply dereddening to flux and noise
		#self.combospec.noise = self.combospec.noise * dereddening
		self.combospec.flux = self.combospec.flux * 10**(0.4*A_lambda) #Apply dereddening to flux and noise
		self.combospec.noise = self.combospec.noise * 10**(0.4*A_lambda)

		


#Class to store and analyze a 2D spectrum
class spec2d:
	def __init__(self, fits_wave, fits_spec, fits_var=[], mask_cosmics=False, interpolate_slit=False):
		try: #First try to see if the wavelength data is from a fits file
			wavedata = fits_wave.get() #Grab fits data for wavelength out of object, first try as if it were a fits object
		except: #If it is not a fits object, say something read out of a json file..
			wavedata = array(fits_wave) #Just copy over the data and get on with it		spec2d = fits_spec.get() #grab all fits data
		spec2d = fits_spec.get() #grab all fits data
		var2d = fits_var.get() #Grab all variance data from fits file
		n_orders = fits_spec.n_orders
		#n_orders = len(spec2d[1].data[:,0]) #Calculate number of orders to use  
		#slit_pixel_length = len(spec2d[0].data[0,:,:]) #Height of slit in pixels for this target and band
		# if interpolate_slit:
		# 	slit_pixel_length = 500 #Height of slit in pixels if we reinterpolate the slit onto a common grid
		# else:
		# 	slit_pixel_length = slit_length  #Height of slit in pixels for this target and band
		slit_pixel_length = slit_length  #Height of slit in pixels for this target and band
		orders = [] #Set up empty list for storing each orders
		#wavedata = wavedata[0].data.byteswap().newbyteorder()
		for i in xrange(n_orders):
			#wave1d = spec2d[1].data[i,:].byteswap().newbyteorder() #Grab wavelength calibration for current order
			wave1d = wavedata[i,:] #Grab wavelength calibration for current order
			#wave2d = tile(wave1d, [slit_pixel_length,1]) #Create a 2D array storing the wavelength solution, to be appended below the data
			#nx, ny, nz = shape(spec2d[0].data.byteswap().newbyteorder())
			#data2d = spec2d[0].data[i,ny-slit_pixel_length-1:ny-1,:].byteswap().newbyteorder() #Grab 2D Spectrum of current order
			nx, ny, nz = shape(spec2d)
			# zero_mask = zeros([ny,nz]) #Find any bad pixels/Cosmic-rays zeroed out by PLP
			# zero_mask[spec2d[i,:,:]==0.] = 1.0
			# zero_mask[roll(spec2d[i,:,:],1,axis=0)==0.] = 1.0 #...along with adjacent pixels...
			# zero_mask[roll(spec2d[i,:,:],-1,axis=0)==0.] = 1.0
			# zero_mask[roll(spec2d[i,:,:],1,axis=1)==0.] = 1.0
			# zero_mask[roll(spec2d[i,:,:],-1,axis=1)==0.] = 1.0
			# spec2d[i,:,:][zero_mask==1.0] = nan #...and turn them into nans, so I don't accidently subtract flux during continuum subtraction (and other possible glitches)
			# var2d[i,:,:][zero_mask==1.0] = nan 
			if interpolate_slit or ny!=slit_pixel_length: #If user specifies interpoalting or the slit size does not match the slit size set by the user, interpolate the darn thing
				data2d = regrid_slit(spec2d[i,:,:], size=slit_pixel_length) 
				noise2d = regrid_slit(sqrt(var2d[i,:,:]), size=slit_pixel_length)
				print 'Slit pixel length does not match, interpolate to fix it!'
			else: #Or just read it in, super simple right?
				data2d = spec2d[i,:,:]
				noise2d = sqrt(var2d[i,:,:])
			# else:
			# 	data2d = spec2d[i,ny-slit_pixel_length-1:ny-1,:]
			# 	#data2d = spec2d[0].data[i,ny-slit_pixel_length-1:ny-1,:].byteswap().newbyteorder() #Grab 2D Spectrum of current order
			# 	#data2d = spec2d[0].data[i,0:slit_pixel_length,:].byteswap().newbyteorder() #Grab 2D Spectrum of current order
			# 	#noise2d = sqrt( var2d[0].data[i,0:slit_pixel_length,:].byteswap().newbyteorder() ) #Grab 2D variance of current order and convert to noise with sqrt(variance)
			# 	#noise2d = sqrt( var2d[0].data[i,ny-slit_pixel_length-1:ny-1,:].byteswap().newbyteorder() ) #Grab 2D variance of current order and convert to noise with sqrt(variance)
			# 	noise2d = sqrt(var2d[i,ny-slit_pixel_length-1:ny-1,:])
			if mask_cosmics: #If user specifies to filter out cosmic rays
				#data2d_vert_sub = data2d - nanmedian(data2d, 0) #subtract vertical spectrum to get rid of sky lines and other junk
				cosmics_found = (abs( (data2d/robust_median_filter(data2d,size=cosmic_horizontal_mask))-1.0) >cosmic_horizontal_limit) & (abs(data2d/noise2d) > cosmic_s2n_min) #Find cosmics where the signal is 100x what is expected from a 3x3 median filter
				data2d[cosmics_found] = nan #And blank the cosmics out
				noise2d[cosmics_found] = nan
			orders.append( spectrum(wave1d, data2d, noise = noise2d) )
		self.orders = orders
		self.n_orders = n_orders
		self.slit_pixel_length = slit_pixel_length
	#This function applies continuum and background subtraction to one order
	def old_subtract_continuum(self, show = False, size = half_block, lines=[], vrange=[-10.0,10.0], linear_fit=False, mask_outliers=False, use_combospec=False, split_trace=False):
		if linear_fit: #WARNING EXPERIMENTAL, If a linear fit, initialize the polynomial fitting routines
			p_init = models.Polynomial1D(degree=2)
			fit_p = fitting.SimplexLSQFitter()
			N = 16 #Divide the order into N segments and calculate the tace along each
			set_size = 2048 / N #Size of each segment to find median of
			median_set = zeros([N, self.slit_pixel_length]) #Initialize variable to hold the median set
			x_for_median_set = arange(0, 2048, set_size) + (set_size/2) #Calculate x pixels for the linear fit for the continuum
			x = arange(2048)
		if use_combospec: #If user specifies to use combined spectrum
			orders = [self.combospec] #Use the combined spectrum
		else: #Else use each order seperately
			orders = self.orders
		for order in orders: #Apply continuum subtraction to each order seperately
			#if sum(isnan(order.flux)) / size(order.flux) < 0.8: #Only try to subtract the continuum if at least 80% of the pixels exist
				#print 'order = ', i, 'number of dimensions = ', num_dimensions
				old_order =  copy.deepcopy(order)
				flux_length = shape(old_order.flux)[1] #Get length (in wavelength space) of flux array
				old_order.flux[nansum(isnan(old_order.flux), axis=1).astype('float')/float(flux_length) > 0.5, :] = 0. #If an entire row is majority nan, zero it out
				if lines != []: #If user supplies a line list
					old_order = mask_lines(old_order, lines, vrange=vrange, ndim=2) #Mask out lines with nan with some velocity range, before applying continuum subtraction
				#stop()
				if use_combospec: #If user wants to use the whole combined spectrum, make seperate traces for H & K bands
					h_band = old_order.wave < 1.85
					k_band = old_order.wave > 1.85
					h_trace = nanmedian(old_order.flux[:,h_band], axis=1) #Get trace of continuum from median of h-band
					k_trace = nanmedian(old_order.flux[:,k_band], axis=1) #Get trace of continuum from median of k-band
					max_y = where(h_trace == flat_nanmax(h_trace))[0][0] #Find peak of trace
					norm_h_trace =  h_trace / median(h_trace[max_y-1:max_y+1]) #Normalize trace
					max_y = where(k_trace == flat_nanmax(k_trace))[0][0] #Find peak of trace
					norm_k_trace =  k_trace / median(k_trace[max_y-1:max_y+1]) #Normalize trace
					#norm_h_trace[isnan(norm_h_trace)] = 0. #Zero out nans in trace incase an entire row in the spectrum is nans
					#norm_k_trace[isnan(norm_k_trace)] = 0. #Zero out nans in trace incase an entire row in the spectrum is nans
				else:
					if split_trace: #If the trace varies siginificantly across orders (for whatever reason), you can try to split the trace and then average the results, this might give a better continuum subtraction
						half_way_point = shape(old_order.flux)[1]/2
						trace = 0.5*(nanmedian(old_order.flux[:,0:half_way_point], axis=1) + nanmedian(old_order.flux[:,half_way_point:]))
					else: #Or else just use the whole order to find the trace, this is the default
						trace = nanmedian(old_order.flux, axis=1) #Get trace of continuum from median of whole order
					trace[isnan(trace)] = 0.0 #Set nan values near edges to zero
					max_y = where(trace == flat_nanmax(trace))[0][0] #Find peak of trace
					norm_trace = trace / median(trace[max_y-1:max_y+1]) #Normalize trace
					#norm_trace[isnan(norm_trace)] = 0. #Zero out nans in trace incase an entire row in the spectrum is nans
				if mask_outliers: #mask columns that deviate significantly from the median continuum trace (ie. emission lines, cosmics, ect.), if the user so desires
					normalize_order_by_column = old_order.flux  / expand_dims(nanmedian(old_order.flux, axis=0), axis=0) #Normalize each column
					if use_combospec: #If user is using the combined spectrum to calculate the h and k band traces seperately
						norm_trace = (norm_h_trace + norm_k_trace) / 2.0 #Simply average the two traces together and use that to find outliers
					divide_normalized_order_by_trace = normalize_order_by_column / expand_dims(norm_trace, axis=1) #Divide the normalized columns by the normalized trace
					deviant_pixels = abs(divide_normalized_order_by_trace) > 200.0 #Find pixels that significantly deviate from the trace, this would be anything from emission lines to high noise
					find_deviant_columns = sum(deviant_pixels, axis=0) > 10 #Find columns with only a few deviant pixels
					old_order.flux[:, find_deviant_columns] = nan #Mask out deviant pixels
					#trace_order = ones([61,2048])*expand_dims(trace, axis = 1)
					#flatten_order_by_trace = old_order.flux / trace_order
					#set_each_column_to_be_unity = flatten_order_by_trace / nanmedian(flatten_order_by_trace, axis=0)
				#old_order.flux[isnan(old_order.flux)] = 0. #Zero out nans when normalizing the flux to get rid of some annoying errors
				if linear_fit: ##WARNING EXPERIMENTAL, If user wants to use a line fit of the trace along the x direction
					for i in xrange(N): #Loop through each segment
						median_set[i,:] = nanmedian(old_order.flux[:,set_size*i: set_size*(i+1)-1], axis=1) #Find trace of this single segment
					normalized_median_set = nanmax(median_set, axis=1) #Normalize the median sets by the trace and then collapse result along slit
					finite_pixels = isfinite(normalized_median_set)
					p = fit_p(p_init, x_for_median_set[finite_pixels], normalized_median_set[finite_pixels])
					result_2d = p(x) * expand_dims(trace, axis=1)
					# p_init = models.Polynomial1D(degree=4)
					# fit_p = fitting.SimplexLSQFitter()
					# nx = len(old_order.flux[0,:])
					# ny = len(old_order.flux[:,0])
				 # 	x = arange(nx)
				 # 	result_2d = zeros([ny,nx])
				 # 	for row in xrange(ny):
				 # 		if any(isfinite(old_order.flux[row,:])):
					# 		#stop()
					# 		p =  fit_p(p_init, x, old_order.flux[row,:])
					#  		result_2d[row,:] = p(x)
					subtracted_flux = order.flux - result_2d
				elif use_combospec: #If user wants to use the whole combined spectrum, make seperate traces for H & K bands
					subtracted_flux = zeros(shape(old_order.flux))
					#Do H-band
					median_result_1d = robust_median_filter(old_order.flux[max_y-1:max_y+1, h_band], size = size) #Take a robust running median along the trace
					median_result_2d = norm_h_trace * expand_dims(median_result_1d, axis = 1) #Expand trace into 2D by multiplying by the robust median
					median_result_2d = median_result_2d.transpose() #Flip axes to match flux axes
					subtracted_flux[:,h_band] = order.flux[:,h_band] - median_result_2d #Apply continuum subtraction
					#Do K-band
					median_result_1d = robust_median_filter(old_order.flux[max_y-1:max_y+1, k_band], size = size) #Take a robust running median along the trace
					median_result_2d = norm_k_trace * expand_dims(median_result_1d, axis = 1) #Expand trace into 2D by multiplying by the robust median
					median_result_2d = median_result_2d.transpose() #Flip axes to match flux axes
					subtracted_flux[:,k_band] = order.flux[:,k_band] - median_result_2d #Apply continuum subtraction

				else: #If user wants to use running median filter
					median_result_1d = robust_median_filter(old_order.flux[max_y-1:max_y+1, :], size = size) #Take a robust running median along the trace
					median_result_2d = norm_trace * expand_dims(median_result_1d, axis = 1) #Expand trace into 2D by multiplying by the robust median
					median_result_2d = median_result_2d.transpose() #Flip axes to match flux axes
					subtracted_flux = order.flux - median_result_2d #Apply continuum subtraction
				order.flux = subtracted_flux
			#if show: #Display subtraction in ds9 if user sets show = True
				#if num_dimensions == 2:
					#show_file = fits.PrimaryHDU(cont_sub.combospec.flux) #Set up fits file object
					#show_file.writeto(scratch_path + 'test_contsub_median.fits', overwrite = True) #Save fits file
					#show_file = fits.PrimaryHDU(old_sci.combospec.flux) #Set up fits file object
					#show_file.writeto(scratch_path + 'test_contsub_before.fits', overwrite = True) #Save fits file
					#show_file = fits.PrimaryHDU(sci.combospec.flux) #Set up fits file object
					#show_file.writeto(scratch_path + 'test_contsub_after.fits', overwrite = True) #Save fits file
					#ds9.open()
					#ds9.show(scratch_path + 'test_contsub_before.fits')
					#ds9.show(scratch_path + 'test_contsub_median.fits', new = True)
					#ds9.show(scratch_path + 'test_contsub_after.fits', new = True)
					#ds9.set('zoom to fit')
					#ds9.set('scale log') #Set view to log scale
					#ds9.set('scale ZScale') #Set scale limits to ZScale, looks okay
					#ds9.set('lock scale')
					#ds9.set('lock colorbar')
					#ds9.set('frame lock image')
					#wait()
					##ds9.close()
				#elif num_dimensions == 1:
					#clf()
					#plot(sci.combospec.wave, sci.combospec.flux, label='Science Target - Continuum Subtracted')
					#plot(old_sci.combospec.wave, old_sci.combospec.flux, label='Science Target - Continuum Not Subtracted')
					#plot(cont_sub.combospec.wave, cont_sub.combospec.flux, label='Continuum Subtraction')
					#legend()
				#else:
					#print 'ERROR: Unable to determine number of dimensions of data, something went wrong'
	def subtract_continuum(self, show = False, size=0, sizes=[500], use_combospec=False): #Subtract continuum and background with an iterative running median
		if size != 0:
			sizes = [size]
		if use_combospec: #If user specifies to use combined spectrum
			orders = [self.combospec] #Use the combined spectrum
		else: #But is usually better to use individual orders instead
			orders = self.orders
		for order in orders: #Apply continuum subtraction to each order seperately
				flux = copy.deepcopy(order.flux)
				whole_order_trace = nanmedian(flux, axis=1)
				whole_order_trace[~isfinite(whole_order_trace)] = 0. #Zero out nans or infinities or other wierd things
				flux = flux - whole_order_trace[:,newaxis] #Do an intiial removal of the flux
				ny, nx = shape(flux) 
				for size in sizes:
					x_left = arange(nx) - size #Create array to store left side of running median
					x_left[x_left < 0] = 0 #Set pixels beyond edge of order to be nonexistant
					x_right = arange(nx) + size #Create array to store right side of running median
					x_right[x_right > nx] = nx - 1 #Set pixels beyond right edge of order to be nonexistant
					x_size = x_right - x_left #Calculate number of pixels in the x (wavelength) direction			
					unmodified_flux = copy.deepcopy(flux)	
					for i in xrange(nx):
							trace = nanmedian(unmodified_flux[:,x_left[i]:x_right[i]], axis=1)
							trace[isnan(trace)] = 0. #Zero out nans or infinities or other wierd things
							flux[:,i] -= trace
				order.flux = flux
	def subtract_median_vertical(self, use_edges=0): #Try to subtract OH residuals and other sky junk by median collapsing along slit and subtracting result. WARNING: ONLY USE FOR POINT OR SMALL SOURCES!
		for i in xrange(self.n_orders-1): #Loop through each order
			if use_edges > 0: #If user specifies using edges, use this many pixels from the edge on each side for median collapse
				edges =concatenate([arange(use_edges), self.slit_pixel_length - arange(use_edges) -1])
				median_along_slit = nanmedian(self.orders[i].flux[edges,:], axis=0) #Collapse median along slit
			else: #Else just median collapse the whole slit
				median_along_slit = nanmedian(self.orders[i].flux, axis=0) #Collapse median along slit
			self.orders[i].flux -= tile(median_along_slit, [self.slit_pixel_length,1]) #Subtract the median
	def combine_orders(self, wave_pivot = default_wave_pivot): #Sitch orders together into one long spectrum
		combospec = copy.deepcopy(self.orders[0]) #Create a spectrum object to append wavelength and flux to
		[order_height, order_length] =  shape(combospec.flux)
		blank = zeros([order_height, order_length*self.n_orders])#Create blanks to store new giant spectrum
		combospec.flux = copy.deepcopy(blank) #apply blanks to everything
		combospec.wave = zeros(order_length*self.n_orders)
 		combospec.noise = copy.deepcopy(blank)
 		#combospec.s2n = copy.deepcopy(blank)
		for i in range(self.n_orders-1, -1, -1): #Loop through each order to stitch one and the following one together
			if i == self.n_orders-1: #If first order, simply throw it in
				xl = 0
				xr = order_length
				#goodpix_next_order =  self.orders[i].wave[0,:] > 0.
				goodpix_next_order =  self.orders[i].wave > 0.
			else: #Else find the wave pivots
				[low_wave_limit, high_wave_limit]  = [flat_nanmin(self.orders[i].wave), combospec.wave[xr-1]] #Find the wavelength of the edges of the already stitched orders and the order currently being stitched to the rest 
				wave_cut = low_wave_limit + wave_pivot*(high_wave_limit-low_wave_limit) #Find wavelength between stitched orders and order to stitch to be the cut where they are combined, with pivot set by global var wave_pivot
				#goodpix_combospec = combospec.wave >= wave_cut #Find pixels in already stitched orders to the left of where the next order will be cut and stitched to
				goodpix_next_order = self.orders[i].wave > wave_cut #Find pixels to the right of the where the order will be cut and stitched to the rest
				#nx = len(self.orders[i].wave[:goodpix_next_order]) #Count number of pixels to add to the blanks
				if combospec.wave[xr-1] > wave_cut:
					xl = where(combospec.wave > wave_cut)[0][0]-1 #Set left pixel to previous right pixel
				else:
					xl = xr-1
				xr = xl + len(self.orders[i].wave[goodpix_next_order])
			combospec.wave[xl:xr] = self.orders[i].wave[goodpix_next_order] #Stitch wavelength arrays together
			combospec.flux[:, xl:xr] = self.orders[i].flux[:, goodpix_next_order]  #Stitch flux arrays together
			combospec.noise[:, xl:xr] = self.orders[i].noise[:, goodpix_next_order] #Stitch noise arrays together
			#combospec.s2n[:, xl:xr] = self.orders[i].s2n[:, goodpix_next_order]  #Stitch S/N arrays together
		combospec.wave = combospec.wave[0:xr] #Get rid of extra pixels at end of arrays
		combospec.flux = combospec.flux[:,0:xr]
		combospec.noise = combospec.noise[:,0:xr]
		#combospec.s2n = combospec.s2n[:,0:xr]
		self.combospec = combospec #save the orders all stitched together
	def plot(self, spec_lines, pause = False, close = False, s2n = False, label_OH = True, num_wave_labels = 50):
		if not hasattr(self, 'combospec'): #Check if a combined spectrum exists
			print 'No spectrum of combined orders found.  Createing combined spectrum.'
			self.combine_orders() #If combined spectrum does not exist, combine the orders
		wave_fits = fits.PrimaryHDU(tile(self.combospec.wave, [self.slit_pixel_length,1]))    #Create fits file containers
		if s2n: #If you want to view the s2n
			spec_fits = fits.PrimaryHDU(self.combospec.s2n())
		else: #You will view the flux
			spec_fits = fits.PrimaryHDU(self.combospec.flux)
		wave_fits.writeto(save.path + 'longslit_wave.fits', overwrite=True)    #Save temporary fits files for later viewing in DS9
		spec_fits.writeto(save.path + 'longslit_spec.fits', overwrite=True)
		ds9.open()  #Display spectrum in DS9
		self.make_label2d(spec_lines, label_lines = True, label_wavelength = True, label_OH = label_OH, num_wave_labels = num_wave_labels) #Label 2D spectrum,
		ds9.show(save.path + 'longslit_wave.fits', new=False)
		self.show_labels() #Load labels
		ds9.show(save.path + 'longslit_spec.fits', new=True)
		self.show_labels() #Load labels
		ds9.set('zoom to fit')
		ds9.set('scale log') #Set view to log scale
		ds9.set('scale ZScale') #Set scale limits to Zscale, looks okay
		ds9.set('frame lock image')
		#Pause for viewing if user specified
		if pause:
			wait()
		#Close DS9 after viewing if user specified (pause should be true or else DS9 will open then close)
		if close:
			ds9.close()
	#Function for labeling up 2D spectrum in DS9, creates a region file storing all the labels and than reads it into, called by show()
	def make_label2d(self, spec_lines, label_lines = True, label_wavelength = True, label_OH = True, num_wave_labels = 50):
		regions = [] #Create list to store strings for creating a DS9 region file
		wave_pixels = self.combospec.wave #Extract 1D wavelength for each pixel
		x = arange(len(wave_pixels)) + 1.0 #Number of pixels across detector
		min_wave  = flat_nanmin(wave_pixels) #Minimum wavelength
		max_wave = flat_nanmax(wave_pixels) #maximum wavelength
		#wave_interp = interp1d(x, wave_pixels, kind = 'linear') #Interpolation for inputting pixel x and getting back wavelength
		x_interp = interp1d(wave_pixels, x, kind = 'linear') #Interpolation for inputting wavlength and getting back pixel x
		top_y = str(self.slit_pixel_length)
		bottom_y = '0'
		label_y = str(1.25*self.slit_pixel_length)
		oh_label_y = str(-0.375*self.slit_pixel_length)
		#x_correction = 2048*(n_orders-i-1) #Push label x position to correct place depending on order      
		if label_wavelength:  #Label wavelengths
			interval = (max_wave - min_wave) / num_wave_labels #Interval between each wavelength label
			wave_labels = arange(min_wave, max_wave, interval) #Store wavleengths of where wave labels are going to go
			x_labels = x_interp(wave_labels) #Grab x positions of the wavelength labels
			for j in xrange(num_wave_labels): #Label the wavelengths #Loop through each wavlength label\
				x_label = str(x_labels[j])
				regions.append('image; line(' + x_label +', '+ top_y + ', ' + x_label + ', ' + bottom_y + ' ) # color=blue ')
				regions.append('image; text('+ x_label +', '+label_y+') # color=blue textangle=90 text={'+str("%12.5f" % wave_labels[j])+'}')
		if label_OH: #Label OH lines
			OH_lines = lines(OH_line_list, delta_v=0.0) #Load OH line list
			show_lines = OH_lines.parse(min_wave, max_wave) #Only grab lines withen the wavelength rang
			num_OH_lines = len(show_lines.wave)
			x_labels = x_interp(show_lines.wave)
			#labels_x of lines to display
			for j in xrange(num_OH_lines): #Label the lines
				x_label = str(x_labels[j])
				regions.append('image; line(' + x_label +', '+ top_y + ', ' + x_label + ', ' + bottom_y + ' ) # color=green ')
				regions.append('image; text('+ x_label +', '+oh_label_y+') # color=green textangle=90 text={OH}')
		if label_lines: #Label lines from a line list
			show_lines = spec_lines.parse(min_wave, max_wave) #Only grab lines withen the wavelength range of the current order
			num_lines = len(show_lines.wave)
			x_labels = x_interp(show_lines.wave)
			#number of lines to display
			for j in xrange(num_lines): #Label the lines
				x_label = str(x_labels[j])
				regions.append('image; line(' + x_label +', '+ top_y + ', ' + x_label + ', ' + bottom_y + ' ) # color=red ')
				regions.append('image; text('+ x_label +', '+label_y+') # color=red textangle=90 text={'+show_lines.label[j]+'}')
		region_file_path = save.path + '2d_labels.reg'
		savetxt(region_file_path, regions, fmt="%s")  #Save region template file for reading into ds9
		#ds9.set('regions ' + region_file_path)
	def show_labels(self): #Called by show() to put line labels in DS9
		region_file_path = save.path + '2d_labels.reg'
		ds9.set('regions ' + region_file_path)
	# def s2n(self): #Estimate noise per pixel
	# 	s2n_obj = copy.deepcopy(self)
	# 	for i in xrange(self.n_orders): #Loop through each order
	# 		median_flux = robust_median_filter(self.orders[i].flux, size=4) #median smooth by four pixels, about the specral & spatial resolution
	# 		random_noise = abs(self.orders[i].flux - median_flux) #Subtract flux from smoothed flux, this should give back the noise
	# 		total_noise = sqrt(random_noise**2 + abs(self.orders[i].flux)) #Calculate S/N from measured random noise and from poisson noise from signal
	# 		s2n = self.orders[i].flux / total_noise
	# 		s2n_obj.orders[i].flux = s2n
	# 	return s2n_obj 
	def deredden(self, A_V): #Deredden spectrum with an assumed A_V, and the extinction curve from Rieke & Lebofsky (1985) Table 3 (see "redden" definition)
		if not hasattr(self, 'combospec'): #Check if a combined spectrum exists
			print 'No spectrum of combined orders found.  Createing combined spectrum.'
			self.combine_orders() #If combined spectrum does not exist, combine the orders 
		R = 3.09 #Assume a Milky way like dust
		E_BV = (-A_V) / R #Calcualte reverse E_BV, by taking the negative of the known A_V
		V = 0. #For dereddening, we reverse redden everything, assuming V mag =0
		B = E_BV #and assuming B mag is the difference between V and B magnitudes E(B-V)
		self.combospec.flux = redden(B, V, self.combospec.wave, self.combospec.flux) #Artificially deredden flux
		self.combospec.noise = redden(B, V, self.combospec.wave, self.combospec.noise) #Artificially scale noise to match S/N of dereddened flux
	def c_deredden(self, c_value): #Deredden spectrum with a value of "c" measured for H-beta from the literature, while assuming the extinction law of Rieke & Lebofsky (1985)		
		#A_lambda = array([1.531, 1.324, 1.000, 0.748, 0.482,  0.282,  0.175,  0.112,  0.058]) #(A_lambda / A_V) extinction curve from Rieke & Lebofsky (1985) Table 3
		#l = array([ 0.365, 0.445, 0.551, 0.658, 0.806,  1.22 ,  1.63 , 2.19 , 3.45 ]) #Wavelengths for extinction curve from Rieke & Lebofsky (1985)
		#extinction_curve = interp1d(l, A_lambda, kind='quadratic') #Create interpolation object for extinction curve from Rieke & Lebofsky (1985)
		#A_V = 0.83446 * 2.5 * c_value #Calcualte A_V from c(h-beta), use linearly interolated A_V/A_hbeta from Rieke & Lebofsky (1985)
		A_V = 2.387 * c_value #Calcualte A_V from c(h-beta), use value given on page 179 of Osterbrock & Ferland 2nd ed at the end of the first paragraph.
		A_K = 0.118 * A_V #Convert A_V to A_K from Fitspatrick (1998)
		#a = 2.14 #extinction curve in the form of a power law from Stead and Hoare (2009)
		a = 1.8
		A_lambda = A_K * self.combospec.wave**(-a) / 2.19**(-a) #Calculate an extinction correction
		#h.F *= 10**(0.4*A_lambda) #Apply extinction correction
		#dereddening = 10**(0.4*extinction_curve(self.combospec.wave)*A_V) #Calculate dereddening as a function of wavelength
		#self.combospec.flux = self.combospec.flux * dereddening #Apply dereddening to flux and noise
		#self.combospec.noise = self.combospec.noise * dereddening
		self.combospec.flux = self.combospec.flux * 10**(0.4*A_lambda) #Apply dereddening to flux and noise
		self.combospec.noise = self.combospec.noise * 10**(0.4*A_lambda)


		
#Generic class for storing a single spectrum, either an order or long spectrum, 1D or 2D
#This serves as the basis for all spectrum objects in this code
class spectrum:
	def __init__(self, wave, flux, noise=[]): #Initialize spectrum by reading in two columned file 
		self.wave = wave #Set up wavelength array 
		self.flux = flux #Set up flux array
		#if "noise" in locals(): #If user specifies a noise
		self.noise_stored = len(noise) >= 1 #Store boolean if noise array actually exists 
		if self.noise_stored:
			self.noise = noise #Save it like flux
			#self.s2n = flux / noise
		else:
			self.noise = zeros(shape(flux))
			#self.s2n = zeros(shape(flux))
	def s2n(self):
		if self.noise_stored: #If noise is stored
			return self.flux/self.noise #Return proper S/N
		else: #But if no noise is stored
			return zeros(shape(self.flux)) #Return an array of zeros for S/N
	def fill_holes(self, ylimit=10, xlimit=3): #Fill nan values with a 3x3 median filter, with the edges avoided by setting ylimit or xlimit
		xsize, ysize = shape(self.flux) #grab size of 2D flux array
		sub_flux = self.flux[xlimit:xsize-xlimit, ylimit:ysize-ylimit] #grab subset of that array with the edges trimmed off
		filtered_flux = median_filter(self.flux, size=[2,2])[xlimit:xsize-xlimit, ylimit:ysize-ylimit]  #Create median filtered data to peg the nan holes with
		find_nans = isnan(sub_flux) #Find the holes
		sub_flux[find_nans] = filtered_flux[find_nans] #Fill in the nan holes with the median filtered data, and our job is now done
	def median_smooth(self, size=[3,3]):  #Experimental feature to median smooth a spectrum to (presumeably) get rid of annoying artifacts
		self.flux = median_filter(self.flux, size=size) #and smooth it, that's it! no more too it really


#~~~~~~~~~~~~~~~~~~~~~~~~~Code for dealing with lines and line lists~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#Class for storing line list
class lines:
	def __init__(self, files, delta_v=0.0, list_dir='line_lists/'): #Initialize line list by providing file in format of wavleength <tab> line label
		if size(files) == 1: #If only one line list is inputted, put single list into array
			files = [files]
			delta_v = [delta_v]
		lab_wave = array([], dtype='f') #Set up array for "lab frame" line wavelengths ie. delta-v = 0 
		wave = array([], dtype='f') #Set up array for line wavelengths
		label = array([], dtype='a')
		count = 0
		for file in files: #Load multiple lists if needed
			if file != 'none':
				input_wave = loadtxt(list_dir+file, unpack=True, dtype='f', delimiter='\t', usecols=(0,)) #Read in line list wavelengths
				input_label = loadtxt(list_dir+file, unpack=True, dtype='a', delimiter='\t', usecols=(1,)) #Read in line list labels
				new_wave = input_wave + input_wave*(delta_v[count]/c) #Shift a line list by some delta_v given by the user
				lab_wave = append(lab_wave, input_wave) #Save wavelengths in the lab frame as well, for later 
				wave = append(wave, new_wave) #Add lines from one list to the (new) wavelength array
				label = append(label, input_label) #Add lines from one list to the label array
			count = count + 1
		sorted = argsort(wave) #sort lines by wavelength
		self.lab_wave = lab_wave[sorted] 
		self.wave = wave[sorted]
		self.label = label[sorted]
	def parse(self, min_wave, max_wave): #Simple function for grabbing only lines with a certain wavelength range
		subset = copy.deepcopy(self) #Make copy of this object to parse
		found_lines = (subset.wave > min_wave) & (subset.wave < max_wave) & (abs(subset.wave - 1.87) > 0.062)   #Grab location of lines only in the wavelength range, while avoiding region between H & K bands
		subset.lab_wave = subset.lab_wave[found_lines] #Filter out lines outside the wavelength range
		subset.wave = subset.wave[found_lines]
		subset.label = subset.label[found_lines]
		return subset #Returns copy of object but with only found lines






#~~~~~~~~~~~~~~~~~~~~~~~~Do a robost running median filter that ignores nan values and outliers, returns result in 1D~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#@jit #Compile Just In Time with numba
def robust_median_filter(input_flux, size = half_block):
	flux = copy.deepcopy(input_flux)
	if ndim(flux) == 2: #For 2D spectrum
		ny, nx = shape(flux) #Calculate npix in x and y
	else: #Else for 1D spectrum
		nx = len(flux) #Calculate npix
	median_result = zeros(nx) #Create array that will store the smoothed median spectrum
	x_left = arange(nx) - size #Create array to store left side of running median
	x_left[x_left < 0] = 0 #Set pixels beyond edge of order to be nonexistant
	x_right = arange(nx) + size #Create array to store right side of running median
	x_right[x_right > nx] = nx - 1 #Set pixels beyond right edge of order to be nonexistant
	x_size = x_right - x_left #Calculate number of pixels in the x (wavelength) direction
	if ndim(flux) == 2: #Run this loop for 2D
		for i in xrange(nx): #This loop does the running of the median down the spectrum each pixel
			median_result[i] = nanmedian(flux[:,x_left[i]:x_right[i]]) #Calculate median between x_left and x_right for a given pixel
	else: #Run this loop for 1D
		for i in xrange(nx): #This loop does the running of the median down the spectrum each pixel
			median_result[i] = nanmedian(flux[x_left[i]:x_right[i]])  #Calculate median between x_left and x_right for a given pixel
	return median_result

#~~~~~~~~~~~~~Mask out lines based on some velocity range, used for not including wide lines in continuum subtraction~~~~~~~~~~~~~~~~~~~~~~~~
def mask_lines(spec, linelist, vrange =[-10.0,10.0], ndim=1):
	sub_linelist = linelist.parse(flat_nanmin(spec.wave), flat_nanmax(spec.wave)) #Pick lines only in wavelength range
	if len(sub_linelist.wave) > 0: #Only do this if there are lines to subtract, if not just pass through the flux array
		for line_wave in sub_linelist.wave: #loop through each line
			velocity = c * ( (spec.wave - line_wave) /  line_wave )
			mask = (velocity >= vrange[0]) & (velocity <= vrange[1])
			#mask = abs(spec.wave - line_wave)  < mask_size #Set up mask around an emission line
			if ndim == 1: #If the number of dimensions is 1
				spec.flux[mask] = nan #Mask emission line in 1D
			else: #else if the number of dimensions is 2
				spec.flux[:,mask] = nan
	return spec #Return flux with lines masked

	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~ Various commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Pauses execution of code to wait for user to hit a key on the command line
def wait():
	raw_input('Press Enter to continue.')


#Simple linear interpolation over nan values
def fill_nans(x, y):
	filled_y = copy.deepcopy(y) #Make copy of y array
	goodpix = isfinite(y) #Find values of y array not filled with nan
	badpix = ~goodpix #Find nans
	interp_y = interp1d(x[goodpix], y[goodpix], bounds_error=False) #Make interpolation object using only finite y values
	filled_y[badpix] = interp_y(x[badpix]) #Replace nan values with interpolated values
	return filled_y #And send it back to where it game from


#~~~~~~~~~~~~~~~~~~~~~~~~~~~Currently unused commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##Apply simple telluric correction by dividing the science spectrum by the flattened standard star spectrum
##This function corrects the orders only and than restitches the orders together into a new combospec
#def simple_telluric_correction(sci, std, quality_cut = True):
	#num_dimensions = ndim(sci.orders[0].wave) #Store number of dimensions
	#if num_dimensions == 2:
		#slit_pixel_length = len(sci.orders[0].flux[:,0]) #Height of slit in pixels for this target and band
	#for i in xrange(sci.n_orders): #Loop through each order
		#if quality_cut: #Generally we throw out bad pixels, but the user can turn this feature off by setting quality_cut = False
			###goodpix = logical_and(sci.orders[i].flux > -100.0, std.orders[i].flux > .1)  #apply the mask
			#goodpix = std.orders[i].flux > .1
			#badpix = ~goodpix
			#std.orders[i].flux[badpix] = nan
		#if num_dimensions == 2:  #For 2D spectra, expand standard star spectrum from 1D to 2D
			#std.orders[i].flux = tile(std.orders[i].flux, [slit_pixel_length,1]) #Expand standard star spectrum into two dimensions
		#sci.orders[i].flux = sci.orders[i].flux / std.orders[i].flux #Divide science spectrum by standard spectrum
	##sci.orders = combine_orders(sci.orders) #Combine the newly corrected orders into one long spectrum
	#return(sci) #Return the new telluric corrected science spectrum
	
##Test expanding and than contracting 2D spectrum for continuum subtraction
#def test_expand_contract(flux):
	#print 'TESTING EXPANSION AND CONTRACTION'
	##stop()
	#flux[~isfinite(flux)] = 0.0 #Make the last few nans =0 so we can actually zoom, (most nans have already been filled)
	#expand = zoom(flux, 4)
	#plot_2d(expand, open = True, new = False, close = True)
	#contract = zoom(expand, 0.25)
	#plot_2d(flux, open = True, new = False, close = False) 
	#plot_2d(contract, open = False, new = True, close = True)
	

#Definition that is a wrapper for displaying a specturm or image in ds9
#Pauses execution of code, continue to close
def plot_2d(image, open = True, new = False, close = True):
	if open: #Open DS9 typically
		ds9.open() #Open DS9
	show_file = fits.PrimaryHDU(image) #Set up fits file object
	show_file.writeto(scratch_path + 'plot_2d.fits', overwrite = True) #Save fits file
	ds9.show(scratch_path + 'plot_2d.fits', new = new) #Show image
	ds9.set('zoom to fit')
	ds9.set('scale log') #Set view to log scale
	ds9.set('scale ZScale') #Set scale limits to ZScale, looks okay
	if new:
		ds9.set('lock scale')
		ds9.set('lock colorbar')
		ds9.set('frame lock image')
	if close:
		wait()
		ds9.close()
	

	
	
#Find lines across all orders and saves it as a line list object
class find_lines:
	def __init__(self, sci, delta_v=0.0, v_range=[-20.0, 20.0], s2n_cut = 100):
		line_waves = array([])
		clf()
		interp_velocity_grid = arange(v_range[0], v_range[1], 0.01) #Velocity grid to interpolate line profiles onto
		master_profile_stack = zeros(size(interp_velocity_grid))
		#for i in xrange(sci.n_orders): #Loop through each order
		for order in sci.orders:
			wave = order.wave
			flux = order.flux
			sig = order.noise
			#flux_filled_nans = fill_nans(wave, flux)
			line_waves_found_for_order = self.search_order(wave, flux)
			line_waves = concatenate([line_waves,  line_waves_found_for_order])
			#interp_flux = interp1d(wave, flux_filled_nans, kind='cubic')
			for line_wave in line_waves_found_for_order:
				velocity = c * ( (wave - line_wave) /  line_wave ) #Get velocity of each pixel
				in_range = (velocity >= 1.1*v_range[0]) & (velocity <= 1.1*v_range[1]) & isfinite(flux) #Isolate pixels in velocity space, near the velocity range desired
				summed_flux = nansum(flux[in_range]) #Sum flux to check for errors
				if summed_flux > 0.: #Fix some errors
					centroid_estimate = abs(nansum(flux[in_range]*velocity[in_range]) /summed_flux) #Find precise centroid of line
					#velocity = c * ( (wave - line_wave) /  line_wave ) - centroid_estimate #Apply a correction for the line centroid
					#in_range = (velocity >= 1.1*v_range[0]) & (velocity <= 1.1*v_range[1]) & isfinite(flux) #Isolate pixels in velocity space, near the velocity range desired
					s2n =  nansum(flux[in_range]) / sqrt(nansum(sig[in_range]**2))
					if s2n > s2n_cut and centroid_estimate < 0.75:
						interp_flux = interp1d(velocity[in_range], flux[in_range], kind='cubic', bounds_error=False) #Cubic interpolate over line profile
						profile = interp_flux(interp_velocity_grid) #Get profile over desired velocity range
						profile = profile / flat_nanmax(profile) #Normalize profile
						plot(interp_velocity_grid, profile)
						master_profile_stack = dstack([master_profile_stack, profile])
			#stop()
			#flux_continuum_subtracted = self.line_continuum_subtract(sci.orders[i].wave, sci.orders[i].flux, line_waves)
			#line_waves = self.search_order(sci.orders[i].wave, flux_continuum_subtracted)
		#stop()
		median_profile = nanmedian(master_profile_stack, 2)[0] #Take median 
		wave = line_waves #Skim over error for now, for some reason waves was = 0.
		self.label = line_waves.astype('|S8')  #Automatically make simple wavelength labels for the found lines
		self.wave = wave #Stores (possibly dopper shifted) waves
		self.lab_wave = line_waves #Save unshifted waves
		self.profile = median_profile
		#wave = line_waves*(delta_v/c)  #Shift a line list by some delta_v given by the user
		self.velocity = interp_velocity_grid
		with PdfPages(save.path + 'save_median_line_profile.pdf') as pdf:
			#clf()
			title('N lines used = ' + str(len(master_profile_stack[0,0,:])))
			ylim([-0.2,1.2])
			plot(interp_velocity_grid, median_profile, '--', color='Black', linewidth=3)
			self.gauss = self.median_fit_gauss() #Fit gaussian, report fwhm
			plot(interp_velocity_grid, self.gauss, ':', color='Red', linewidth=3)
			pdf.savefig()

		

	#Function finds lines using the 2nd derivitive test and saves them as a line list
	def search_order(self, wave, flux, per_order=30):
		#plot(wave, flux)
		finite = isfinite(flux) #Use only finitie pixels (ignore nans)
		fit = UnivariateSpline(wave[finite], flux[finite], s=50.0, k=4) #Fit an interpolated spline
		#for i in range(5):
			#neo_flux = fit(wave)
			#fit = UnivariateSpline(wave, neo_flux, s=50.0, k=4) #Fit an interpolated spline
		extrema = fit.derivative().roots() #Grabe the roots (where the first derivitive = 0) of the fit, these are the extrema (maxes and mins)
		second_deriv = fit.derivative(n=2) #Take second derivitive of fit where the extrema are for 2nd derivitive test
		extrema_sec_deriv = second_deriv(extrema)  #store 2nd derivitives
		i_maxima = extrema_sec_deriv < 0. #Apply the concavity theorm to find maxima
		#i_minima = extrema_sec_deriv > 0. #Ditto for minima
		wave_maxima = extrema[i_maxima]
		flux_maxima = fit(wave_maxima) #Grab flux of the maxima
		#wave_minima = extrema[i_minima]
		#flux_minima = fit(wave_minima)
		flux_smoothed = fit(wave) #Read in spline smoothed fit for plotting
		#plot(wave, flux_smoothed) #Plot fit
		#plot(wave_maxima, flux_maxima, 'o', color='red') #Plot maxima found that pass the cut
		#plot(wave, spline_obj.derivitive(
		#print 'TEST SMOOTHING CONTINUUM'
		#for i in range(len(extrema)): #Print results
			#print extrema[i], extrema_sec_deriv[i]
		#Now cut out lines that are below a standard deviation cut
		#####stddev_flux = std(flux) #Stddeviation of the pixel fluxes
		#####maxima_stddev = flux_maxima / stddev_flux
		#####good_lines = maxima_stddev > threshold
		n_maxima = len(wave_maxima)
		distance_to_nearest_minima = zeros(n_maxima)
		elevation_check = zeros(n_maxima)
		dist_for_elevation_check = 0.00007 #um
		height_fraction = 0.1 #fraction of height
		for i in range(n_maxima):
			#distance_to_nearest_minima[i] = min(abs(wave_maxima[i] - wave_minima))
			peak_height = flux_maxima[i]
			left_height = fit(wave_maxima[i] - dist_for_elevation_check)
			right_height = fit(wave_maxima[i] + dist_for_elevation_check)
			#elevation_check[i] = (peak_height > left_height + height_for_elevation_check) and (peak_height > right_height + height_for_elevation_check)
			elevation_check[i] = (peak_height > left_height / height_fraction) and (peak_height > right_height / height_fraction)
		#good_lines = (distance_to_nearest_minima > 0.00001) & (elevation_check == True)
		good_lines = elevation_check == True
		wave_maxima = wave_maxima[good_lines]
		flux_maxima = flux_maxima[good_lines]
		s = argsort(flux_maxima)[::-1][0:per_order] #Find brightest N per_order (default 15) lines per order
		#plot(wave_maxima, flux_maxima, 'o', color='blue') #Plot maxima found that pass the cut
		#line_object = found_lines(wave_maxima, flux_maxima)
		#return [wave_maxima, flux_maxima]
		#stop()
		return wave_maxima[s]

	#Function masks out existing lines, then tries to find lines again
	def line_continuum_subtract(self, wave, flux, line_waves, line_cut=0.0005):
		for line_wave in line_waves: #Mask out each emission line found
			line_mask = abs(wave - line_wave)  < line_cut #Set up mask around an emission line
			flux[line_mask] = nan #Cut emission line out
		fit = UnivariateSpline(wave, flux, s=1e4, k=4) #Smooth remaining continuum
		continuum = fit(wave) #Grab smoothed continuum
		#stop()
		return flux - continuum #Return flux with continuum subtracted
	def parse(self, min_wave, max_wave): #Simple function for grabbing only lines with a certain wavelength range
		subset = copy.deepcopy(self) #Make copy of this object to parse
		found_lines = (subset.wave > min_wave) & (subset.wave < max_wave) & (abs(subset.wave - 1.87) > 0.062)   #Grab location of lines only in the wavelength range, while avoiding region between H & K bands
		subset.lab_wave = subset.lab_wave[found_lines] #Filter out lines outside the wavelength range
		subset.wave = subset.wave[found_lines]
		subset.label = subset.label[found_lines]
		return subset #Returns copy of object but with only found lines
	def median_fit_gauss(self): #Fit gaussian to median line profile and print results
		fit_g = fitting.LevMarLSQFitter() #Initialize minimization algorithim for fitting gaussian
		g_init = models.Gaussian1D(amplitude=max(self.profile), mean=0.0, stddev=8.0) #Initialize gaussian model for this specific line, centered at 0 km/s with a first guess at the dispersion to be the spectral resolution
		g = fit_g(g_init, self.velocity, self.profile) #Fit gaussian to line
		g_mean = g.mean.value #Grab mean of gaussian fit
		g_stddev = g.stddev.value
		g_fwhm = g_stddev * 2.355
		g_flux = g(self.velocity) 
		g_residuals = self.profile - g_flux
		print 'Median line profile FWHM: ', g_fwhm
		return(g(self.velocity)) #Return gaussian fit

