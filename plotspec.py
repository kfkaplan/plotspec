#This library will eventually be the ultimate IGRINS emission line viewability/analysis code
#
#start as test_new_plotspec.py

#Import libraries
from astropy.io import fits #Use astropy for processing fits files
import pyregion #For reading in regions from DS9 into python
from pylab import *  #Always import pylab because we use it for everything
from scipy.interpolate import interp1d, splev, UnivariateSpline #For interpolating
#from scipy.ndimage import zoom #Was used for continuum subtraction at one point, commented out for now
import ds9 #For scripting DS9
import h2 #For dealing with H2 spectra
import copy #Allow objects to be copied
#from astropy.convolution import convolve, Gaussian1DKernel #, Gaussian2DKernel #For smoothing, not used for now, commented out
from pdb import set_trace as stop #Use stop() for debugging
ion() #Turn on interactive plotting for matplotlib
from matplotlib.backends.backend_pdf import PdfPages
try:  #Try to import bottleneck library, this greatly speeds up things such as nanmedian, nanmax, and nanmin
	from bottleneck import * #Library to speed up some numpy routines
except ImportError:
	print "Bottleneck library not installed.  Code will still run but might be slower.  You can try to bottleneck with 'pip install bottleneck' or 'sudo port install bottleneck' for a speed up."


#Global variables, set after installing plotspec.py
pipeline_path = '/Volumes/IGRINS_data/plp-interpolate/'#Define path to pipeline directory where reduced data is stored
scratch_path = '/Volumes/IGRINS_data/scratch/' #Define path for saving temporary files
data_path = pipeline_path + 'outdata/'
calib_path = pipeline_path + 'calib/primary/'
OH_line_list = 'OH.dat' #Read in OH line list
read_variance = True #Boolean that tells code to use variance for 2D maps (or not) NOTE: This is an experimental feature and not yet implemented in the official pipeline.
#default_wave_pivot = 0.625 #Scale where overlapping orders (in wavelength space) get stitched (0.0 is blue side, 1.0 is red side, 0.5 is in the middle)
default_wave_pivot = 0.75 #Scale where overlapping orders (in wavelength space) get stitched (0.0 is blue side, 1.0 is red side, 0.5 is in the middle)
velocity_range =100.0 # +/- km/s for interpolated velocity grid
velocity_res = 1.0 #Resolution of velocity grid
c = 2.99792458e5 #Speed of light in km/s
slit_length = 62 #Number of pixels along slit in both H and K bands
block = 300 #Block of pixels used for median smoothing, using iteratively bigger multiples of block
half_block = block / 2 #Half of the block used for running median smoothing



#~~~~~~~~~~~~~~~~~~~~~~~~~Code for modifying spectral data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
def telluric_and_flux_calib(sci, std, std_flattened, quality_cut = False, show_plots=True, tweak_test = False):
	vega_file = pipeline_path + 'master_calib/A0V/vegallpr25.50000resam5' #Directory storing Vega standard spectrum     #Set up reading in Vega spectrum
	vega_wave, vega_flux, vega_cont = loadtxt(vega_file, unpack=True) #Read in Vega spectrum
	vega_wave = vega_wave / 1e3 #convert angstroms to microns
	interp_vega_obj = interp1d(vega_wave, vega_flux, kind='linear') #set up interopolation object for calibrated vega spectrum
	interp_cont_obj = interp1d(vega_wave, vega_cont, kind='linear') #ditto for the vega continuum
	num_dimensions = ndim(sci.orders[0].wave) #Store number of dimensions
	if num_dimensions == 2:
		slit_pixel_length = len(sci.orders[0].flux[:,0]) #Height of slit in pixels for this target and band
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#~~~~~~~~~TEST FOR TWEAKING TELLURIC CORRECTION, RUNS A BUNCH OF VARIATIONS IN SCALING AND DELTA-V AND PLOTS THE RESULTS
	#~~~~~~~~~~~IT DOESN'T LOOK LIKE THIS TWEAKING HELPS VERY MUCH,
	if tweak_test:
		clf()
		#while raw_input('Tweak telluric correction? (y/n) ') == 'y':
		#velocity_shift = 0.0
		#a = axes()
		#a.set_autoscale_on(False) 
		for i in xrange(sci.n_orders):
				if i == 0:
					plot(sci.orders[i].wave, sci.orders[i].flux, label = 'No Telluric Correction', color='black')
				else:
					plot(sci.orders[i].wave, sci.orders[i].flux, color='black')
		n=21
		colors=iter(cm.rainbow(np.linspace(0,1,n)))
		for scale in arange(0.8, 1.2, 0.02):
			print 'Current scale is = ', scale
			delta_v = 0.0
			color = next(colors)
			for i in xrange(sci.n_orders):
				shifted_waves = std_flattened.orders[i].wave / ((delta_v/c) + 1.0)
				goodpix = isfinite(std_flattened.orders[i].flux)
				fit = UnivariateSpline(shifted_waves[goodpix], std_flattened.orders[i].flux[goodpix], k=4, s=0.0) #Fit an interpolated spline
				tweaked_telluric_correction = fit(std_flattened.orders[i].wave)**scale
				if i == 0:
					plot(sci.orders[i].wave, sci.orders[i].flux / tweaked_telluric_correction, label = scale, color=color)
				else:
					plot(sci.orders[i].wave, sci.orders[i].flux / tweaked_telluric_correction, color=color)
		legend()
		suptitle('Scaling telluric correction by a power')
		stop()
		# clf()
		# scale = 1.0
		# for i in xrange(sci.n_orders):
		# 		if i == 0:
		# 			plot(sci.orders[i].wave, sci.orders[i].flux, label = 'No Telluric Correction', color='black')
		# 		else:
		# 			plot(sci.orders[i].wave, sci.orders[i].flux, color='black')
		# n=21
		# colors=iter(cm.rainbow(np.linspace(0,1,n)))
		# for smooth in arange(0.0, 5.0, 0.25):
		# 	#g  = Gaussian1DKernel(smooth)
		# 	print 'Current smoothing s is = ', smooth
		# 	delta_v = 0.0
		# 	color = next(colors)
		# 	for i in xrange(sci.n_orders):
		# 		shifted_waves = std_flattened.orders[i].wave / ((delta_v/c) + 1.0)
		# 		goodpix = isfinite(std_flattened.orders[i].flux)
		# 		fit = UnivariateSpline(shifted_waves[goodpix], std_flattened.orders[i].flux[goodpix], k=3, s=smooth) #Fit an interpolated spline
		# 		tweaked_telluric_correction = fit(std_flattened.orders[i].wave)**scale
		# 		#tweaked_telluric_correction = convolve(std_flattened.orders[i].flux, g, boundary='extend')
		# 		if i == 0:
		# 			plot(sci.orders[i].wave, sci.orders[i].flux / tweaked_telluric_correction, label = smooth, color=color)
		# 		else:
		# 			plot(sci.orders[i].wave, sci.orders[i].flux / tweaked_telluric_correction, color=color)
		# legend()
		# suptitle('Smoothing AOV spectrum')
		# stop()
		clf()
		#while raw_input('Tweak telluric correction? (y/n) ') == 'y':
		#velocity_shift = 0.0
		#a = axes()
		#a.set_autoscale_on(False) 
		for i in xrange(sci.n_orders):
				if i == 0:
					plot(sci.orders[i].wave, sci.orders[i].flux, label = 'No Telluric Correction', color='black')
				else:
					plot(sci.orders[i].wave, sci.orders[i].flux, color='black')
		scale = 1.0
		n=10
		colors=iter(cm.rainbow(np.linspace(0,1,n)))
		for delta_v in arange(-2.0, 2.5, 0.5):
			print 'Current velocity is = ', delta_v
			#delta_v = 0.0
			color = next(colors)
			for i in xrange(sci.n_orders):
				shifted_waves = std_flattened.orders[i].wave / ((delta_v/c) + 1.0)
				goodpix = isfinite(std_flattened.orders[i].flux)
				fit = UnivariateSpline(shifted_waves[goodpix], std_flattened.orders[i].flux[goodpix], k=4, s=0.0) #Fit an interpolated spline
				tweaked_telluric_correction = fit(std_flattened.orders[i].wave)**scale
				if i == 0:
					plot(sci.orders[i].wave, sci.orders[i].flux / tweaked_telluric_correction, label = delta_v, color=color)
				else:
					plot(sci.orders[i].wave, sci.orders[i].flux / tweaked_telluric_correction, color=color)
		legend()
		suptitle('Velocity shifting')
		stop()
	#~~~~~~~~~~~~~~~DONE WITH TELLURIC CORRECTOIN TWEAK TEST~~~~~~~~~~~~~~~~~~~~~~~~~~
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	clf()
	for i in xrange(std.n_orders): #Loop through each order
		if quality_cut: #Generally we throw out bad pixels, but the user can turn this feature off by setting quality_cut = False
			goodpix = std_flattened.orders[i].flux > .05
			badpix = ~goodpix
			std.orders[i].flux[badpix] = nan
		#std_continuum =  mask_hydrogen_lines(std.orders[i].wave, std.orders[i].flux / std_flattened.orders[i].flux) #Get back continuum of standard star by dividing it by it's own telluric correction, and interpolate resulting continuum over any H I lines
		std_continuum =  std.orders[i].flux / std_flattened.orders[i].flux #Get back continuum of standard star by dividing it by it's own telluric correction,
		flux_calib = interp_cont_obj(std.orders[i].wave) / std_continuum  #Try a very simple normalization #Try a very simple normalization
		if show_plots:
			#plot(std.orders[i].wave, flux_calib*std_continuum)
			plot(std.orders[i].wave, std_continuum, color='red') #Plot relative flux calibration with H I lines masked on the A0V star
			plot(std.orders[i].wave,  std.orders[i].flux / std_flattened.orders[i].flux, color='blue') #Plot relative flux calibration without H I lines masked on the A0V star
		if num_dimensions == 2:  #For 2D spectra, expand standard star spectrum from 1D to 2D
			std.orders[i].flux = tile(std.orders[i].flux, [slit_pixel_length,1]) #Expand standard star spectrum into two dimensions
			if read_variance:
				std.orders[i].s2n = tile(std.orders[i].s2n, [slit_pixel_length,1]) #Expand standard star spectrum S/N into two dimensions
		sci.orders[i].flux = sci.orders[i].flux * flux_calib/ (std_flattened.orders[i].flux)  #Apply telluric correction and flux calibration
		if read_variance or num_dimensions == 1:
			sci.orders[i].s2n = 1.0/sqrt(sci.orders[i].s2n**-2 + std.orders[i].s2n**-2) #Error propogation after telluric correction, see https://wikis.utexas.edu/display/IGRINS/FAQ or http://chemwiki.ucdavis.edu/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error#Arithmetic_Error_Propagation
			sci.orders[i].noise = sci.orders[i].flux / sci.orders[i].s2n #It's easiest to just work back the noise from S/N after calculating S/N, plus it is now properly scaled to match the (relative) flux calibration
	if show_plots: #Plot Vega spectrum as well for comparison if user wants to see the flux calibration
		#plot(vega_wave, vega_flux)
		show()
		stop()
	return(sci) #Return the spectrum object (1D or 2D) that is now flux calibrated and telluric corrected


#Attempt to corrrect OH residuals by taking the difference between two sky frames and adding it to the science frame
def OH_correction(sci, sky):
	for i in xrange(sci.n_orders): #Loop through each order
		sci.orders[i].flux = sci.orders[i].flux - sky.orders[i].flux #Apply correction
	sci.combine_orders()  #Combine now continuum subracted orders into one long slit spectrum
	return sci



#Class creates, stores, and displays lines as position velocity diagrams, one of the main tools for analysis
class position_velocity:
	def __init__(self, spec1d, spec2d, line_list, s2n=False):
		slit_pixel_length = len(spec2d.flux[:,0]) #Height of slit in pixels for this target and band
		wave_pixels = spec2d.wave[0,:] #Extract 1D wavelength for each pixel
		x = arange(len(wave_pixels)) + 1.0 #Number of pixels across detector
		min_wave  = nanmin(wave_pixels) #Minimum wavelength
		max_wave = nanmax(wave_pixels) #maximum wavelength
		#wave_interp = interp1d(x, wave_pixels, kind = 'linear') #Interpolation for inputting pixel x and getting back wavelength
		x_interp = interp1d(wave_pixels, x, kind = 'linear') #Interpolation for inputting wavlength and getting back pixel x
		interp_velocity = arange(-velocity_range, velocity_range, velocity_res) #Velocity grid to interpolate each line onto
		show_lines = line_list.parse(min_wave, max_wave) #Only grab lines withen the wavelength range of the current order
		flux = [] #Set up list of arrays to store 1D fluxes
		var1d = []
		for line_wave in show_lines.wave: #Label the lines
			pv_velocity = c * ( (spec2d.wave - line_wave) /  line_wave ) #Calculate velocity offset for each pixel from c*delta_wave / wave
			pixel_cut = abs(pv_velocity[0]) <= velocity_range #Find only pixels in the velocity range, this is for conserving flux
			ungridded_velocities = pv_velocity[0, pixel_cut]
			ungridded_flux_1d = spec1d.flux[pixel_cut] #PV diagram ungridded on origional pixels
			ungridded_flux_2d = spec2d.flux[:,pixel_cut] #PV diagram ungridded on origional pixels			
			ungridded_variance_1d = spec1d.noise[pixel_cut]**2 #PV diagram variance ungridded on original pixesl
			if read_variance: #If user specifies read in the 2D variance map
				ungridded_variance_2d = spec2d.noise[:,pixel_cut]**2 #PV diagram variance ungridded on original pixels
			interp_flux_1d= interp1d(ungridded_velocities, ungridded_flux_1d, kind='slinear', bounds_error=False) #Create interp. object for 1D flux
			interp_flux_2d = interp1d(ungridded_velocities, ungridded_flux_2d, kind='slinear', bounds_error=False) #Create interp obj for 2D flux
			interp_variance_1d= interp1d(ungridded_velocities, ungridded_variance_1d, kind='slinear', bounds_error=False) #Create interp obj for 1D variance
			if read_variance: #If user specifies read in the 2D variance map
				interp_variance_2d = interp1d(ungridded_velocities, ungridded_variance_2d, kind='slinear', bounds_error=False) #Create interp obj for 2D variance
			#ungridded_flux_1d = interp_flux_1d(ungridded_velocities) #PV diagram ungridded on origional pixels
			#ungridded_flux_2d = interp_flux_2d(ungridded_velocities) #PV diagram ungridded on origional pixels
			gridded_flux_1d = interp_flux_1d(interp_velocity) #PV diagram velocity gridded
			gridded_flux_2d = interp_flux_2d(interp_velocity) #PV diagram velocity gridded	
			gridded_variance_1d = interp_variance_1d(interp_velocity) #PV diagram variance velocity gridded
			if read_variance: #If user specifies read in 2D variance
				gridded_variance_2d = interp_variance_2d(interp_velocity) #PV diagram variance velocity gridded
			if nanmin(ungridded_flux_1d) != nan: #Check if everything near line is nan, if so skip over this code to avoid bug
				scale_flux_1d = 1.0
				scale_flux_2d = 1.0
				scale_variance_1d = 1.0
				if read_variance: #If user specifies read in 2D variance
					scale_variance_2d = 1.0
			else:
				# if not s2n: #Check that 1D data is not an array of S/N and actually is flux
				# 	scale_flux_1d = nansum(ungridded_result_1d) / nansum(gridded_result_1d) #Scale interpolated flux to original flux so that flux is conserved post-interpolation
				# else: #Or scale signal to noise per pixel to S/N per resolution element (for comparing to the ETC, for example)
				# 	scale_flux_1d = sqrt(3.3) 
				scale_flux_1d = nansum(ungridded_flux_1d) / nansum(gridded_flux_1d) #Scale interpolated flux to original flux so that flux is conserved post-interpolation
				scale_flux_2d = nansum(ungridded_flux_2d) / nansum(gridded_flux_2d) #Scale interpolated flux to original flux so that flux is conserved post-interpolation
				scale_variance_1d = nansum(ungridded_variance_1d) / nansum(gridded_variance_1d) #Scale interpolated variance to original variance so that flux is conserved post-interpolation
				if read_variance: #If user specifies read in 2D variance
					scale_variance_2d = nansum(ungridded_variance_2d) / nansum(gridded_variance_2d) #Scale interpolated variance to original variance so that flux is conserved post-interpolation
			gridded_flux_1d[gridded_flux_1d == nan] = 0. #Get rid of nan values by setting them to zero
			gridded_flux_2d[gridded_flux_2d == nan] = 0. #Get rid of nan values by setting them to zero
			gridded_variance_1d[gridded_variance_1d == nan] = 0. #Get rid of nan values by setting them to zero
			if read_variance: #If user specifies read in 2D variance
				gridded_variance_2d[gridded_variance_2d == nan] = 0. #Get rid of nan values by setting them to zero
			#******************************************************************
			#******************************************************************
			#NEED TO UPDATE LINES BELOW THIS TO HANDLE VARIANCE / UNCERTAINITY
			#******************************************************************
			#******************************************************************
			flux.append(gridded_flux_1d *  scale_flux_1d) #Append 1D flux array with line
			var1d.append(gridded_variance_1d * scale_variance_1d) #Append 1D variacne array with line
			if 'pv' not in locals(): #First line start datacube for 2D spectrum
				pv = gridded_flux_2d.transpose() * scale_flux_2d #Start datacube for 2D PV spectra
				if read_variance: #If user specifies read in 2D variance
					var2d = gridded_variance_2d.transpose() * scale_variance_2d #Start datacube for 2D PV variance
			else: #For all other lines, add 2D PV spectra to the existing datacube
				pv = dstack([pv, gridded_flux_2d.transpose() * scale_flux_2d]) #Stack PV spectrum of lines into a datacube
				if read_variance: #If user specifies read in 2D variance
					 var2d = dstack([var2d, gridded_variance_2d.transpose() * scale_variance_2d]) #Stack PV variance of lines into a datacube
		pv = swapaxes(pv, 0, 2) #Flip axes around in cube so that it can later be saved as a fits file
		if read_variance: #If user specifies read in 2D variance
			var2d = swapaxes(var2d, 0, 2) #Flip axes around in cube so that it can later be saved as a fits file
		self.flux = flux #Save 1D PV fluxes
		self.var1d = var1d #Save 1D PV variances
		self.pv = pv #Save datacube of stack of 2D PV diagrams for each line
		if read_variance: #If user specifies read in 2D variance
			self.var2d = var2d #Save 2D PV variance
		self.velocity = interp_velocity #Save aray storing velocity grid all lines were interpolated onto
		self.label = show_lines.label #Save line labels
		self.lab_wave = show_lines.lab_wave #Save lab wavelengths for all the lines
		self.wave = show_lines.wave #Save line wavelengths
		self.slit_pixel_length = slit_pixel_length #Store number of pixels along slit
		self.s2n = s2n #Store boolean operator if spectrum is a S/N spectrum or not
	def view(self, line='', wave=0.0,  pause = False, close = False, printlines=False): #Function loads 2D PV diagrams in DS9 and plots 1D diagrams
		self.save_fits() #Save a fits file of the pv diagrams for opening in DS9
		ds9.open() #Open DS9
		ds9.show(scratch_path + 'pv.fits', new = False) #Load PV diagrams into DS9
		#if read_variance:
			#XXXXXXXXX
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
			for i in xrange(len(self.label)): #Loop through each line
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
		with PdfPages(scratch_path + pdf_file_name) as pdf: #Make a multipage pd
			for i in xrange(len(self.flux)):
				label_string = self.label[i]
				wave_string = "%12.5f" % self.lab_wave[i]
				title = label_string + '   ' + wave_string + ' $\mu$m'
				self.plot_1d_velocity(i, title=title) #Make 1D plot postage stamp of line
				pdf.savefig() #Save as a page in a PDF file
	def make_2D_postage_stamps(self, pdf_file_name): #Make a PDF showing all 2D lines in a single PDF file
		#figure(figsize=(2,1), frameon=False)
		with PdfPages(scratch_path + pdf_file_name) as pdf: #Make a multipage pd
			for i in xrange(len(self.flux)):
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
	def plot_1d_velocity(self, line_index, title=''): #Plot 1D spectrum in velocity space (corrisponding to a PV Diagram), called when viewing a line
		clf() #Clear plot space
		velocity = self.velocity
		flux = self.flux[line_index]
		noise = sqrt(self.var1d[line_index])
		max_flux = nanmax(flux + noise, axis=0) #Find maximum flux in slice of spectrum
		fill_between(velocity, flux - noise, flux + noise, facecolor = 'red')
		plot(velocity, flux, color='black') #Plot 1D spectrum slice
		#plot(velocity, flux + noise, ':', color='red') #Plot noise level for 1D spectrum slice
		#plot(velocity, flux - noise, ':', color='red') #Plot noise level for 1D spectrum slice
		plot([0,0], [-0.1*max_flux, max_flux], '--', color='blue') #Plot velocity zero point
		xlim([-velocity_range, velocity_range]) #Set xrange to be +/- the velocity range set for the PV diagrams
		ylim([-0.1*max_flux, max_flux]) #Set yrange
		if title != '': #Add title to plot showing line name, wavelength, etc.
			suptitle(title, fontsize=20)
		#if label != '' and wave > 0.0:
			#title(label + ' ' + "%12.5f" % wave + '$\mu$m')
		#elif label != '':
			#title(label)
		#elif wave > 0.0:
			#title("%12.5f" % wave + '$\mu$m')
		xlabel('Velocity [km s$^{-1}$]', fontsize=18) #Label x axis
		#if self.s2n:
		#	ylabel('S/N per resolution element (~3.3 pixels)', fontsize=18) #Label y axis as S/N for S/N spectrum
		#else:
		#	ylabel('Relative Flux', fontsize=18) #Or just label y-axis as relative flux
		ylabel('Relative Flux', fontsize=18) #Or just label y-axis as relative flux 
		#draw()
		#show()
	def save_fits(self): #Save fits file of PV diagrams
		pv_file = fits.PrimaryHDU(self.pv) #Set up fits file object
		#Add WCS for linear interpolated velocity
		pv_file.header['CTYPE1'] = 'km/s' #Set unit to "Optical velocity" (I know it's really NIR but whatever...)
		pv_file.header['CRPIX1'] = (velocity_range / velocity_res) + 1 #Set zero point to where v=0 km/s (middle of stamp)
		pv_file.header['CDELT1'] = velocity_res #Set zero point to where v=0 km/s (middle of stamp)
		pv_file.header['CUNIT1'] = 'km/s' #Set label for x axis to be km/s
		pv_file.header['CTYPE2'] = 'Slit Position' #Set unit for slit length to something generic
		pv_file.header['CRPIX2'] = 1 #Set zero point to 0 pixel for slit length
		pv_file.header['CDELT2'] = 1.0 / self.slit_pixel_length #Set slit length to go from 0->1 so user knows what fraction from the bottom they are along the slit
		pv_file.writeto(scratch_path + 'pv.fits', clobber  = True) #Save fits file
		# if read_variance: #If variance is read in, save signal to noise fits file to also be displayed with pv diagram
		# 	s2n_file = fits.PrimaryHDU(self.s2n) #Set up fits file object
		# 	#Add WCS for linear interpolated velocity
		# 	s2n_file.header['CTYPE1'] = 'km/s' #Set unit to "Optical velocity" (I know it's really NIR but whatever...)
		# 	s2n_file.header['CRPIX1'] = (velocity_range / velocity_res) + 1 #Set zero point to where v=0 km/s (middle of stamp)
		# 	s2n_file.header['CDELT1'] = velocity_res #Set zero point to where v=0 km/s (middle of stamp)
		# 	s2n_file.header['CUNIT1'] = 'km/s' #Set label for x axis to be km/s
		# 	s2n_file.header['CTYPE2'] = 'Slit Position' #Set unit for slit length to something generic
		# 	s2n_file.header['CRPIX2'] = 1 #Set zero point to 0 pixel for slit length
		# 	s2n_file.header['CDELT2'] = 1.0 / self.slit_pixel_length #Set slit length to go from 0->1 so user knows what fraction from the bottom they are along the slit
		# 	s2n_file.writeto(scratch_path + 'pv_s2n.fits', clobber  = True) #Save fits file
	def getline(self, line): #Grabs PV diagram for a single line given a line label
		i =  where(self.label == line)[0][0] #Search for line by label
		return self.pv[i] #Return line found
	def ratio(self, numerator, denominator):  #Returns PV diagram of a line ratio
		return self.getline(numerator) / self.getline(denominator)
	def normalize(self, line): #Normalize all PV diagrams by a single line
		self.pv = self.pv / self.getline(line)
	def basic_flux(self, x_range, y_range):
		sum_along_x = nansum(self.pv[:, y_range[0]:y_range[1], x_range[0]:x_range[1]], axis=2) #Collapse along velocity space
		total_sum = nansum(sum_along_x, axis=1) #Collapse along slit space
		return(total_sum) #Return the integrated flux found for each line in the box defined by the user




class region: #Class for reading in a DS9 region file, and applying it to a position_velocity object
	def __init__(self, pv, file='', background=''):
		use_background_region = False
		line_labels =  pv.label #Read out line labels
		line_wave = pv.lab_wave #Read out (lab) line wavelengths
		pv_data = pv.pv #Holder for flux datacube
		pv_variance = pv.var2d #holder for variance datacube
		bad_data = pv_data < -10000.0  #Mask out bad pixels and cosmic rays that somehow made it through
		pv_data[bad_data] == nan
		pv_variance[bad_data] == nan
		pv_shape = shape(pv_data[0,:,:]) #Read out shape of a 2D slice of the pv diagram cube
		n_lines = len(pv_data[:,0,0]) #Read out number of lines
		if file == '': #If no region file is specified by the user, prompt user for the path to the region file
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
			use_background_regiuon = False
		on_region = pyregion.open(file)  #Open region file for reading flux
		on_mask = on_region.get_mask(shape = pv_shape) #Make mask around region file
		on_patch_list, on_text_list = on_region.get_mpl_patches_texts() #Do some stuff
		if use_background_region: #If you want to use another region to designate the background, read it in here
			off_region = pyregion.open(background) #Read in background region file
			off_mask = off_region.get_mask(shape = pv_shape) #Set up mask
			off_patch_list, off_text_list = off_region.get_mpl_patches_texts() #Do some stuff
		figure(figsize=(3.0,1.5), frameon=False) #Set up figure check size
		with PdfPages(scratch_path + 'flux_check.pdf') as pdf: #Make a multipage pdf
			line_flux = zeros(n_lines) #Set up array to store line fluxes
			line_s2n = zeros(n_lines) #Set up array to store line S/N, set = 0 if no variance is found
			line_sigma = zeros(n_lines) #Set up array to store 1 sigma uncertainity
			for i in xrange(n_lines): #Loop through each line
				#subplot(n_subfigs, n_subfigs, i+1)
				clf() #Clear plot field
				frame = gca() #Turn off axis number labels
				frame.axes.get_xaxis().set_ticks([]) #Turn off axis number labels
				frame.axes.get_yaxis().set_ticks([]) #Turn off axis number labels
				ax = subplot(111) #Turn on "ax"
				on_data = pv_data[i,:,:][on_mask] #Find data inside the region for grabbing the flux
				on_variance = pv_variance[i,:,:][on_mask]
				sigma = 0.0
				if use_background_region: #If a backgorund region is specified
					off_data = pv_data[i,:,:][off_mask] #Find data in the background region for calculating the background
					background = nanmedian(off_data) * size(on_data) #Calculate backgorund from median of data in region and multiply by area of region used for summing flux
				else: #If no background region is specified by the user, use the whole field 
					background = nanmedian(pv_data[i,:,:]) * size(on_data) #Get background from median of all data in field and multiply by area of region used for summing flux
				line_flux[i] = nansum(on_data) - background #Calculate flux from sum of pixels in region minus the background (which is the median of some region or the whole field, multiplied by the area of the flux region)
				if read_variance: #If user sets variance to be read in
					line_sigma[i] =  sqrt( nansum(on_variance) ) #Store 1 sigma uncertainity for line
					line_s2n[i] = line_flux[i] / line_sigma[i] #Calculate the S/N in the region of the line
				imshow(pv_data[i,:,:], cmap='gray') #Save preview of line and region(s)
				suptitle('i = ' + str(i+1) + ',    '+ line_labels[i] +'  '+str(line_wave[i])+',   Flux = ' + '%.3e' % line_flux[i] + ',   S/N = ' + '%.1f' % line_s2n[i] ,fontsize=6)
				for p in on_patch_list: #Display DS9 regions in matplotlib
				    ax.add_patch(p)
				for t in on_text_list:
				    ax.add_artist(t)
				if use_background_region:
					for p in off_patch_list:
					    ax.add_patch(p)
					for t in off_text_list:
					    ax.add_artist(t)
				pdf.savefig() #Add figure as a page in the pdf
		figure(figsize=(11, 8.5), frameon=False) #Reset figure size
		self.wave = line_wave #Save wavelength of lines
		self.label = line_labels #Save labels of lines
		self.flux = line_flux #Save line fluxes
		self.s2n = line_s2n #Save line S/N
		self.sigma = line_sigma #Save the 1 sigma limit
	#def process_h2(self): #Create and store fluxes for H2 for this region, used mainly for testing, user should use this code in their own scripts to combine H2 lines from H and K bands
		#h2_transitions = h2.make_line_list() #Set up H2 transition object
		#h2_transitions.set_flux_in_region_obj(self.label, self.flux, self.s2n, self.sigma) #Read in fluxes for this region and assign them each to the appropriate H2 line
		#h2_transitions.calculate_column_density() #Calculate column densities
		#h2_transitions.quick_plot()
		#print 'Done with H2 stuffits'
		#stop() #For testing




#~~~~~~~~~~~~~~~~~~~~~~~~~Code for reading in analyzing spectral data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~j

#Convenience function for making a single spectrum object in 1D or 2D that combines both H & K bands while applying telluric correction and flux calibration
#The idea is that the user can call a single line and get a single spectrum ready to go
def getspec(date, waveno, frameno, stdno, twodim=True):
	#Make 1D spectrum object for standard star
	H_std_obj = makespec(date, 'H', waveno, stdno) #Read in H-band
	K_std_obj = makespec(date, 'K', waveno, stdno) #Read in H-band
	std_obj = H_std_obj #Create master object
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
		H_sci2d_obj =  makespec(date, 'H', waveno, frameno, twodim=True) #Read in H-band
		K_sci2d_obj =  makespec(date, 'K', waveno, frameno, twodim=True) #Read in K-band
		#if H_sci2d_obj.slit_pixel_length != K_sci2d_obj.slit_pixel_length:
		#print 'H slit length: ', H_sci2d_obj.slit_pixel_length
		#print 'K slit length: ', K_sci2d_obj.slit_pixel_length
		sci2d_obj = H_sci2d_obj #Create master object
		sci2d_obj.orders = K_sci2d_obj.orders + H_sci2d_obj.orders #Combine orders
		sci2d_obj.n_orders = K_sci2d_obj.n_orders + H_sci2d_obj.n_orders #Find new total number of orders
	#Apply telluric correction & relative flux calibration
	spec1d = telluric_and_flux_calib(sci1d_obj, std_obj, stdflat_obj, show_plots=False) #For 1D spectrum
	if twodim: #If user specifies this object has a 2D spectrum
		spec2d = telluric_and_flux_calib(sci2d_obj, std_obj, stdflat_obj, show_plots=False) #Run for 2D spectrum
	#Return either 1D and 2D spectra, or just 1D spectrum if no 2D spectrum exists
	if twodim:
		return spec1d, spec2d #Return both 1D and 2D spectra objects
	else:
		return spec1d #Only return 1D spectra object
		



#Wrapper for easily creating a 1D or 2D comprehensive spectrum object of any type, allowing user to import an entire specturm object in one line
def makespec(date, band, waveno, frameno, std=False, twodim=False, s2n=False):
	spec_data = fits_file(date, frameno, band, std=std, twodim=twodim, s2n=s2n) #Read in data from spectrum
	wave_data = fits_file(date, waveno, band, wave=True) #If 1D, read in data from wavelength solution
	if twodim and not read_variance: #If spectrum is 2D but no variance data to be read in
		spec_obj = spec2d(wave_data, spec_data) #Create 2D spectrum object
	elif twodim and read_variance: #If spectrum is 2D and variance data will be read in
		var_data = fits_file(date, frameno, band, var2d=True) #Grab data for 2D variance cube
		spec_obj = spec2d(wave_data, spec_data, fits_var=var_data) #Create 2D spectrum object, with variance data inputted to get S/N
	else: #If spectrum is 1D
		var_data = fits_file(date, frameno, band, var1d=True) 
		spec_obj = spec1d(wave_data, spec_data, var_data) #Create 1D spectrum object
	return(spec_obj) #Return the fresh spectrum object!
	
	

#Class stores information about a fits file that has been reduced by the PLP
class fits_file:
	def __init__(self, date, frameno, band, std=False, wave=False, twodim=False, s2n=False, var1d=False, var2d=False):
		self.date = '%.4d' % int(date) #Store date of observation
		self.frameno =  '%.4d' % int(frameno) #Store first frame number of observation
		self.band = band #Store band name 'H' or 'K'
		self.std = std #Store if file is a standard star
		self.wave = wave #Store if file is a wavelength solution
		self.s2n = s2n #Store if file is the S/N spectrum
		self.twodim = twodim #Store if file is of a 2D spectrum instead of a 1D spectrum
		self.var1d = var1d
		self.var2d = var2d #Store if file is a 2D variance map (like twodim but with variance instead of signal)
		self.path = self.filepath() #Determine path and filename for fits file
		self.data = fits.open(self.path) #Open fits file and put data into memory
		#print self.path
	def filepath(self): #Given input variables, determine the path to the target fits file
		prefix =  'SDC' + self.band + '_' + self.date + '_' + self.frameno #Set beginning (prefix) of filename
		if self.std: #If file is for a standard star and you want the flattened spectrum
			postfix = '.spec_flattened.fits'
			master_path = data_path
		elif self.wave: #If file is the 1D wavelength calibration
			prefix = 'SKY_' + prefix
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
		return self.data

#Class to store and analyze a 1D spectrum
class spec1d:
	def __init__(self, fits_wave, fits_spec, fits_var):
		wavedata = fits_wave.get() #Grab fits data for wavelength out of object
		specdata = fits_spec.get() #Grab fits data for flux out of object
		vardata = fits_var.get() #Grab fits data for variance
		orders = [] #Set up empty list for storing each orders
		n_orders = len(specdata[0].data[:,0]) #Count number of orders in spectrum
		wavedata = wavedata[0].data.byteswap().newbyteorder() #Read out wavelength and flux data from fits files into simpler variables
		fluxdata = specdata[0].data.byteswap().newbyteorder() #Read out wavelength and flux data from fits files into simpler variables
		noisedata = sqrt( vardata[0].data.byteswap().newbyteorder() ) #Read out noise from fits file into a simpler variable by taking the square root of the variance
		for i in xrange(n_orders): #Loop through to process each order seperately
			orders.append( spectrum(wavedata[i,:], fluxdata[i,:], noise=noisedata[i,:])  ) #Append order to order list
		self.n_orders = n_orders
		self.orders = orders
	def subtract_continuum(self, show = False): #Subtract continuum using robust running median
		if show: #If you want to watch the continuum subtraction
			clf() #Clear interactive plot
			first_order = True #Keep track of where we are in the so we only create the legend on the first order
		for order in self.orders: #Apply continuum subtraction to each order seperately
			old_flux = copy.deepcopy(order.flux) #Make copy of flux array so the original is not modified
			wave = order.wave #Read n wavelength array
			median_result_1d = robust_median_filter(old_flux) #Take a robust running median along the trace, this is the found continuum
			subtracted_flux = old_flux - median_result_1d #Apply continuum subtraction
			if show: #If you want to watch the continuum subtraction
				if first_order:  #If on the first order, make the legend along with plotting the order
					plot(wave, subtracted_flux, label='Science Target - Continuum Subtracted', color='black')
					plot(wave, old_flux, label='Science Target - Continuum Not Subtracted', color='blue')
					plot(wave, median_result_1d, label='Continuum Subtraction', color='green')
					first_order = False #Now that we are done, just plot the ldata for all the other orders without making a long legend
				else: #Else just plot the order
					plot(wave, subtracted_flux, color='black')
					plot(wave, old_flux, color='blue')
					plot(wave, median_result_1d, color='green')
			order.flux = subtracted_flux #Replace this order's flux array with one that has been continuum subtracted
		if show: #If you want to watch the continuum subtraction
			legend() #Show the legend in the plot
	def combine_orders(self, wave_pivot = default_wave_pivot): #Sitch orders together into one long spectrum
		combospec = copy.deepcopy(self.orders[0]) #Create a spectrum object to append wavelength and flux to
		for i in xrange(self.n_orders-1): #Loop through each order to stitch one and the following one together
			[low_wave_limit, high_wave_limit]  = [nanmin(combospec.wave), nanmax(self.orders[i+1].wave)] #Find the wavelength of the edges of the already stitched orders and the order currently being stitched to the rest 
			wave_cut = low_wave_limit + wave_pivot*(high_wave_limit-low_wave_limit) #Find wavelength between stitched orders and order to stitch to be the cut where they are combined, with pivot set by global var wave_pivot
			goodpix_combospec = combospec.wave >= wave_cut #Find pixels in already stitched orders to the left of where the next order will be cut and stitched to
			goodpix_next_order = self.orders[i+1].wave < wave_cut #Find pixels to the right of the where the order will be cut and stitched to the rest
			combospec.wave = concatenate([self.orders[i+1].wave[goodpix_next_order], combospec.wave[goodpix_combospec] ]) #Stitch wavelength arrays together
			combospec.flux = concatenate([self.orders[i+1].flux[goodpix_next_order], combospec.flux[goodpix_combospec] ]) #Stitch flux arrays together
			combospec.noise = concatenate([self.orders[i+1].noise[goodpix_next_order], combospec.noise[goodpix_combospec] ])  #Stitch noise arrays together
			combospec.s2n = concatenate([self.orders[i+1].s2n[goodpix_next_order], combospec.s2n[goodpix_combospec] ]) #Stitch S/N arrays together
		self.combospec = combospec #save the orders all stitched together
	#Simple function for plotting a 1D spectrum orders
	def plot(self):
		for order in self.orders: #Plot each order
			plot(order.wave, order.flux)
		xlabel('Wavelength [$\mu$m]')
		ylabel('Relative Flux')
		show()
		#draw()
	#Plot spectrum with lines from line list overplotted
	def plotlines(self, linelist, threshold=0.0, model='', rows=5):
		if not self.combospec in vars(): #Check if a combined spectrum exists
			print 'No spectrum of combined orders found.  Createing combined spectrum.'
			self.combine_orders() #Combine spectrum before plotting, if not done already
		clf() #Clear plot
		min_wave  = min(self.combospec.wave) #Find maximum wavelength
		max_wave  = max(self.combospec.wave) #Find minimum wavelength
		max_flux = nanmax(self.combospec.flux, axis=0)
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
				if sub_linelist.flux[i] > max_flux*threshold and nanmax(sub_linelist.flux[other_lines], axis=0) == sub_linelist.flux[i]: #if line is the highest of all surrounding lines within some window
					if sub_linelist.label[i] == '{OH}': #If OH lines appear in line list.....
						mask_these_pixels = abs(self.combospec.wave-sub_linelist.wave[i]) < 0.00006 #Create mask of OH lines...
						self.combospec.flux[mask_these_pixels] = nan #Turn all pixels with OH lines into numpy nans so the OH lines don't get plotted
						#plot([linelist_wave[i], linelist_wave[i]], [linelist_flux[i]+max_flux*0.025, max_flux*0.92], ':', color='gray')
						#text(linelist_wave[i], linelist_flux[i]+max_flux*0.02, '$\oplus$', rotation=90, fontsize=9, verticalalignment='bottom', horizontalalignment='center', color='black') 
					else:   #If no OH lines found, plot lines on figure
						plot([sub_linelist.wave[i], sub_linelist.wave[i]], [sub_linelist.flux[i], sub_linelist.flux[i] + max_flux*0.065], ':', color='black') #Plot location of line as a dotted line a little bit above the spectrum
						text(sub_linelist.wave[i], sub_linelist.flux[i] +  max_flux*0.073, sub_linelist.label[i], rotation=90, fontsize=9.5, verticalalignment='bottom', horizontalalignment='center', color='black')  #Label line with text
			plot(self.combospec.wave[sci_in_range], self.combospec.flux[sci_in_range], color='blue') #Plot actual spectrum
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
	def mask_OH(self): #Mask OH lines, use only after processing and combinging spectrum to make a cleaner 1D spectrum
		OH_lines = lines(OH_line_list, delta_v=0.0) #Load OH line list
		parsed_OH_lines = OH_lines.parse( min(self.combospec.wave), max(self.combospec.wave))
		for i in xrange(len(parsed_OH_lines.wave)): #Loop through each line
			mask_these_pixels = abs(self.combospec.wave-parsed_OH_lines.wave[i]) < 0.00006 #Create mask of OH lines...
			self.combospec.flux[mask_these_pixels] = nan #Turn all pixels with OH lines into numpy nans so the OH lines don't get plotted



#Class to store and analyze a 2D spectrum
class spec2d:
	def __init__(self, fits_wave, fits_spec, fits_var=[]):
		wavedata = fits_wave.get() #Grab fits data for wavelength out of object
		spec2d = fits_spec.get() #grab all fits data
		if read_variance: #If reading in 2D variance data
			var2d = fits_var.get() #Grab all variance data from fits file
		n_orders = len(spec2d[1].data[:,0]) #Calculate number of orders to use  
		#slit_pixel_length = len(spec2d[0].data[0,:,:]) #Height of slit in pixels for this target and band
		slit_pixel_length = slit_length  #Height of slit in pixels for this target and band
		orders = [] #Set up empty list for storing each orders
		wavedata = wavedata[0].data.byteswap().newbyteorder()
		for i in xrange(n_orders):
			#wave1d = spec2d[1].data[i,:].byteswap().newbyteorder() #Grab wavelength calibration for current order
			wave1d = wavedata[i,:] #Grab wavelength calibration for current order
			wave2d = tile(wave1d, [slit_pixel_length,1]) #Create a 2D array storing the wavelength solution, to be appended below the data
			data2d = spec2d[0].data[i,0:slit_pixel_length,:].byteswap().newbyteorder() #Grab 2D Spectrum of current order
			if read_variance:
				noise2d = sqrt( var2d[0].data[i,0:slit_pixel_length,:].byteswap().newbyteorder() ) #Grab 2D variance of current order and convert to noise with sqrt(variance)
				orders.append( spectrum(wave2d, data2d, noise = noise2d) )
			else: 
				orders.append( spectrum(wave2d, data2d) )
		self.orders = orders
		self.n_orders = n_orders
		self.slit_pixel_length = slit_pixel_length
	#This function applies continuum and background subtraction to one order
	def subtract_continuum(self, show = False):
		for order in self.orders: #Apply continuum subtraction to each order seperately
			#print 'order = ', i, 'number of dimensions = ', num_dimensions
			old_flux =  copy.deepcopy(order.flux)
			#stop()
			trace = nanmedian(old_flux, axis=1) #Get trace of continuum from median of whole order
			trace[isnan(trace)] = 0.0 #Set nan values near edges to zero
			max_y = where(trace == max(trace))[0][0] #Find peak of trace
			norm_trace = trace / median(trace[max_y-1:max_y+1]) #Normalize trace
			median_result_1d = robust_median_filter(old_flux[max_y-1:max_y+1, :]) #Take a robust running median along the trace
			median_result_2d = norm_trace * expand_dims(median_result_1d, axis = 1) #Expand trace into 2D by multiplying by the robust median
			median_result_2d = median_result_2d.transpose() #Flip axes to match flux axes
			subtracted_flux = old_flux - median_result_2d #Apply continuum subtraction
			order.flux = subtracted_flux
		#if show: #Display subtraction in ds9 if user sets show = True
			#if num_dimensions == 2:
				#show_file = fits.PrimaryHDU(cont_sub.combospec.flux) #Set up fits file object
				#show_file.writeto(scratch_path + 'test_contsub_median.fits', clobber = True) #Save fits file
				#show_file = fits.PrimaryHDU(old_sci.combospec.flux) #Set up fits file object
				#show_file.writeto(scratch_path + 'test_contsub_before.fits', clobber = True) #Save fits file
				#show_file = fits.PrimaryHDU(sci.combospec.flux) #Set up fits file object
				#show_file.writeto(scratch_path + 'test_contsub_after.fits', clobber = True) #Save fits file
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
	def combine_orders(self, wave_pivot = default_wave_pivot): #Sitch orders together into one long spectrum
		combospec = copy.deepcopy(self.orders[0]) #Create a spectrum object to append wavelength and flux to
		for i in xrange(self.n_orders-1): #Loop through each order to stitch one and the following one together
			[low_wave_limit, high_wave_limit]  = [nanmin(combospec.wave), nanmax(self.orders[i+1].wave)] #Find the wavelength of the edges of the already stitched orders and the order currently being stitched to the rest 
			wave_cut = low_wave_limit + wave_pivot*(high_wave_limit-low_wave_limit) #Find wavelength between stitched orders and order to stitch to be the cut where they are combined, with pivot set by global var wave_pivot
			goodpix_combospec = combospec.wave[0,:] >= wave_cut #Find pixels in already stitched orders to the left of where the next order will be cut and stitched to
			goodpix_next_order = self.orders[i+1].wave[0,:] < wave_cut #Find pixels to the right of the where the order will be cut and stitched to the rest
			combospec.wave = concatenate([self.orders[i+1].wave[:, goodpix_next_order], combospec.wave[:, goodpix_combospec] ], axis=1) #Stitch wavelength arrays together
			combospec.flux = concatenate([self.orders[i+1].flux[:, goodpix_next_order], combospec.flux[:, goodpix_combospec] ], axis=1)#Stitch flux arrays together
			combospec.noise = concatenate([self.orders[i+1].noise[:, goodpix_next_order], combospec.noise[:, goodpix_combospec] ], axis=1) #Stitch noise arrays together
			combospec.s2n = concatenate([self.orders[i+1].s2n[:, goodpix_next_order], combospec.s2n[:, goodpix_combospec] ], axis=1) #Stitch s2n arrays together
		self.combospec = combospec #save the orders all stitched together
	#Simple function for displaying the combined 2D spectrum
	def plot(self, spec_lines, pause = False, close = False, s2n = False):
		if not self.combospec in vars(): #Check if a combined spectrum exists
			print 'No spectrum of combined orders found.  Createing combined spectrum.'
			self.combine_orders() #If combined spectrum does not exist, combine the orders
		wave_fits = fits.PrimaryHDU(self.combospec.wave)    #Create fits file containers
		if s2n: #If you want to view the s2n
			spec_fits = fits.PrimaryHDU(self.combospec.s2n)
		else: #You will view the flux
			spec_fits = fits.PrimaryHDU(self.combospec.flux)
		wave_fits.writeto(scratch_path + 'longslit_wave.fits', clobber=True)    #Save temporary fits files for later viewing in DS9
		spec_fits.writeto(scratch_path + 'longslit_spec.fits', clobber=True)
		ds9.open()  #Display spectrum in DS9
		self.make_label2d(spec_lines, label_lines = True, label_wavelength = True, label_OH = True, num_wave_labels = 50) #Label 2D spectrum,
		ds9.show(scratch_path + 'longslit_wave.fits', new=False)
		self.show_labels() #Load labels
		ds9.show(scratch_path + 'longslit_spec.fits', new=True)
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
		wave_pixels = self.combospec.wave[0,:] #Extract 1D wavelength for each pixel
		x = arange(len(wave_pixels)) + 1.0 #Number of pixels across detector
		min_wave  = nanmin(wave_pixels, axis=0) #Minimum wavelength
		max_wave = nanmax(wave_pixels, axis=0) #maximum wavelength
		wave_interp = interp1d(x, wave_pixels, kind = 'linear') #Interpolation for inputting pixel x and getting back wavelength
		x_interp = interp1d(wave_pixels, x, kind = 'linear') #Interpolation for inputting wavlength and getting back pixel x
		top_y = str(self.slit_pixel_length)
		bottom_y = '0'
		label_y = str(1.5*self.slit_pixel_length)
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
				regions.append('image; text('+ x_label +', '+label_y+') # color=green textangle=90 text={OH}')
		if label_lines: #Label lines from a line list
			show_lines = spec_lines.parse(min_wave, max_wave) #Only grab lines withen the wavelength range of the current order
			num_lines = len(show_lines.wave)
			x_labels = x_interp(show_lines.wave)
			#number of lines to display
			for j in xrange(num_lines): #Label the lines
				x_label = str(x_labels[j])
				regions.append('image; line(' + x_label +', '+ top_y + ', ' + x_label + ', ' + bottom_y + ' ) # color=red ')
				regions.append('image; text('+ x_label +', '+label_y+') # color=red textangle=90 text={'+show_lines.label[j]+'}')
		region_file_path = scratch_path + '2d_labels.reg'
		savetxt(region_file_path, regions, fmt="%s")  #Save region template file for reading into ds9
		#ds9.set('regions ' + region_file_path)
	def show_labels(self): #Called by show() to put line labels in DS9
		region_file_path = scratch_path + '2d_labels.reg'
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




		
#Generic class for storing a single spectrum, either an order or long spectrum, 1D or 2D
#This serves as the basis for all spectrum objects in this code
class spectrum:
	def __init__(self, wave, flux, noise=[]): #Initialize spectrum by reading in two columned file 
		self.wave = wave #Set up wavelength array 
		self.flux = flux #Set up flux array
		if "noise" in locals(): #If user specifies a noise
			self.noise = noise #Save it like flux
			self.s2n = flux / noise
		else:
			self.noise = zeros(shape(flux))
			self.s2n = zeros(shape(flux))

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


#~~~~~~~Definitions for identifying lines in a spectrum, givin a wavelength and flux


#~~~~~~~~~~~~~~~~~~~~~~~~Do a robost running median filter that ignores nan values and outliers, returns result in 1D~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~ Various commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Pauses execution of code to wait for user to hit a key on the command line
def wait():
	raw_input('Press Enter to continue.')


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
	
##Simple function for masking out lines in wavelength space, returns flux array with lines masked as 'nan'
#def mask_lines(spec, linelist, mask_size =  0.00006):
	#sub_linelist = linelist.parse(bottleneck.nanmin(spec.wave), bottleneck.nanmax(spec.wave)) #Pick lines only in wavelength range
	#if len(sub_linelist.wave) > 0: #Only do this if there are lines to subtract, if not just pass through the flux array
		#for line_wave in sub_linelist.wave: #loop through each line 
			#mask = abs(spec.wave - line_wave)  < mask_size #Set up mask around an emission line
			#spec.flux[mask] = nan #Mask emission line
		#print 'masked', len(sub_linelist.wave), lines, ' with wavlenegth mask of ', mask_size, 'um'
	#return spec.flux #Return flux with lines masked


# #Definition that is a wrapper for displaying a specturm or image in ds9
# #Pauses execution of code, continue to close
# def plot_2d(image, open = True, new = False, close = True):
# 	if open: #Open DS9 typically
# 		ds9.open() #Open DS9
# 	show_file = fits.PrimaryHDU(image) #Set up fits file object
# 	show_file.writeto(scratch_path + 'plot_2d.fits', clobber = True) #Save fits file
# 	ds9.show(scratch_path + 'plot_2d.fits', new = new) #Show image
# 	ds9.set('zoom to fit')
# 	ds9.set('scale log') #Set view to log scale
# 	ds9.set('scale ZScale') #Set scale limits to ZScale, looks okay
# 	if new:
# 		ds9.set('lock scale')
# 		ds9.set('lock colorbar')
# 		ds9.set('frame lock image')
# 	if close:
# 		wait()
# 		ds9.close()
	

##Do a simple relative flux calibration by smoothing the standard star spectrum,
##masking out Brackett lines, and dividing by the continuum of Vega (file from PLP)
## OLD, but might incorporate features into future flux calibration code
#def simple_telluric_and_flux_calibration(sci, std):
	#num_dimensions = ndim(sci.orders[0].wave) #Store number of dimensions
	#if num_dimensions == 2:
		#slit_pixel_length = len(sci.orders[0].flux[:,0]) #Height of slit in pixels for this target and band
	#for i in xrange(sci.n_orders): #Loop through each order
		#if quality_cut: #Generally we throw out bad pixels, but the user can turn this feature off by setting quality_cut = False
			##goodpix = logical_and(sci.orders[i].flux > -100.0, std.orders[i].flux > .1)  #apply the mask
			#goodpix = std.orders[i].flux > .05
			#badpix = ~goodpix
			#std.orders[i].flux[badpix] = nan
		#if num_dimensions == 2:  #For 2D spectra, expand standard star spectrum from 1D to 2D
			#std.orders[i].flux = tile(std.orders[i].flux, [slit_pixel_length,1]) #Expand standard star spectrum into two dimensions
		#sci.orders[i].flux = sci.orders[i].flux / std.orders[i].flux #Divide science spectrum by standard spectrum
	#sci.orders = combine_orders(sci.orders) #Combine the newly corrected orders into one long spectrum
	#return(sci) #Return the new telluric corrected science spectrum
	
	
##Find lines across all orders and saves it as a line list object
#class find_lines:
	#def __init__(self, sci, delta_v=0.0):
		#for i in xrange(sci.n_orders): #Loop through each order
			
			#line_waves = self.search_order(sci.orders[i].wave, fill_nans(sci.orders[i].flux), axis=0)
			##flux_continuum_subtracted = self.line_continuum_subtract(sci.orders[i].wave, sci.orders[i].flux, line_waves)
			##line_waves = self.search_order(sci.orders[i].wave, flux_continuum_subtracted)
		#wave = line_waves*(delta_v/c)  #Shift a line list by some delta_v given by the user
		#self.label = line_waves.astype('|S8')  #Automatically make simple wavelength labels for the found lines
		#self.wave = wave #Stores (possibly dopper shifted) waves
		#self.lab_wave = line_waves #Save unshifted waves
	##Function finds lines using the 2nd derivitive test and saves them as a line list
	#def search_order(self, wave, flux):
		##plot(wave, flux)
		#fit = UnivariateSpline(wave, flux, s=50.0, k=4) #Fit an interpolated spline
		##for i in range(5):
			##neo_flux = fit(wave)
			##fit = UnivariateSpline(wave, neo_flux, s=50.0, k=4) #Fit an interpolated spline
		#extrema = fit.derivative().roots() #Grabe the roots (where the first derivitive = 0) of the fit, these are the extrema (maxes and mins)
		#second_deriv = fit.derivative(n=2) #Take second derivitive of fit where the extrema are for 2nd derivitive test
		#extrema_sec_deriv = second_deriv(extrema)  #store 2nd derivitives
		#i_maxima = extrema_sec_deriv < 0. #Apply the concavity theorm to find maxima
		#i_minima = extrema_sec_deriv > 0. #Ditto for minima
		#wave_maxima = extrema[i_maxima]
		#flux_maxima = fit(wave_maxima) #Grab flux of the maxima
		#wave_minima = extrema[i_minima]
		#flux_minima = fit(wave_minima)
		#flux_smoothed = fit(wave) #Read in spline smoothed fit for plotting
		##plot(wave, flux_smoothed) #Plot fit
		##plot(wave_maxima, flux_maxima, 'o', color='red') #Plot maxima found that pass the cut
		##plot(wave, spline_obj.derivitive(
		##print 'TEST SMOOTHING CONTINUUM'
		##for i in range(len(extrema)): #Print results
			##print extrema[i], extrema_sec_deriv[i]
		##Now cut out lines that are below a standard deviation cut
		######stddev_flux = std(flux) #Stddeviation of the pixel fluxes
		######maxima_stddev = flux_maxima / stddev_flux
		######good_lines = maxima_stddev > threshold
		#n_maxima = len(wave_maxima)
		#distance_to_nearest_minima = zeros(n_maxima)
		#elevation_check = zeros(n_maxima)
		#dist_for_elevation_check = 0.00006 #um
		##height_for_elevation_check = 0.02*max(flux_smoothed) #fraction of height
		#for i in range(n_maxima):
			##distance_to_nearest_minima[i] = min(abs(wave_maxima[i] - wave_minima))
			#peak_height = flux_maxima[i]
			#left_height = fit(wave_maxima[i] - dist_for_elevation_check)
			#right_height = fit(wave_maxima[i] + dist_for_elevation_check)
			##elevation_check[i] = (peak_height > left_height + height_for_elevation_check) and (peak_height > right_height + height_for_elevation_check)
			#elevation_check[i] = (peak_height > left_height) and (peak_height > right_height)
		##good_lines = (distance_to_nearest_minima > 0.00001) & (elevation_check == True)
		#good_lines = elevation_check == True
		#wave_maxima = wave_maxima[good_lines]
		#flux_maxima = flux_maxima[good_lines]
		##plot(wave_maxima, flux_maxima, 'o', color='blue') #Plot maxima found that pass the cut
		##line_object = found_lines(wave_maxima, flux_maxima)
		##return [wave_maxima, flux_maxima]
		#stop()
		#return wave_maxima
	##Function masks out existing lines, then tries to find lines again
	#def line_continuum_subtract(self, wave, flux, line_waves, line_cut=0.0005):
		#for line_wave in line_waves: #Mask out each emission line found
			#line_mask = abs(wave - line_wave)  < line_cut #Set up mask around an emission line
			#flux[line_mask] = nan #Cut emission line out
		#fit = UnivariateSpline(wave, flux, s=1e4, k=4) #Smooth remaining continuum
		#continuum = fit(wave) #Grab smoothed continuum
		##stop()
		#return flux - continuum #Return flux with continuum subtracted
