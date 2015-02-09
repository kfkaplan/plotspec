#This library will eventually be the ultimate IGRINS emission line viewability/analysis code
#
#start as test_new_plotspec.py

#Import libraries
from astropy.io import fits #Use astropy for processing fits files
from pylab import *  #Always import pylab because we use it for everything
from scipy.interpolate import interp1d, splev, UnivariateSpline #For interpolating
import bottleneck #Library to speed up things
#from scipy.ndimage import zoom #Was used for continuum subtraction at one point, commented out for now
import ds9 #For scripting DS9
import copy #Allow objects to be copied
#from astropy.convolution import convolve, Gaussian1DKernel #, Gaussian2DKernel #For smoothing, not used for now, commented out
from pdb import set_trace as stop #Use stop() for debugging
ion() #Turn on interactive plotting for matplotlib


#Global variables, set after installing plotspec.py
pipeline_path = '/Volumes/IGRINS_data/plp-interpolate/'#Define path to pipeline directory where reduced data is stored
scratch_path = '/Volumes/IGRINS_data/scratch/' #Define path for saving temporary files
data_path = pipeline_path + 'outdata/'
calib_path = pipeline_path + 'calib/primary/'
OH_line_list = 'OH.dat' #Read in OH line list
default_wave_pivot = 0.625 #Scale where overlapping orders (in wavelength space) get stitched (0.0 is blue side, 1.0 is red side, 0.5 is in the middle)
velocity_range = 60.0 # +/- km/s for interpolated velocity grid
velocity_res = 1.0 #Resolution of velocity grid
c = 2.99792458e5 #Speed of light in km/s
block = 300 #Block of pixels used for median smoothing, using iteratively bigger multiples of block
half_block = block / 2 #Half of the block used for running median smoothing



#~~~~~~~~~~~~~~~~~~~~~~~~~Code for modifying spectral data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
	for i in xrange(std.n_orders): #Loop through each order
		if quality_cut: #Generally we throw out bad pixels, but the user can turn this feature off by setting quality_cut = False
			goodpix = std_flattened.orders[i].flux > .05
			badpix = ~goodpix
			std.orders[i].flux[badpix] = nan
		std_continuum =  std.orders[i].flux / std_flattened.orders[i].flux #Get back continuum of standard star by dividing it by it's own telluric correction
		flux_calib = interp_cont_obj(std.orders[i].wave) / std_continuum  #Try a very simple normalization #Try a very simple normalization
		if show_plots:
			plot(std.orders[i].wave, flux_calib*std_continuum)
		if num_dimensions == 2:  #For 2D spectra, expand standard star spectrum from 1D to 2D
			std.orders[i].flux = tile(std.orders[i].flux, [slit_pixel_length,1]) #Expand standard star spectrum into two dimensions
		sci.orders[i].flux = sci.orders[i].flux * flux_calib/ (std_flattened.orders[i].flux)  #Apply telluric correction and flux calibration
	if show_plots: #Plot Vega spectrum as well for comparison if user wants to see the flux calibration
		plot(vega_wave, vega_flux)
		show()
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
		min_wave  = bottleneck.nanmin(wave_pixels) #Minimum wavelength
		max_wave = bottleneck.nanmax(wave_pixels) #maximum wavelength
		#wave_interp = interp1d(x, wave_pixels, kind = 'linear') #Interpolation for inputting pixel x and getting back wavelength
		x_interp = interp1d(wave_pixels, x, kind = 'linear') #Interpolation for inputting wavlength and getting back pixel x
		interp_velocity = arange(-velocity_range, velocity_range, velocity_res) #Velocity grid to interpolate each line onto
		show_lines = line_list.parse(min_wave, max_wave) #Only grab lines withen the wavelength range of the current order
		flux = [] #Set up list of arrays to store 1D fluxes
		for line_wave in show_lines.wave: #Label the lines
			#x_center = round(x_interp(line_wave)) #Find nearest pixel to center of line
			pv_velocity = c * ( (spec2d.wave - line_wave) /  line_wave ) #Calculate velocity offset for each pixel from c*delta_wave / wave
			pixel_cut = abs(pv_velocity[0]) <= velocity_range #Find only pixels in the velocity range, this is for conserving flux
			interp_obj_1d = interp1d(pv_velocity[0][pixel_cut], spec1d.flux[pixel_cut], kind='slinear', bounds_error=False)
			interp_obj_2d = interp1d(pv_velocity[0][pixel_cut], spec2d.flux[:,pixel_cut], kind='slinear', bounds_error=False)
			ungridded_result_1d = interp_obj_1d(pv_velocity[0][pixel_cut]) #PV diagram ungridded on origional pixels
			ungridded_result_2d = interp_obj_2d(pv_velocity[0][pixel_cut]) #PV diagram ungridded on origional pixels
			gridded_result_1d = interp_obj_1d(interp_velocity) #PV diagram velocity gridded
			gridded_result_2d = interp_obj_2d(interp_velocity) #PV diagram velocity gridded
			if not s2n: #Check that 1D data is not an array of S/N and actually is flux
				scale_flux_1d = bottleneck.nansum(ungridded_result_1d) / bottleneck.nansum(gridded_result_1d) #Scale interpolated flux to original flux so that flux is conserved post-interpolation
			else: #Or scale signal to noise per pixel to S/N per resolution element (for comparing to the ETC, for example)
				scale_flux_1d = sqrt(3.3) 
			scale_flux_2d = bottleneck.nansum(ungridded_result_2d) / bottleneck.nansum(gridded_result_2d) #Scale interpolated flux to original flux so that flux is conserved post-interpolation
			flux.append(gridded_result_1d *  scale_flux_1d) #Append 1D flux array with line
			if 'pv' not in locals(): #First line start datacube for 2D spectrum
				pv = gridded_result_2d.transpose() * scale_flux_2d #Start datacube for 2D PV spectra
			else: #For all other lines, add 2D PV spectra to the existing datacube
				pv = dstack([pv, gridded_result_2d.transpose() * scale_flux_2d]) #Stack PV spectrum of lines into a datacube
		pv = swapaxes(pv, 0, 2) #Flip axes around in cube so that it can later be saved as a fits file
		self.flux = flux #Save 1D PV fluxes
		self.pv = pv #Save datacube of stack of 2D PV diagrams for each line
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
		ds9.set('zoom to fit') #Zoom PV diagram to fit ds9 window
		ds9.set('zoom 0.9') #Zoom out a little bit to see the coordinate grid
		ds9.set('scale log') #Set view to log scale
		ds9.set('scale ZMax') #Set scale limits to Zmax, looks okay
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
	def plot_1d_velocity(self, line_index, title=''): #Plot 1D spectrum in velocity space (corrisponding to a PV Diagram), called when viewing a line
		clf() #Clear plot space
		max_flux = bottleneck.nanmax(self.flux[line_index], axis=0) #Find maximum flux in slice of spectrum
		plot(self.velocity, self.flux[line_index], color='black') #Plot 1D spectrum slice
		plot([0,0], [0,2*max_flux], '--') #Plot velocity zero point
		xlim([-velocity_range, velocity_range]) #Set xrange to be +/- the velocity range set for the PV diagrams
		ylim([0, max_flux]) #Set yrange
		if title != '': #Add title to plot showing line name, wavelength, etc.
			suptitle(title, fontsize=20)
		#if label != '' and wave > 0.0:
			#title(label + ' ' + "%12.5f" % wave + '$\mu$m')
		#elif label != '':
			#title(label)
		#elif wave > 0.0:
			#title("%12.5f" % wave + '$\mu$m')
		xlabel('Velocity [km s$^{-1}$]', fontsize=18) #Label x axis
		if self.s2n:
			ylabel('S/N per resolution element (~3.3 pixels)', fontsize=18) #Label y axis as S/N for S/N spectrum
		else:
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
	def getline(self, line): #Grabs PV diagram for a single line given a line label
		i =  where(self.label == line)[0][0] #Search for line by label
		return self.pv[i] #Return line found
	def ratio(self, numerator, denominator):  #Returns PV diagram of a line ratio
		return self.getline(numerator) / self.getline(denominator)
	def normalize(self, line): #Normalize all PV diagrams by a single line
		self.pv = self.pv / self.getline(line)
	def basic_flux(self, x_range, y_range):
		sum_along_x = bottleneck.nansum(self.pv[:, y_range[0]:y_range[1], x_range[0]:x_range[1]], axis=2) #Collapse along velocity space
		total_sum = bottleneck.nansum(sum_along_x, axis=1) #Collapse along slit space
		return(total_sum) #Return the integrated flux found for each line in the box defined by the user

#~~~~~~~~~~~~~~~~~~~~~~~~~Code for reading in analyzing spectral data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Wrapper for easily creating a 1D or 2D comprehensive spectrum object of any type, allowing user to import an entire specturm object in one line
def makespec(date, band, waveno, frameno, std=False, twodim=False, s2n=False):
	spec_data = fits_file(date, frameno, band, std=std, twodim=twodim, s2n=s2n) #Read in data from spectrum
	if twodim: #If spectrum is 2D
		spec_obj = spec2d(spec_data) #Create 2D spectrum object
	else: #If spectrum is 1D
		wave_data = fits_file(date, waveno, band, wave=True) #If 1D, read in data from wavelength solution
		spec_obj = spec1d(wave_data, spec_data) #Create 1D spectrum object
	return(spec_obj) #Return the fresh spectrum object!
	
	

#Class stores information about a fits file that has been reduced by the PLP
class fits_file:
	def __init__(self, date, frameno, band, std=False, wave=False, twodim=False, s2n=False):
		self.date = '%.4d' % int(date) #Store date of observation
		self.frameno =  '%.4d' % int(frameno) #Store first frame number of observation
		self.band = band #Store band name 'H' or 'K'
		self.std = std #Store if file is a standard star
		self.wave = wave #Store if file is a wavelength solution
		self.s2n = s2n #Store if file is the S/N spectrum
		self.twodim = twodim #Store if file is of a 2D spectrum instead of a s spectrum
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
	def __init__(self, fits_wave, fits_spec):
		wavedata = fits_wave.get() #Grab fits data for wavelength out of object
		specdata = fits_spec.get() #Grab fits data for flux out of object
		orders = [] #Set up empty list for storing each orders
		n_orders = len(specdata[0].data[:,0]) #Count number of orders in spectrum
		wavedata = wavedata[0].data.byteswap().newbyteorder() #Read out wavelength and flux data from fits files into simpler variables
		fluxdata = specdata[0].data.byteswap().newbyteorder() #Read out wavelength and flux data from fits files into simpler variables
		for i in xrange(n_orders): #Loop through to process each order seperately
			orders.append( spectrum(wavedata[i,:], fluxdata[i,:])  ) #Append order to order list
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
			[low_wave_limit, high_wave_limit]  = [bottleneck.nanmin(combospec.wave), bottleneck.nanmax(self.orders[i+1].wave)] #Find the wavelength of the edges of the already stitched orders and the order currently being stitched to the rest 
			wave_cut = low_wave_limit + wave_pivot*(high_wave_limit-low_wave_limit) #Find wavelength between stitched orders and order to stitch to be the cut where they are combined, with pivot set by global var wave_pivot
			goodpix_combospec = combospec.wave >= wave_cut #Find pixels in already stitched orders to the left of where the next order will be cut and stitched to
			goodpix_next_order = self.orders[i+1].wave < wave_cut #Find pixels to the right of the where the order will be cut and stitched to the rest
			combospec.wave = concatenate([self.orders[i+1].wave[goodpix_next_order], combospec.wave[goodpix_combospec] ]) #Stitch wavelength arrays together
			combospec.flux = concatenate([self.orders[i+1].flux[goodpix_next_order], combospec.flux[goodpix_combospec] ]) #Stitch flux arrays together
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
		max_flux = bottleneck.nanmax(self.combospec.flux, axis=0)
		total_wave_coverage = max_wave - min_wave #Calculate total wavelength coverage
		if (model != '') and (model != 'none'): #Load model for comparison if needed
			model_wave, model_flux = loadtxt(model, unpack=True) #Read in text file of model with format of two columns with wave <tab> flux
			model_max_flux = bottleneck.nanmax(model_flux[logical_and(model_wave > min_wave, model_wave < max_wave)], axis=0) #find tallest line in model
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
				if sub_linelist.flux[i] > max_flux*threshold and bottleneck.nanmax(sub_linelist.flux[other_lines], axis=0) == sub_linelist.flux[i]: #if line is the highest of all surrounding lines within some window
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



#Class to store and analyze a 2D spectrum
class spec2d:
	def __init__(self, fits_spec):
		spec2d = fits_spec.get() #grab all fits data
		n_orders = len(spec2d[1].data[:,0]) #Calculate number of orders to use  
		slit_pixel_length = len(spec2d[0].data[0,:,:]) #Height of slit in pixels for this target and band
		orders = [] #Set up empty list for storing each orders
		for i in xrange(n_orders):
			wave1d = spec2d[1].data[i,:].byteswap().newbyteorder() #Grab wavelength calibration for current order
			data2d = spec2d[0].data[i,:,:].byteswap().newbyteorder() #Grab 2D Spectrum of current order
			wave2d = tile(wave1d, [slit_pixel_length,1]) #Create a 2D array storing the wavelength solution, to be appended below the data
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
			trace = bottleneck.nanmedian(old_flux, axis=1) #Get trace of continuum from median of whole order
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
			[low_wave_limit, high_wave_limit]  = [bottleneck.nanmin(combospec.wave), bottleneck.nanmax(self.orders[i+1].wave)] #Find the wavelength of the edges of the already stitched orders and the order currently being stitched to the rest 
			wave_cut = low_wave_limit + wave_pivot*(high_wave_limit-low_wave_limit) #Find wavelength between stitched orders and order to stitch to be the cut where they are combined, with pivot set by global var wave_pivot
			goodpix_combospec = combospec.wave[0,:] >= wave_cut #Find pixels in already stitched orders to the left of where the next order will be cut and stitched to
			goodpix_next_order = self.orders[i+1].wave[0,:] < wave_cut #Find pixels to the right of the where the order will be cut and stitched to the rest
			combospec.wave = concatenate([self.orders[i+1].wave[:, goodpix_next_order], combospec.wave[:, goodpix_combospec] ], axis=1) #Stitch wavelength arrays together
			combospec.flux = concatenate([self.orders[i+1].flux[:, goodpix_next_order], combospec.flux[:, goodpix_combospec] ], axis=1)#Stitch flux arrays together
		self.combospec = combospec #save the orders all stitched together
	#Simple function for displaying the combined 2D spectrum
	def plot(self, spec_lines, pause = False, close = False):
		if not self.combospec in vars(): #Check if a combined spectrum exists
			print 'No spectrum of combined orders found.  Createing combined spectrum.'
			self.combine_orders() #If combined spectrum does not exist, combine the orders
		wave_fits = fits.PrimaryHDU(self.combospec.wave)    #Create fits file containers
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
		ds9.set('scale ZMax') #Set scale limits to Zmax, looks okay
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
		min_wave  = bottleneck.nanmin(wave_pixels, axis=0) #Minimum wavelength
		max_wave = bottleneck.nanmax(wave_pixels, axis=0) #maximum wavelength
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
	def __init__(self, wave, flux): #Initialize spectrum by reading in two columned file 
		self.wave = wave #Set up wavelength array 
		self.flux = flux #Set up flux array

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
		found_lines = (subset.wave > min_wave) & (subset.wave < max_wave) #Grab location of lines only in the wavelength range
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
			median_result[i] = bottleneck.nanmedian(flux[:,x_left[i]:x_right[i]]) #Calculate median between x_left and x_right for a given pixel
	else: #Run this loop for 1D
		for i in xrange(nx): #This loop does the running of the median down the spectrum each pixel
			median_result[i] = bottleneck.nanmedian(flux[x_left[i]:x_right[i]])  #Calculate median between x_left and x_right for a given pixel
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
