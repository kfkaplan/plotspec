#Library for making and processing the IGRINS datacube
from pylab import *
from bottleneck import *
from scipy.ndimage import zoom #For resizing images
from scipy.interpolate import griddata
from astropy.io import fits #Use astropy for processing fits files
#from astropy.convolution import convolve, Gaussian2DKernel
from plotspec import * #Import plotspec library
import h2 #Import H2 library
from numpy import round

#~~~~~~~~~~~~~~~MODIFY THESE PARAMETERS FOR IGRINS DATA~~~~~~~~~~~~~~~~
n_pointings = 6 #Number of slit positions used in scan
save.name('Datacube Name') #Name to save results as
date = 20141125 #Date (directory name) of observations
stdno = 110 #Standard star frame no.
waveno = 120 #Sky frame no. for using OH lines to corrected wave solution in PLP
spectral_lines = lines('H2_datacube.dat', delta_v = 8.0) #Spectral lines to use
B = 7.412 #B and V magnitudes of the std star HD 34317
V = 7.410
a0v_HI_y_scale = 1.0 #Scale HI line depth for A0V star
a0v_HI_gauss_smooth = 1.0 #Gaussian smooth HI lines for A0V star ()
framenos = [124,120,121,122,126,134] #Frame numbers for pointings for map going from SW -> NE
exposure = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #Relative exposure times for each frame, default = 1.0 but can be set to another value to equalize across whole datacube
xshift = array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) #Placement of slit in arcsecond space (1 arsec = 4 pixels) perpendicular to the slit (typically 1'' apart)
yshift = array([0.0, 0.0, 0.0, 0.0 ,0.0 ,0.0]) #Placement of slit in arcsecond space (1 arsec = 4 pixels) parallel to the slit
#~~~~~~~~~~~~~~~MODIFY THESE PARAMETERS FOR ASTROMETRY~~~~~~~~~~~~~~~~
ra = 83.8319	#RA of reference pixel in decimal degrees
dec = -5.4245	#Dec of reference pixel in decimal degrees
x_reference_pixel = 3.0 #Set reference pixel perpendicular to the slit (x-axis), NOTE: these are pixel coordinates NOT ARCSECONDS!
y_reference_pixel = 30.0 #Set reference pixel along slit length (y-axis)
pa = 135.0 #Position Angle of slit on the sky (East of north) in degrees
#~~~~~~~~~~~~~ONLY MODIFY THESE PARAMETERS IF YOU KNOW WHAT YOU ARE DOING~~~~~~~~~~~~~~~~
plate_scale = 0.25 #Plate scale in arcseconds per pixel
x_expand = 1.0/plate_scale #Number of times to expand along the x axis perpendicular to the slit, this  make the x and y plat
#Shape of final 4D array in the form of (# of spectral lines, pixels along slit,velocity, pixels prependicular to  slit, ), x_expand expands along the x-axis (pointings) to make place scale same in x & y
cube_size = [len(spectral_lines.label), slit_length + 5, 200, n_pointings*x_expand + 4] 


#~~~~~~~~~~~~~~~~~~~~FITS HEADER INFORMATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save(line_cube, fname): #Save datacube for an extracted cube for a spectral line as a fits file, for 3D datacubes add header info about velocity information
	n_dimensions = ndim(line_cube) #Calculate number of dimensions
	cube_fits = fits.PrimaryHDU(line_cube) #Open astropy fits object
	cube_fits.header['RADECSYS'] = 'FK5' #Define coord system
	cube_fits.header['EQUINOX'] = 2000 #Define equinox, here set to J2000
	cube_fits.header['CTYPE1'] = 'RA--TAN' #Set unit to declination
	cube_fits.header['CTYPE2'] = 'DEC--TAN' #Set unit for slit length to something generic
	if n_dimensions == 3: #If velocity information exists.... else fits file saved will be a 2D image
		cube_fits.header['CTYPE3'] = 'km/s' #Set unit to "Optical velocity" (I know it's really NIR but whatever...)
	cube_fits.header['CRVAL1'] = ra	 #RA of reference pixel
	cube_fits.header['CRVAL2'] = dec #Dec of reference pixel
	if n_dimensions == 3: #If velocity information exists.... else fits file saved will be a 2D image
		cube_fits.header['CRVAL3'] = 0.0 #Velocity of reference pixel
	cube_fits.header['CRPIX1'] = x_reference_pixel #Set zero point along the x-axis (pointings)
	cube_fits.header['CRPIX2'] = y_reference_pixel #Set zero point to 0 pixel for slit length
	if n_dimensions == 3: #If velocity information exists.... else fits file saved will be a 2D image
		cube_fits.header['CRPIX3'] = (velocity_range / velocity_res) + 1 #Set zero point to where v=0 km/s (middle of stamp)
		cube_fits.header['CDELT3'] = velocity_res #Set zero point to where v=0 km/s (middle of stamp)
	fits_plate_scale = plate_scale*(1./3600.) #Set plate scale to 0.25"
	angle =  (pa + 270.0) * pi / 180.0 #Rotation angle converted from degrees into radians
	x_reflect = -1.0 #Set to reflect across x axis
	cube_fits.header['CD1_1'] = -fits_plate_scale*cos(angle) * x_reflect #cd[0,0] #Save rotation and plate scale transformation matrix into fits file header
	cube_fits.header['CD1_2'] = -fits_plate_scale*sin(angle) #cd[1,0]
	cube_fits.header['CD2_1'] = -fits_plate_scale*sin(angle) * x_reflect #cd[0,1]
	cube_fits.header['CD2_2'] = fits_plate_scale*cos(angle) #cd[1,1]
	cube_fits.writeto(fname, clobber  = True) #Save fits file



#Class that stores orion bar cube bar data
class data():
	def __init__(self): #Initialize class and create datacube in memory
		master_cube = zeros(cube_size)  #Initialize array to store flux in a datacube
		master_cube_var = zeros(cube_size) #Initialize array to store
		master_cube_overlap = zeros(cube_size) #Initialize array to store how many pointings overlap with a given pixel
		xshift_pix = round(xshift/plate_scale).astype(int) #Find starting x position in pixels for each pionting
		yshift_pix = round(yshift/plate_scale).astype(int) #Find starting y position in pixels for each pionting
		for i in xrange(n_pointings): #Loop through each pointing along the scan of the Orion Bar
			spec1d, spec2d = getspec(date, waveno, framenos[i], stdno, B=B, V=V, y_scale=a0v_HI_y_scale, wave_smooth=a0v_HI_gauss_smooth) #Create 1D and 2D spectra objects for all orders combining both H and K bands (easy eh?)
			spec2d.subtract_continuum() #Subtract continuum from 2D spectrum, comment out to not subtract continuum
			spec1d.combine_orders() #Combine all orders in 1D spectrum into one very long spectrum
			spec2d.combine_orders() #Combine all orders in 2D spectrum into one very long spectrum
			pv = position_velocity(spec1d.combospec, spec2d.combospec, spectral_lines, shift_lines=v_shift_file) #Extract and create a datacube in position-velocity space of all lines in line list(s) found in spectrum
			nans = ~isfinite(pv.pv) #Zero out nans before stacking
			pv.pv[nans] = 0.
			pv.var2d[nans] = 0.
			if exposure[i] != 1.0: #If exposure times vary
				pv.pv = pv.pv / exposure[i]  #Scale flux
				pv.var2d = pv.var2d / (exposure[i]**2) #Scale variance (to propogate uncertainity)
			flux = repeat(pv.pv[:,:,:,newaxis],4,3) #Set up slice of RA, Dec. grid to put flux into
			var = repeat(pv.var2d[:,:,:,newaxis],4,3)  #Set up slice of RA, Dec. grid to put variance into
			x_start = xshift_pix[i] #Set up beginning and ending pixels to paint 
			x_end = xshift_pix[i] + x_expand
			y_start = yshift_pix[i]
			y_end = yshift_pix[i] + slit_length
			master_cube[:,y_start:y_end,:,x_start:x_end] = master_cube[:,y_start:y_end,:,x_start:x_end] + flux #Load slit into master data cube, normalize flux by expnasion done along x-axis
			master_cube_var[:,y_start:y_end,:,x_start:x_end] = master_cube_var[:,y_start:y_end,:,x_start:x_end] + var  #Load slit into master variance cube
			used_pixels = flux != 0.0 #Find pixels where we just placed data on the cube grid
			master_cube_overlap[:,y_start:y_end,:,x_start:x_end][used_pixels] = master_cube_overlap[:,y_start:y_end,:,x_start:x_end][used_pixels] + 1.0 #Increment the counter for how many pointings these pixels used
		master_cube = master_cube / master_cube_overlap #Scale overlapping pixels down by the number of pointings overlapping a given pixel
		master_cube_var = master_cube_var / master_cube_overlap**2  #Also for the variance: scale overlapping pixels down by the number of pointings overlapping a given pixel
		empty_pixels = master_cube == 0.0 #Blank out empty pixels with nans
		master_cube[empty_pixels] = nan
		master_cube_var[empty_pixels] = nan
		self.cube = master_cube / float(x_expand) #store full datacube in bar object, and normalize flux to number of pointings (divide by factor we expand x axis by)
		self.var = master_cube_var / float(x_expand)**2 #Store full variance datacube in bar object, and normalize uncertainity to number of pointings
		self.label = spectral_lines.label #Save labels of spectral lines so that lines can be later retrieved by matching their label strings to what the user wants
		self.velocity = pv.velocity #Save velocity per pixel along velocity axis for later being able to cut and collapse parts of the datacubes
	def getline(self, input_label, variance=False): #Grab a datacube for a chosen spectral line
		chosen_line = where(self.label == input_label)[0][0] #Select which spectral line to extract from string inputted by user to match the label of the line
		if variance: #If user specifies they want the variance...
			grab_line_cube = swapaxes(self.var[chosen_line,:,:,:],0,1) #Extract datacube for that line
		else: #Else grab the flux cube (default)
			grab_line_cube = swapaxes(self.cube[chosen_line,:,:,:],0,1) #Extract datacube for that line
		return(grab_line_cube) #Return the extracted cube 
	def getimage(self, input_label, vrange=velocity_range, variance=False): #Get a line and collapse into 2D, can specifiy a velocity range to reduce noise
		line_cube = self.getline(input_label, variance=variance) #Grab line cube
		velocity_cut = (self.velocity > vrange[0]) & (self.velocity < vrange[1]) #Make a cut in velocity space, if the user so specifies, otherwise use all +/- 100 km/s in the cube
		collapsed_cube = nansum(line_cube[velocity_cut,:,:], 0) #Collapse cube along velocity access into an image
		collapsed_cube[collapsed_cube == 0.] = nan #Blank out unused pixels with nan
		return collapsed_cube
	def saveimage(self, input_label, fname, variance=False, vrange=velocity_range): #Save a 2D collapsed image
		img = self.getimage(input_label, vrange=vrange, variance=variance)
		save(img, fname) #save image
	def savecube(self, input_label, fname, variance=False, collapse=False): #Extract and save a spectral line cube as a fits file
		line_cube = self.getline(input_label, variance=variance) #Grab line cube
		save(line_cube, fname) #Save linecube as a fits file
	def saveratio(self, input_label_1, input_label_2, vrange=velocity_range, s2n_cut=0.0, fname=''): #Save image of a ratio of two lines
		flux1 = self.getimage(input_label_1, vrange=vrange) #Grab collapsed images of both lines
		flux2 = self.getimage(input_label_2, vrange=vrange)
		sig1 = sqrt(self.getimage(input_label_1, vrange=vrange, variance=True))#Grab 1-sigma uncertainity for each image
		sig2 = sqrt(self.getimage(input_label_2, vrange=vrange, variance=True))
		ratio = flux1 / flux2 #Take ratio of line 1 / line 2
		sigma = ratio * sqrt((sig1/flux1)**2 + (sig2/flux2)**2) #Uncertainity in ratio
		s2n = ratio/sigma #Calculate S/N on ratio
		if s2n_cut != 0.0: #If user specifies a S/N cut, mask regions with lower S/N
			low_s2n_regions = s2n < 3.0 #Find regions with low S/N
			ratio[low_s2n_regions] = 0. #Set regions with low S/N to zero and ignore them
		if fname =='': #If file name is not specified by user
			fname = 'ratio_'+input_label_1+'_over_'+input_label_2+'.fits' #automatically create own filename
		save(ratio, fname) #Save ratio map
	def extract_half(self, vrange=velocity_range, dim=False, use_line='1-0 S(1)', median_multiplier=1.0): #Extract flux from above some fraction of flux for the specified line
		h = h2.make_line_list() #Set up object for storing H2 transitions
		img =  self.getimage(use_line, variance=False, vrange=vrange)
		median_flux = median(img)
		if dim == False: #Use bright regions or...
			use_region = img >= median_flux * median_multiplier
		else: #Use dim regions
			use_region = img <= median_flux * median_multiplier
		for label in self.label: #Loop through each line and save the various results into an h2_transitions object
			img = self.getimage(label, vrange=vrange)
			var = self.getimage(label, vrange=vrange, variance=True)
			total_flux = nansum(img[use_region]) 
			total_var = nansum(var[use_region])
			total_sigma = sqrt(total_var)
			find_line = h.label == label
			h.F[find_line] = total_flux
			h.sigma[find_line] = total_sigma
			h.s2n[find_line] = total_flux / total_sigma
		return(h)
	def extract_full(self, vrange=velocity_range): #Extract full datacube to get H2 line fluxes
		h = self.extract_half(vrange=vrange, dim=False, median_multiplier=0.0) #Pretend we aqre extracting a bright or dim region, but extract everything by setting threshold to 0.
		return(h) #Return the result
	#Function fills any gaps 
	def fill_gaps(self, size=1):
		cube = copy.deepcopy(self.cube)
		var = copy.deepcopy(self.var)
		n_lines, ny, n_velocity, nx = shape(cube) #Calculate pixel sizes
		filled_slice = zeros([ny,nx]) #Create array that will store the smoothed median spectrum
		filled_slice_var = zeros([ny,nx])
		x_left = arange(nx) - size #Create array to store left side of running median
		x_right = arange(nx) + size + 1 #Create array to store right side of running median
		x_left[x_left < 0] = 0 #Set pixels beyond edge of order to be nonexistant
		x_right[x_right > nx] = nx - 1 #Set pixels beyond right edge of order to be nonexistant
		x_size = x_right - x_left #Calculate number of pixels in the x (wavelength) direction
		#g = Gaussian2DKernel(stddev=5.0, x_size=3, y_size=3) #Set up convolution kernel
		for i in xrange(n_lines): #Loop through every spectral line
			for j in xrange(n_velocity): #Loop through every velocity slice
				for k in xrange(nx): #This loop does the running of the median down the spectrum each pixel
					filled_slice[:,k] = nanmedian(cube[i,:,j,x_left[k]:x_right[k]], axis=1) #Calculate median between x_left and x_right for a given pixel
					filled_slice_var[:,k] = nanmedian(var[i,:,j,x_left[k]:x_right[k]], axis=1) #Calculate median between x_left and x_right for a given pixel
				nans_found = ~isfinite(cube[i,:,j,:]) #Find nans
				cube[i,:,j,:][nans_found] = filled_slice[nans_found] #Fill in the gaps
				var[i,:,j,:][nans_found]  = filled_slice_var[nans_found] #Fill in the gaps
		self.cube = cube #Update datacube
		self.var = var #Update datacube variance









