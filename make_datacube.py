#Library for making and processing the IGRINS datacube
from pylab import *
from scipy.ndimage import zoom, binary_closing #For resizing images
from astropy.io import fits, ascii #Use astropy for processing fits files, ASCII text files
from astropy.convolution import interpolate_replace_nans
#from astropy.convolution import convolve, Gaussian2DKernel
from plotspec import * #Import plotspec library
import h2 #Import H2 library
from numpy import round
import gc #Load in python's garbage collector
from bottleneck import *


#~~~~~~~~~~~~~~~MODIFY THESE PARAMETERS FOR IGRINS DATA~~~~~~~~~~~~~~~~
save.name('Datacube Name') #Name to save results as
#~~~~~~~~~~~~~~~~~~~~~READ INPUT FILE~~~~~~~~~~~~~~~~~~~~~~~~~~~
input_file = 'demo_datacube_input.dat'
input = ascii.read(input_file)
n_pointings = input['Date'].size
#~~~~~~~~~~~~~~~MODIFY THESE PARAMETERS FOR ASTROMETRY~~~~~~~~~~~~~~~~
velocity_range = 100.0 #Set range of velocity axis in datacube
velocity_res = 1.0 #Set size of pixel in velocity space
ra = 83.83205 #RA of center of reference slit pointing in decimal degrees
dec = -5.42417 #Dec of center of reference slit pointing in decimal degrees
reference_pointing_date = 20141125  #Night of pointing where the center of the slit is used to determine the RA and Dec. for the astrometry
reference_pointing_frameno = 120  #Frame number of pointing where center of the slit is used to determine the RA and Dec for the astrometry
flux_calibrate = True #Turn on or off relative flux calibration
subtract_continuum = False #Turn on or off continuum subtraction
use_blocks = False #Use blocks for flux calibration (set in input file)?  If not using blocks, turn it off here to speed up datacube building
fill_nans = True #Turn on or off filling nans
save_checks = True #Save pdfs of the output of each pointing (used for checking flux calibration, ect.)
flux_calibration_line = '1-0 S(1)' #Name of emission line to use for flux calibration, should be something bright!
flux_calibration_velocity_range = 10.0 #Range of velocities to collapse for flux calibration in +/- km/s
flux_calibration_s2n_thresh = 3.0 #Threshold for pixel S/N to be used for flux calibration
length_of_slit_on_sky = 15.0 #Length of the slit on the sky, in arcseconds, depends on the telescope IGRINS was on, here it is set for McDonald


#~~~~~~~~~~~~~ONLY MODIFY THESE PARAMETERS IF YOU KNOW WHAT YOU ARE DOING~~~~~~~~~~~~~~~~
plate_scale = length_of_slit_on_sky / float(slit_length) #Calculate slit length by dividing how many arcseconds the slit is on the sky by the number of pixels the slit is in the data
x_expand = int((length_of_slit_on_sky/15.0)/plate_scale) #Number of times to expand along the x axis perpendicular to the slit, this  make the x and y plate scale match
#Shape of final 4D array in the form of (# of spectral lines, pixels along slit,velocity, pixels prependicular to  slit, ), x_expand expands along the x-axis (pointings) to make place scale same in x & y



#~~~~~~~~~~~~~~~~~~~~Main Library~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
	global pa_reference_pixel, x_reference_pixel, y_reference_pixel #Since the reference pixes are set as global variables, set to access them globally here
	cube_fits.header['CRPIX1'] = x_reference_pixel #Set reference pixel along the x-axis
	cube_fits.header['CRPIX2'] = y_reference_pixel #Set reference pixel along the y-axis
	if n_dimensions == 3: #If velocity information exists.... else fits file saved will be a 2D image
		cube_fits.header['CRPIX3'] = (velocity_range / velocity_res) + 1 #Set zero point to where v=0 km/s (middle of stamp)
		cube_fits.header['CDELT3'] = velocity_res #Set zero point to where v=0 km/s (middle of stamp)
	fits_plate_scale = plate_scale*(1./3600.) #Set plate scale to 0.25"
	angle = pa_reference_pixel * pi/180.0 #Set angle with the PA of the reference pixel
	cube_fits.header['CD1_1'] = fits_plate_scale*cos(angle)  #cd[0,0] #Save rotation and plate scale transformation matrix into fits file header
	cube_fits.header['CD1_2'] = fits_plate_scale*sin(angle) #cd[1,0]
	cube_fits.header['CD2_1'] = -fits_plate_scale*sin(angle) #cd[0,1]
	cube_fits.header['CD2_2'] = fits_plate_scale*cos(angle) #cd[1,1]
	cube_fits.writeto(fname, overwrite = True) #Save fits file

class cube_geometry(): #Store all geometry based on reading in all the rotations and shifts for each slit
	def __init__(self, input_PAs, input_x_shifts, input_y_shifts):
		global pa_reference_pixel, x_reference_pixel, y_reference_pixel #Since the reference pixes are set as global variables, set to access them globally here
		n = len(input_PAs) #Number of slits to store
		PA_difference = (input_PAs - pa_reference_pixel) % 360.0 #Store difference in all PAs from the first frame
		x_start = zeros(n) #Store starting pixels coords in x for a given position
		x_end = zeros(n) #Store ending pixels coords in x  for a given position
		y_start = zeros(n) #Store starting pixels coords in y for a given position
		y_end = zeros(n) #Store ending pixels coords in y for a given position
		if x_expand % 2 == 1: #If x_expand is odd
			negative_width_shift = -(x_expand-1) / 2  #Calculate position of the ends of the slit width
			positive_width_shift = ((x_expand-1)/2) +1
		else: #If xexpand is even
			negative_width_shift = -x_expand / 2  #Calculate position of the ends of the slit width
			positive_width_shift = x_expand / 2 
		negative_length_shift = -slit_length / 2
		if slit_length % 2 == 1: #If slit length is odd
			positive_length_shift = (slit_length / 2) #Add one to end of slit to acommidate the odd value of pixels in the slit length direction
		else: #If slit length is even
			positive_length_shift = (slit_length / 2) #Just divide an even numbered slit length by two, simple eh?
		for i in xrange(n): #Loop through each slit
			if PA_difference[i] == 0.0 or PA_difference[i] == 180.0: #or PA_difference[i] == 180.0: #If PA of this pointing is same PA as the first pointing
				x_start[i] = input_x_shifts[i] + negative_width_shift #Set up coordinates for this slit to be painted into the datacube
				x_end[i] = input_x_shifts[i] + positive_width_shift 
				y_start[i] = input_y_shifts[i] + negative_length_shift
				y_end[i] = input_y_shifts[i] + positive_length_shift
			elif PA_difference[i] == 90.0 or PA_difference[i] == 270.0: #or  PA_difference[i] == 270.0: #If PA rotated 90 degrees from first pointing
				x_start[i] = input_x_shifts[i] + negative_length_shift #Set up coordinates for this slit to be painted into the datacube
				x_end[i] = input_x_shifts[i] + positive_length_shift
				y_start[i] = input_y_shifts[i] + negative_width_shift
				y_end[i] = input_y_shifts[i] + positive_width_shift
			# elif PA_difference[i] == 180.0: #If PA rotated 180 degrees from first pointing
			# 	x_start[i] = input_x_shifts[i] + negative_width_shift #Set up coordinates for this slit to be painted into the datacube
			# 	x_end[i] = input_x_shifts[i] + positive_width_shift
			# 	y_start[i] = input_y_shifts[i] + negative_length_shift
			# 	y_end[i] = input_y_shifts[i] + positive_length_shift
			# elif PA_difference[i] == 270.0: #If PA rotated 270 degrees from first pointing
			# 	x_start[i] = input_x_shifts[i] + negative_length_shift#Set up coordinates for this slit to be painted into the datacube
			# 	x_end[i] =  input_x_shifts[i] + positive_length_shift
			# 	y_start[i] = input_y_shifts[i] + negative_width_shift
			# 	y_end[i] =  input_y_shifts[i] + positive_width_shift
			else: #catch error of improper PA inputs
				print "WARNING: PA must be in 90 degree incriments from first "
				stop()
		all_x_start_and_end_points = concatenate([x_start, x_end]) #Cram all x and y start and ending variables into single arrays
		all_y_start_and_end_points = concatenate([y_start, y_end])
		x_size =  max(all_x_start_and_end_points) - min(all_x_start_and_end_points)   #Calculate x,y dimensions of datacube
		y_size =  max(all_y_start_and_end_points) - min(all_y_start_and_end_points)
		cube_x_start = min(x_start) #Find the starting corner of the cube
		cube_y_start = min(y_start)
	 	x_start, x_end = [x_start - cube_x_start, x_end - cube_x_start] #Shift pixels in x direction to make left-most pixel 0
	 	y_start, y_end = [y_start - cube_y_start, y_end - cube_y_start] #Shift pixels in y direction to make left-most pixel 0
	 	min_x_start = min(x_start) #Find the minimum x and y starting pixels to search for negative numbers
	 	min_y_start = min(y_start)
	 	min_x_end = min(x_end)
	 	min_y_end = min(y_end)
	 	if min_x_start < 0: #If any negative numbers are found, shift pixel starting positions to make all numbers positive
	 		x_start = x_start - min_x_start
	 		x_end = x_end - min_x_start
	 	if min_y_start < 0:
	 		y_start = y_start - min_y_start
	 		y_end = y_end - min_y_start
	 	if min_x_end < 0:
	 		x_start = x_start - min_x_end
	 		x_end = x_end - min_x_end
	 	if min_y_end < 0:
	 		y_start = y_start - min_y_end
	 		y_end = y_end - min_y_end
	 	self.x_start = x_start #Store pixels shifts
	 	self.y_start = y_start
	 	self.x_end = x_end
	 	self.y_end = y_end
	 	self.PA_difference = PA_difference #Store PA differences
	 	self.x_size  = x_size #Store cube sizes
	 	self.y_size = y_size




#Class that stores orion bar cube bar data
class data():
	def __init__(self): #Initialize class and create datacube in memory
		global x_reference_pixel, y_reference_pixel, pa_reference_pixel #Store reference pixels as global variables so the "save" definition can easily access them when trying to build fits headers, everything else are static varibles set at the beginning of the code so this is the only place we need to use globals
		ref_pixel_index = (input["Date"] == reference_pointing_date) & (input["FrameNo"] == reference_pointing_frameno) #Grab index of the reference pixel
		pa_reference_pixel = float(input["PA"][ref_pixel_index]) #Grab PA of the refernce pixel
		xshift_pix = -round(input["X_Shift"]/plate_scale).astype(int) #Find starting x position in pixels for each pionting
		yshift_pix = round(input["Y_Shift"]/plate_scale).astype(int) #Find starting y position in pixels for each pionting
		geo = cube_geometry(input['PA'], xshift_pix, yshift_pix) #create object storing PA and coordinates for all pointings, along with dynamically sizing the cube
		for i in xrange(n_pointings): #Loop through each pointing to paint into datacube
			spectral_lines = lines(input['Spec_Line_File'][i], delta_v =input['Delta_V'][i])  #Spectral lines to use
			v_shift_file = input['V_Shift_File'][i] #File that defines any velocity shifts between lines, can be used to correct for species moving at different velocities or 
			spec1d, spec2d = getspec(input["Date"][i], input["WaveNo"][i], input["FrameNo"][i], input["StdNo"][i], B=input['A0V_B'][i], V=input['A0V_V'][i], #Create 1D and 2D spectra objects for all orders combining both H and K bands (easy eh?)
				y_scale=input["A0V_HI_Scale"][i], wave_smooth=input["A0V_Smooth"][i], savechecks=save_checks) 
			if subtract_continuum: #If user specifies to subtract the continuum
				spec2d.subtract_continuum() #Subtract continuum from 2D spectrum
			spec1d.combine_orders() #Combine all orders in 1D spectrum into one very long spectrum
			spec2d.combine_orders() #Combine all orders in 2D spectrum into one very long spectrum
			if fill_nans: #If user specifies to fill nans
				spec2d.fill_nans(size=5) #Fill nans with a median filter on a column by column basis
			if i==0: #if this is the first slit, trim line list and inialize arrays to store the flux and variacne datacubes
				parsed_spectral_lines = spectral_lines.parse(nanmin(spec2d.combospec.wave), nanmax(spec2d.combospec.wave)) #Only grab lines withen the wavelength range of the current order
				use_line_for_flux_calib = parsed_spectral_lines.label == flux_calibration_line
				#Shape of final 4D array in the form of (# of spectral lines, pixels along slit,velocity, pixels prependicular to  slit, ), x_expand expands along the x-axis (pointings) to make place scale same in x & y
				cube_size = array([len(parsed_spectral_lines.label), geo.y_size, 2*velocity_range/velocity_res, geo.x_size]).astype(int)
				master_cube = zeros(cube_size)  #Initialize master array to store flux in a datacube
				master_cube_var = zeros(cube_size) #Initialize master array to store variance
				master_cube_overlap = zeros(cube_size) #Initialize master array to store how many pointings overlap with a given pixel
				if use_blocks: #if user specifies to use blocks for relative flux calibrating overlapping pointings
					block_cube = zeros(cube_size)  #Initialize temporary block array to store flux in a datacube
					block_cube_var = zeros(cube_size) #Initialize temporary block array to store variance
					block_cube_overlap = zeros(cube_size) #Initialize temporary block array to store how many pointings overlap with a given pixel				
			else: #If this is not the first slit....
				if use_blocks and input["Block"][i] != input["Block"][i-1]: #If we are in a new block...
					block_cube[:] = 0.  #re-blank the block arrays for the next block
					block_cube_var[:] = 0.
					block_cube_overlap[:] = 0
				parsed_spectral_lines.recalculate_wavelengths(input['Delta_V'][i]) #Recalculate the wavelenfths in the line list based on the velocity of this observation
			try: #Try to see if v_shift_file works
				if v_shift_file == '-': #If user specifies no v_shift_file
					v_shift_file = '' #Tell position_velocity object not to use it
				pv = position_velocity(spec1d.combospec, spec2d.combospec, parsed_spectral_lines, shift_lines=v_shift_file) #Extract and create a datacube in position-velocity space of all lines in line list(s) found in spectrum
			except: #If not just catch the error and ignore it for now
				pv = position_velocity(spec1d.combospec, spec2d.combospec, parsed_spectral_lines) #Extract and create a datacube in position-velocity space of all lines in line list(s) found in spectrum
			velocity = pv.velocity
			velocity_cut = (velocity > -flux_calibration_velocity_range) & (velocity < flux_calibration_velocity_range) #Make a cut in velocity space, if the user so specifies, otherwise use all +/- 100 km/s in the cube
			nans = isnan(pv.pv) #Zero out nans before stacking
			pv.pv[nans] = 0.
			pv.var2d[nans] = 0.
			if input['Exp'][i] != 1.0: #If exposure times vary
				pv.pv /= input['Exp'][i]  #Scale flux
				pv.var2d /= input['Exp'][i]**2 #Scale variance (to propogate uncertainity
			print 'PA diff = ', geo.PA_difference[i]
			if geo.PA_difference[i] == 0.0 or geo.PA_difference[i] == 180.0:#If PA of this pointing is same PA or 180 degrees as the first pointing
				flux = tile(pv.pv[:,:,:, newaxis], x_expand) #Expand out flux and variance arrays to match this angle
				var = tile(pv.var2d[:,:,:, newaxis], x_expand)
				if geo.PA_difference[i] == 0.0: #If PAs between reference pointing and this are the same
					print 'Flipping order of flux array for i=', i,   'Shape= ', shape(flux)
					flux = flux[:,::-1,:,:] #Invert the flux and var arrays along the x axis
					var =  var[:,::-1,:,:]
			elif geo.PA_difference[i] == 90.0 or  geo.PA_difference[i] == 270.0:  #If PA rotated 90 degrees one way or the other from first pointing
				flux = swapaxes( tile(pv.pv[:,newaxis,:,:], (1,x_expand,1,1) ) , 2,3) #Expand out flux and variance arrays to match this angle
				var = swapaxes( tile(pv.var2d[:,newaxis,:,:],  (1,x_expand,1,1) ) , 2,3) 
				if geo.PA_difference[i] == 90.0: #if PAs between the reference pointing and this pointing arte the same
					print 'Flipping order of flux array for i=', i,   'Shape= ', shape(flux)
					flux = flux[:,:,:,::-1] #Invert the flux and var arrays along the y axis
					var =  var[:,:,:,::-1]
			x1 = geo.x_start[i].astype(int) #Grab x and y pixel ranges to use for constructing the datacube
			x2 = geo.x_end[i].astype(int)
			y1 = geo.y_start[i].astype(int)
			y2 =  geo.y_end[i].astype(int)
			find_nans = isnan(flux) #Blank set nans to zero
			flux[find_nans] = 0.
			var[find_nans] = 0.
			flux_holder = flux #Place flux and variance into temporary holders
			var_holder = var
			#Flux calibrate if whole width of any part of the slit overlaps part of the datacube that is already built
			ratio = 1.0
			if flux_calibrate: #If user turns flux calibraiton on, flux calibrate the current slit inside current block
				if use_blocks: #If user want to flux calibrate in blocks
					cube_slice = block_cube[use_line_for_flux_calib,y1:y2,velocity_cut,x1:x2] #grab slice of flux and variance in block cube overlapping current pointing
					cube_var_slice = block_cube_var[use_line_for_flux_calib,y1:y2,velocity_cut,x1:x2]
				else: #If user does not want to flux calibrate in blocks
					cube_slice = master_cube[use_line_for_flux_calib,y1:y2,velocity_cut,x1:x2] #grab slice of flux and variance in main datacube overlapping current pointing
					cube_var_slice = master_cube_var[use_line_for_flux_calib,y1:y2,velocity_cut,x1:x2]		
				#if abs(x1-x2) == slit_length: #If the slit is placed in the x-direction
				#	useable_pixels =  (cube_slice/sqrt(cube_var_slice) > flux_calibration_s2n_thresh) & isfinite(sum(cube_slice, axis=1, keepdims=True))#Find pixels along the slit length that can be used for flux calibration, if a nan exists along the slit width, it gets naned out in the sum above, thereby we only use pixels along the slit width if the existing cube fully covers them
				#else: #If the slit is placed in the y directio
				#	useable_pixels =  (cube_slice/sqrt(cube_var_slice) > flux_calibration_s2n_thresh) & isfinite(sum(cube_slice, axis=1, keepdims=True))#Find pixels along the slit length that can be used for flux calibration, if a nan exists along the slit width, it gets naned out in the sum above, thereby we only use pixels along the slit width if the existing cube fully covers them
				useable_pixels =  cube_slice/cube_var_slice**0.5 > flux_calibration_s2n_thresh #sigma clip out low S/N pixels so that we only use pixels of sufficient S/N for the flux calibration of this current pointing
				if any(useable_pixels): #If any pixels are useable, do the flux calibration by summing everything up and taking the ratio
					flux_slice = flux_holder[use_line_for_flux_calib,:,velocity_cut,:]
					ratio = nansum(cube_slice[useable_pixels])/nansum(flux_slice[useable_pixels])
					print 'i=',i,' Ratio =', ratio, ' # of usable pixels=', sum(useable_pixels), 'shape of cube slice=', shape(cube_slice)
			if use_blocks: #If user wants to flux calibrate in blocks, run through the current block and flux calibrate that block.  When the end of the block is reached, add it to the 
				block_cube_overlap[:,y1:y2,:,x1:x2] +=  1.0 #Increment the counter for how many pointings these pixels used
				inverse_overlap_scale = 1.0 / block_cube_overlap[:,y1:y2,:,x1:x2]
				#block_cube[:,y1:y2,:,x1:x2] =  (block_cube[:,y1:y2,:,x1:x2] * (1.0-inverse_overlap_scale)) +  (flux_holder * ratio * inverse_overlap_scale) #Load slit into master data cube, normalize flux by expnasion done along x-axis
				block_cube[:,y1:y2,:,x1:x2] *= 1.0-inverse_overlap_scale
				block_cube[:,y1:y2,:,x1:x2] += flux_holder * ratio * inverse_overlap_scale
				#block_cube_var[:,y1:y2,:,x1:x2] =  block_cube_var[:,y1:y2,:,x1:x2] * (1.0-inverse_overlap_scale) + (var_holder * ratio**2 * inverse_overlap_scale) #Load slit into master variance cube
				block_cube_var[:,y1:y2,:,x1:x2] *= 1.0-inverse_overlap_scale
				block_cube_var[:,y1:y2,:,x1:x2] += var_holder * ratio**2 * inverse_overlap_scale
				end_of_block = False #First set the end of block boolean to = False
				if i > 0 or n_pointings == 1: #But now search if the end of the block is true
					if (i == n_pointings-1): #If this is the last pointing, it is definitely the end of the last block
						end_of_block = True
					elif (input["Block"][i] != input["Block"][i+1]): #But if it is not the last pointing, check if the next pointing is in a different block, if it is this pointing is the end of the current block
						end_of_block = True
				if end_of_block: #If we are in a new block or at the end of the list, flux calibrate the last pionting in this block and then flux calibrate block to the master datacube
					if flux_calibrate and input["Block"][i] > 1: #And flux calibration is set to on
						block_cube_slice = block_cube[use_line_for_flux_calib,:,velocity_cut,:] #Make a cut of the block for the flux calibration velocity range and line to use
						#block_var_slice = block_cube_var[use_line_for_flux_calib,:,velocity_cut,:] #Make a cut of the block variance for the flux calibration velocity range and line to use
						master_cube_slice = master_cube[use_line_for_flux_calib,:,velocity_cut,:] #Make a cut of the master cube for the flux calibration velocity range and line to use
						#master_cube_var_slice = master_cube_var[use_line_for_flux_calib,:,velocity_cut,:] #Make a cut of the master cube variance for the flux calibration velocity range and line to use
						useable_pixels = (block_cube_slice/block_cube_var[use_line_for_flux_calib,:,velocity_cut,:]**0.5 > flux_calibration_s2n_thresh) & (master_cube_slice/master_cube_var[use_line_for_flux_calib,:,velocity_cut,:]**0.5 > flux_calibration_s2n_thresh) #Find pixels that overlap in block and 
						ratio = nansum(master_cube_slice[useable_pixels]) / nansum(block_cube_slice[useable_pixels]) #Calculate ratio to apply relative flux calibration of master cube to the current block
					else: #If there is no flux calibration, do not flux calibrate and just set ratio = 1
						useable_pixels = False
						ratio = 1.0
					print 'Block = ', input["Block"][i],' Ratio =', ratio, ' # of usable pixels=', sum(useable_pixels)
					block_pixels = block_cube_overlap > 0.0 #Find all pixels that block occupies
					block_fraction = block_cube_overlap[block_pixels] / (master_cube_overlap[block_pixels] + block_cube_overlap[block_pixels]) #Find fraction of overlapping slits the block occupies in the master cube
					#master_cube[block_pixels] = (master_cube[block_pixels]*(1.0-block_fraction)) + (block_cube[block_pixels]*block_fraction*ratio) #Load slit into master data cube, normalize flux by expnasion done along x-axis
					master_cube[block_pixels] *= 1.0-block_fraction
					master_cube[block_pixels] += block_cube[block_pixels]*block_fraction*ratio
					#master_cube_var[block_pixels] =  (master_cube_var[block_pixels]*(1.0-block_fraction)) + (block_cube_var[block_pixels]*block_fraction*ratio**2) #Load slit into master variance cube
					master_cube_var[block_pixels] *= 1.0-block_fraction
					master_cube_var[block_pixels] += block_cube_var[block_pixels]*block_fraction*ratio**2
					master_cube_overlap[block_pixels] +=  block_cube_overlap[block_pixels] #Increment the counter for how many pointings these pixels used
			else: #If user does not want to use blocks for flux calibration, flux calibrate this pointing directly into the master datacube
				master_cube_overlap[:,y1:y2,:,x1:x2] +=  1.0 #Increment the counter for how many pointings these pixels used
				inverse_overlap_scale = 1.0 / master_cube_overlap[:,y1:y2,:,x1:x2]
				master_cube[:,y1:y2,:,x1:x2] *= 1.0-inverse_overlap_scale
				master_cube[:,y1:y2,:,x1:x2] += flux_holder * ratio * inverse_overlap_scale
				master_cube_var[:,y1:y2,:,x1:x2] *= 1.0-inverse_overlap_scale
				master_cube_var[:,y1:y2,:,x1:x2] += var_holder * ratio**2 * inverse_overlap_scale
			if input["Date"][i] == reference_pointing_date and input["FrameNo"][i] == reference_pointing_frameno:
				x_reference_pixel = (float(x1)+float(x2))/2.0 #Set reference pixels to be center of first slit pointing
				y_reference_pixel = (float(y1)+float(y2))/2.0 #Set reference pixels to be center of first slit pointing
				#pa_reference_pixel = float(input["PA"][i]) #Set PA of reference pixel
			gc.collect() #Do garbage collection to free up memory lying around
		#master_cube /= master_cube_overlap.astype(float) * float(x_expand)#Scale overlapping pixels down by the number of pointings overlapping a given pixel , #store full datacube in bar object, and normalize flux to number of pointings (divide by factor we expand x axis by)
		#master_cube_var /= master_cube_overlap.astype(float)**2 * float(x_expand)**2  #Also for the variance: scale overlapping pixels down by the number of pointings overlapping a given pixel #Store full variance datacube in bar object, and normalize uncertainity to number of pointings
		master_cube /= float(x_expand)#Scale overlapping pixels down by the number of pointings overlapping a given pixel , #store full datacube in bar object, and normalize flux to number of pointings (divide by factor we expand x axis by)
		master_cube_var /= float(x_expand)**2  #Also		
		empty_pixels = master_cube == 0.0 #Blank out empty pixels with nans
		master_cube[empty_pixels] = nan
		master_cube_var[empty_pixels] = nan
		self.cube = master_cube# #store full datacube in bar object, and normalize flux to number of pointings (divide by factor we expand x axis by)
		self.var = master_cube_var # #Store full variance datacube in bar object, and normalize uncertainity to number of pointings
		self.label = parsed_spectral_lines.label #Save labels of spectral lines so that lines can be later retrieved by matching their label strings to what the user wants
		self.velocity = velocity #Save velocity per pixel along velocity axis for later being able to cut and collapse parts of the datacubes
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
		sig1 = self.getimage(input_label_1, vrange=vrange, variance=True)**0.5#Grab 1-sigma uncertainity for each image
		sig2 = self.getimage(input_label_2, vrange=vrange, variance=True)**0.5
		ratio = flux1 / flux2 #Take ratio of line 1 / line 2
		sigma = ratio * ((sig1/flux1)**2 + (sig2/flux2)**2)**0.5 #Uncertainity in ratio
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
			total_sigma = total_var**0.5
			find_line = h.label == label
			h.F[find_line] = total_flux
			h.sigma[find_line] = total_sigma
			h.s2n[find_line] = total_flux / total_sigma
		return(h)
	def extract_full(self, vrange=velocity_range): #Extract full datacube to get H2 line fluxes
		h = self.extract_half(vrange=vrange, dim=False, median_multiplier=0.0) #Pretend we aqre extracting a bright or dim region, but extract everything by setting threshold to 0.
		return(h) #Return the result
	#Function fills any gaps 
	# def fill_gaps(self, size=1, axis='x'):
	# 	cube = copy.deepcopy(self.cube)
	# 	var = copy.deepcopy(self.var)
	# 	n_lines, ny, n_velocity, nx = shape(cube) #Calculate pixel sizes
	# 	filled_slice = zeros([ny,nx]) #Create array that will store the smoothed median spectrum
	# 	filled_slice_var = zeros([ny,nx])
	# 	if axis == 'x':
	# 		x_left = arange(nx) - size #Create array to store left side of running median
	# 		x_right = arange(nx) + size + 1 #Create array to store right side of running median
	# 		x_left[x_left < 0] = 0 #Set pixels beyond edge of order to be nonexistant
	# 		x_right[x_right > nx] = nx - 1 #Set pixels beyond right edge of order to be nonexistant
	# 		x_size = x_right - x_left #Calculate number of pixels in the x (wavelength) direction
	# 		#g = Gaussian2DKernel(stddev=5.0, x_size=3, y_size=3) #Set up convolution kernel
	# 		for i in xrange(n_lines): #Loop through every spectral line
	# 			for j in xrange(n_velocity): #Loop through every velocity slice
	# 				for k in xrange(nx): #This loop does the running of the median down the spectrum each pixel
	# 					filled_slice[:,k] = nanmedian(cube[i,:,j,x_left[k]:x_right[k]], axis=1) #Calculate median between x_left and x_right for a given pixel
	# 					filled_slice_var[:,k] = nanmedian(var[i,:,j,x_left[k]:x_right[k]], axis=1) #Calculate median between x_left and x_right for a given pixel
	# 				nans_found = ~isfinite(cube[i,:,j,:]) #Find nans
	# 				cube[i,:,j,:][nans_found] = filled_slice[nans_found] #Fill in the gaps
	# 				var[i,:,j,:][nans_found]  = filled_slice_var[nans_found] #Fill in the gaps
	# 	elif axis == 'y':
	# 		y_left = arange(ny) - size #Create array to store left side of running median
	# 		y_right = arange(ny) + size + 1 #Create array to store right side of running median
	# 		y_left[y_left < 0] = 0 #Set pixels beyond edge of order to be nonexistant
	# 		y_right[y_right > ny] = ny - 1 #Set pixels beyond right edge of order to be nonexistant
	# 		y_size = y_right - y_left #Calculate number of pixels in the y (wavelength) direction
	# 		#g = Gaussian2DKernel(stddev=5.0, x_size=3, y_size=3) #Set up convolution kernel
	# 		for i in xrange(n_lines): #Loop through every spectral line
	# 			for j in xrange(n_velocity): #Loop through every velocity slice
	# 				for k in xrange(ny): #This loop does the running of the median down the spectrum each pixel
	# 					filled_slice[k,:] = nanmedian(cube[i,y_left[k]:y_right[k],j,:], axis=0) #Calculate median between x_left and x_right for a given pixel
	# 					filled_slice_var[k,:] = nanmedian(var[i,y_left[k]:y_right[k],j,:], axis=0) #Calculate median between x_left and x_right for a given pixel
	# 				nans_found = ~isfinite(cube[i,:,j,:]) #Find nans
	# 				cube[i,:,j,:][nans_found] = filled_slice[nans_found] #Fill in the gaps
	# 				var[i,:,j,:][nans_found]  = filled_slice_var[nans_found] #Fill in the gaps
	# 	self.cube = cube #Update datacube
	# 	self.var = var #Update datacube variance
	def fill_gaps(self, size=[3,3]): #Function fills any gaps, experimental version at the moment, comment out this one and uncomment the above one to go back to the old versoin
		n_lines, ny, n_velocity, nx = shape(self.cube) #Calculate pixel sizes
		mask = zeros([ny+2, nx+2], dtype=int) #Create an array to store pixle 
		kernel = Gaussian2DKernel(stddev=0.25, x_size=size[0], y_size=size[1])
		structure = ones(size, dtype=int)
		for i in xrange(n_lines):
			for j in xrange(n_velocity):
				cube_slice = self.cube[i,:,j,:]
				var_slice = self.var[i,:,j,:]
				smoothed_cube_slice = interpolate_replace_nans(cube_slice, kernel=kernel)
				smoothed_var_slice = interpolate_replace_nans(var_slice, kernel=kernel)
				mask[1:-1,1:-1][isfinite(cube_slice)] = 1
				pixels_to_replace = (mask - binary_closing(mask, structure=structure))[1:-1,1:-1] < 0
				mask[:] = 0
				cube_slice[pixels_to_replace] = smoothed_cube_slice[pixels_to_replace]
				var_sclice = smoothed_var_slice[pixels_to_replace]




