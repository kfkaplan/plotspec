#Test script
from plotspec import *

bands = ['K']
#bands = ['H', 'K']
#M 1-11
date = 20141204
frameno = 152
stdno = 164
waveno = 153
spectral_lines = lines(['H2_lite.dat','H_and_He.dat'], delta_v = [30.0,30.0]) 
# #Select observations of Orion Bar 0,0
# date = 20141023
# frameno = 120
# stdno = 126
# waveno = 118
# skyno = 130
# spectral_lines = lines(['H2_lite.dat','H_and_He.dat'], delta_v = [8.0,-7.0]) 
##Select observations of Hb-12
#date = 20140923
#frameno = 290
#stdno = 286
#waveno = 279
#spectral_lines = lines(['H2_lite.dat','H_and_He.dat'], delta_v = [-20.0,-20.0]) 
#spectral_lines = lines('H2_lite.dat', delta_v = 0.0) 
##Select observation of NGC 7023
#date = 20140712
#frameno = 20
#stdno = 8
#waveno = 32
#spectral_lines = lines('H2_lite.dat', delta_v = -19.5) 

for band in bands: #Loop through each band
	##Read in data
	##Create 1D and 2D spectrum objects
	one_dim_spec_obj = makespec(date, band, waveno, frameno)
	two_dim_spec_obj = makespec(date, band, waveno, frameno, twodim=True)
	std_obj = makespec(date, band, waveno, stdno)
	stdflat_obj =  makespec(date, band, waveno, stdno, std=True)
	
	#Test OH residual correction, eventually roll code into a definition where i just send off the skyno,
	#so the user doesn't have to declare all the extra objects like this
	#two_dim_sky_obj =  makespec(date, band, waveno, skyno, twodim=True)
	#two_dim_spec_obj = OH_correction(two_dim_spec_obj, two_dim_sky_obj)	


	#Apply telluric correction from standard star
	one_dim_spec_obj = telluric_and_flux_calib(one_dim_spec_obj, std_obj, stdflat_obj, show_plots=False)
	two_dim_spec_obj = telluric_and_flux_calib(two_dim_spec_obj, std_obj, stdflat_obj, show_plots=False)

	#Test continuum subtraction
	one_dim_spec_obj.subtract_continuum(show = True)
	two_dim_spec_obj.subtract_continuum(show = True)
	
	#Plot 1D spectrum with orders seperate
	#Plot fancier 1D spectrum with lines labeld
	one_dim_spec_obj.plot()
	
	#Show combined 2D spectrum
	one_dim_spec_obj.combine_orders() #Issue command to combine the orders
	two_dim_spec_obj.combine_orders() #Issue command to combine the orders
	#plot_line_list(one_dim_spec_obj.combospec, spectral_lines, rows=3)
	two_dim_spec_obj.plot(spectral_lines, pause = True, close = True)
	#show2d(two_dim_spec_obj.combospec, spectral_lines, pause = True, close = True)
	#Make and show PV diagrams, starting by pointing to the 1-0 S(1) line
	pv = position_velocity(one_dim_spec_obj.combospec, two_dim_spec_obj.combospec, spectral_lines)
	#pv.normalize('1-0 S(1)') #Test normalizing by bracket gamma' #Try normalizing by H2 1-0 S(1)
	#pv.view(line = '8-6 S(1)', pause = True, close = False) #H2 line for H-band
	pv.view(line = '1-0 S(1)', pause = False, close = False, printlines=True) #H2 line in K-band

