#2D demo - IGRINS Conference 2015 in Korea
#by Kyle Kaplan
#~~~~~~~~~~~~~~~~~~~~IMPORT LIBRARIES~~~~~~~~~~~~~~~~~~~~~~~~~~~
from plotspec import * #Import plotspec library
import h2 #Import H2 library
#~~~~~~~~~~~~~~~~~~~~SCIENCE TARGET INFORMATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##M 1-11
save.name('Demo')
date = 20141204 #Date of IGRINS observations
frameno = 152 #First frame # for science target
stdno = 164 #First frame # for A0V standard star
B =  4.714 #B magnitude for A0V std.
V = 4.669  #v magnitude for A0V std.
waveno = 153 #Frame # of sky frame for wavelength calibration
ohno = 162 #Frame # for sky difference, this is used for 
demo_lines = lines('demo.dat', delta_v = 30.0) #Emission line list + velocity
h2_lines = lines('m1-11_good.dat', delta_v=30.0) #Load specific H2 line list for M 1-11
#~~~~~~~~~~~~~~~~~~~~PROCESS SCIENCE DATA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
spec1d, spec2d = getspec(date, waveno, frameno, stdno, B=B, V=V, y_scale=0.6, oh=ohno, oh_scale=0.2) #Create 1D and 2D spectra objects for all orders combining both H and K bands (easy eh?)
spec1d.combine_orders() #Combine all orders in 1D spectrum into one very long spectrum
spec2d.combine_orders() #Combine all orders in 2D spectrum into one very long spectrum
spec2d.plot(demo_lines, pause=True, close=True) #View long 2D spectrum
#~~~~~~~~~~~~~~~~~~~~SUBTRACT CONTINUUM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
spec1d.subtract_continuum(lines=demo_lines, vrange=[-50.0,50.0]) #Subtract continuum in 1D
spec1d.combine_orders() #Combine all orders in 1D spectrum into one very long spectrum
spec2d.subtract_continuum(lines=demo_lines, vrange=[-50.0,50.0]) #Subtract continuum in 2D
spec2d.combine_orders() #Combine all orders in 2D spectrum into one very long spectrum
spec2d.plot(demo_lines, pause=True, close=True) #View long 2D spectrum
#~~~~~~~~~~~~~~~~~~~~POSITION-VELOCITY DIAGRAMS AND ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pv = position_velocity(spec1d.combospec, spec2d.combospec, demo_lines) #Extract and create a datacube in position-velocity space of all lines in line list(s) found in spectrum
pv.view(line='H2 1-0 S(1)', printlines=True, pause=True, close=True) #View extracted lines and draw circle around them.
pv = position_velocity(spec1d.combospec, spec2d.combospec, h2_lines) #Extract and create a datacube in position-velocity space of all lines in line list(s) found in spectrum
demo_extract_region =  region(pv, file='demo.reg', background='all', name='Demo Region', show_regions=True) #Extract flux for region defined in DS9
ans = raw_input('Press any key to continue.')
#~~~~~~~~~~~~~~~~~~~~~~~S/N EXTRACTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
demo_extract_sn = region(pv, name='SN_Demo', background='all', s2n_cut = 0.0, s2n_mask = 5.0, line='1-0 S(1)', pixel_range=[-10,10]) #Grab line fluxes from a user specified region, here defined in this script
ans = raw_input('Press any key to continue.')
#~~~~~~~~~~~~~~~~~~~~MAKE A PLOT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
h2_demo = h2.make_line_list() #Set up object for storing H2 transitions
h2_demo.set_flux(demo_extract_region)  #Read H2 line fluxes into object to calculate dolumn density of H2
h2_demo.calculate_column_density() #Calculate column density of H2
#Test making a plot
use_V_ring = [1,2,3,4,5,6,7,8,9,10,11] #Plot the folowing vibration states
h2_demo.v_plot(plot_single_temp=False, show_upper_limits=False, s2n_cut = 3.0, show_labels=False, V=use_V_ring, rot_temp=True, savepdf=True) #Plot Boltzmann Diagram


