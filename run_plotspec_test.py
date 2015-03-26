#Test script for plotspec.py library to demonstrate what library can do
from plotspec import * #Import plotspec library
import h2 #Import H2 library

#~~~~~~~~~~~~~~~~~~~~SCIENCE TARGET INFORMATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#M 1-11
# date = 20141204
# frameno = 152
# stdno = 164
# waveno = 153
# spectral_lines = lines(['H2_lite.dat','H_and_He.dat'], delta_v = [30.0,30.0]) 
#Select observations of Orion Bar 0,0
date = 20141023
frameno = 120
stdno = 126
#waveno = 118
waveno = 126
skyno = 130
#spectral_lines = lines(['H2_lite.dat','H_and_He.dat'], delta_v = [8.0,-7.0]) 
spectral_lines = lines('H2_lite.dat', delta_v = 8.0) 
#Select observations of Hb-12
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
##NGC 7027
#date = 20141023
#frameno = 51
#stdno = 59
#waveno = 118
##waveno = 59
#skyno = 118
###spectral_lines = lines(['H2_lite.dat','H_and_He.dat'], delta_v = [8.0,-7.0]) 
###spectral_lines = lines(['neutron_capture_species_ngc7027.dat', 'HI_ngc7027.dat'], delta_v = [43.0, 33.0]) 
##spectral_lines = lines('neutron_capture_species_ngc7027.dat', delta_v = 43.0)
#spectral_lines = lines( 'HI_ngc7027.dat', delta_v =  20.0)

#~~~~~~~~~~~~~~~~~~~~SET UP H2 TRANSITIONS OBJECT IF USING A LINE LIST WITH H2~~~~~~~~~~~~~~~~~~~~~~~~~~~~
h2_transitions = h2.make_line_list() #Set up object for storing H2 transitions

#~~~~~~~~~~~~~~~~~~~~SCRIPT FOR ANALYSING SPECTRA~~~~~~~~~~~~~~~~~~~~~~~~~~~~
spec1d, spec2d = getspec(date, waveno, frameno, stdno) #Create 1D and 2D spectra objects for all orders combining both H and K bands (easy eh?)
#spec1d.subtract_continuum() #Subtract continuum from 1D spectrum, comment out to not subtract continuum
#spec2d.subtract_continuum() #Subtract continuum from 2D spectrum, comment out to not subtract continuum
spec1d.combine_orders() #Combine all orders in 1D spectrum into one very long spectrum
spec2d.combine_orders() #Combine all orders in 2D spectrum into one very long spectrum
spec1d.plot() #Plot 1D spectrum
#spec2d.plot(spectral_lines, pause = True, close = True)  #Plot 2D spectrum in DS9
pv = position_velocity(spec1d.combospec, spec2d.combospec, spectral_lines) #Extract and create a datacube in position-velocity space of all lines in line list(s) found in spectrum
pv.view(line = '1-0 S(1)', pause=True, close=False, printlines=True) #View position-velocity datacube of all lines in DS9
test_integrate_region = region(pv, file='test.reg', background='all') #Grab line fluxes from a user specified region, here defined in this script
#test_integrate_region = region(pv) #Grab line fluxes from a user specified region, set by user on command line
h2_transitions.set_flux(test_integrate_region) #Read fluxes into H2 transition object
h2_transitions.calculate_column_density() #Calculate column density of H2 transition upper states from fluxes
h2.fit_extinction_curve(h2_transitions) #Test fit an extinction curve of power law alpha, NOTE: experimental feature
h2_transitions.v_plot(plot_single_temp = True) #Plot Boltmann Diagram of H2 transition upper states labeled with upper V states

