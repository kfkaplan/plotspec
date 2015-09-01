#Test script for plotspec.py library to demonstrate what library can do
from plotspec import * #Import plotspec library
import h2 #Import H2 library

#~~~~~~~~~~~~~~~~~~~~SCIENCE TARGET INFORMATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~

save.name('NGC 7027')
date = 20141023
frameno = 51
stdno = 59
H =  5.813 #H & K magnitudes for std A0V star HD 205314 for redding A0V model continuum for relative flux calibration
K = 5.788
waveno = 118
#waveno = 59
skyno = 118
HI_lines = lines('HI_ngc7027.dat', delta_v = 43.0)
ncapture_lines = lines('neutron_capture_species_ngc7027.dat', delta_v = 33.0)

#~~~~~~~~~~~~~~~~~~~~SET UP H2 TRANSITIONS OBJECT IF USING A LINE LIST WITH H2~~~~~~~~~~~~~~~~~~~~~~~~~~~~
h2_transitions = h2.make_line_list() #Set up object for storing H2 transitions

#~~~~~~~~~~~~~~~~~~~~SCRIPT FOR ANALYSING SPECTRA~~~~~~~~~~~~~~~~~~~~~~~~~~~~
spec1d, spec2d = getspec(date, waveno, frameno, stdno, H=H, K=K, y_scale=1.0, wave_smooth=0.0, delta_v=0.0) #Create 1D and 2D spectra objects for all orders combining both H and K bands (easy eh?), also input H & K mags for std. star, y_scale scales A0V H I line fit, wave_smooth smooths A0V H I line fit, delta_v moves A0V H I lines in velocity space
#spec1d.subtract_continuum() #Subtract continuum from 1D spectrum, comment out to not subtract continuum
spec2d.subtract_continuum() #Subtract continuum from 2D spectrum, comment out to not subtract continuum
spec1d.combine_orders() #Combine all orders in 1D spectrum into one very long spectrum
spec2d.combine_orders() #Combine all orders in 2D spectrum into one very long spectrum
#spec1d.plot() #Plot 1D spectrum
#spec1d.plotlines(spectral_lines, rows = 2, ymax=1e7, fontsize=14)
spec2d.plot(ncapture_lines, pause = True, close = True, label_OH = True, num_wave_labels = 1000)  #Plot 2D spectrum in DS9
pv = position_velocity(spec1d.combospec, spec2d.combospec, HI_lines) #Extract and create a datacube in position-velocity space of all lines in line list(s) found in spectrum
test_integrate_region = region(pv, file='n7027_HI.reg', background='all', name='HI') #Grab line fluxes from a user specified region, here defined in this script
pv = position_velocity(spec1d.combospec, spec2d.combospec, ncapture_lines) #Extract and create a datacube in position-velocity space of all lines in line list(s) found in spectrum
pv.view(line = '1-0 S(1)', pause=True, close=False, printlines=True) #View position-velocity datacube of all lines in DS
test_integrate_region = region(pv, file='n7027_ncapture.reg', background='all', name='ncapture') #Grab line fluxes from a user specified region, here defined in this script
#h2_transitions.set_flux(test_integrate_region) #Read fluxes into H2 transition object
#h2_transitions.calculate_column_density() #Calculate column density of H2 transition upper states from fluxes
#h2_transitions.v_plot(plot_single_temp = True, show_upper_limits = False) #Plot Boltmann Diagram of H2 transition upper states labeled with upper V states

