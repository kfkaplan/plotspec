from pdb import set_trace as stop #Use stop() for debugging
#from scipy import *
from pylab import *
from matplotlib.backends.backend_pdf import PdfPages  #For outputting a pdf with multiple pages (or one page)
from mpl_toolkits.mplot3d import Axes3D #For making 3D plots
from astropy.modeling import models, fitting #Import astropy models and fitting for fitting linear functions for tempreatures (e.g. rotation temp)
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import linregress
import copy
from scipy.linalg import lstsq
from bottleneck import *
from astropy.io import ascii
#from numba import jit #Import numba


#Global variables, modify
default_single_temp = 1500.0 #K
default_single_temp_y_intercept = 22.0
alpha = arange(0.0, 10.0, 0.01) #Save range of power laws to fit extinction curve [A_lambda = A_lambda0 * (lambda/lambda0)^alpha
lambda0 = 2.12 #Wavelength in microns for normalizing the power law exctinoction curve, here it is set to the K-badn at 2.12 um
wave_thresh = 0.05 #Set wavelength threshold (here 0.1 um) for trying to measure extinction, we need the line pairs to be far enough apart we can get a handle on the extinction

#Global variables, do not modify
#cloudy_dir = '/Volumes/home/CLOUDY/'
#cloudy_dir = '/Volumes/IGRINS_Data/CLOUDY/'
#cloudy_dir = '/Users/kfkaplan/Desktop/CLOUDY/'
cloudy_dir = '/Volumes/IGRINS_Data_Backup/CLOUDY/'
#cloudy_dir = '/Users/kkaplan1/Desktop/workathome_igrins_data/CLOUDY/'
data_dir = 'data/' #Directory where H2 data is stored for cloudy
# energy_table = data_dir + 'energy_X.dat' #Name of table where Cloudy stores data on H2 electronic ground state rovibrational energies
# transition_table = data_dir + 'transprob_X.dat' #Name of table where Cloudy stores data on H2 transition probabilities (Einstein A coeffs.)
energy_table = data_dir + 'roueff_2019_energies.dat' #Path to table that stores data on H2 electronic ground state rovibrational energies from Table 2 in Roueff et al. (2019)
roueff_2019_table = data_dir + 'roueff_2019_table2.tsv' #Path to table storing theoretical molecular data from Table 2 in Roueff et al. (2019)
k = 0.69503476 #Botlzmann constant k in units of cm^-1 K^-1 (from http://physics.nist.gov/cuu/Constants/index.html)
h = 6.6260755e-27 #Plank constant in erg s, used for converting energy in wave numbers to cgs
c = 2.99792458e10 #Speed of light in cm s^-1, used for converting energy in wave numbers to cgs

#Make array of color names
max_color = 15.0 #Set maximum color index
color_gradient = cm.jet(arange(max_color)/max_color) #Set up color list from a canned color bar found in python
color_list = ['black','gray','darkorange','blue','red','green','orange','magenta','darkgoldenrod','purple','deeppink','darkolivegreen', 'cyan','yellow','beige']
symbol_list = ['o','v','8','x','s','*','h','D','^','8','1','o','o','o','o','o','o','o'] #Symbol list for rotation ladders on black and white Boltzmann plot
#for c in matplotlib.colors.cnames:
    #color_list.append(c)

# def plot_with_subtracted_temperature(transitions):
# 	# row and column sharing
# 	f, axs = subplots(8, 2, sharex='col', sharey='row')v
# 	for x in range(8):
# 		axs[i]



def get_surface(h2obj, v_range=[2,13], s2n_cut=-1.0): #Find and plot the "fundamental plane"
	x = h2obj.J.u #Set up x,y,z for all data points
	y = h2obj.V.u
	z = log(h2obj.N)
	#surf = find_surface(x,y,z) #Fit surface
	# Fit the data using astropy.modeling
	i = (h2obj.s2n > s2n_cut) & (h2obj.N > 0.) & (h2obj.V.u >= v_range[0]) & (h2obj.V.u <= v_range[1]) #Find datapoints in high enough vibration states
	j = (h2obj.s2n > s2n_cut) & (h2obj.N > 0.) #Find all useful datapoints
	p_init = models.Polynomial2D(degree=1)
	fit_p = fitting.LevMarLSQFitter()
	p = fit_p(p_init, x[i], y[i] ,z[i])
	print(p)
	stop()
	surf_obj = make_line_list() #Set up H2 line object to store results from fit
	surf_obj.N = e**p(surf_obj.J.u, surf_obj.V.u) #Store results from fit
	return surf_obj #Return H2 object storing surface fit



# def find_surface(x, y, z, iterations=10): #Iteratively fit surface
# 	tot_delta_z = zeros(len(z)) #Store all changes in delta_z, and x and y slopes
# 	z = copy.deepcopy(z) #Make sure we don't modify the original
# 	for i in range(iterations): #Loop through number of iterations
# 		#delta_z = median(z) #Find delta z
# 		#z = z - delta_z #Subtract delta z
# 		#stop()
# 		#tot_delta_z = tot_delta_z + delta_z #Store total change in z direction
# 		fit_x = linregress(x, z) #Do al inear fit to x and get the difference
# 		delta_z = x*fit_x.slope + fit_x.intercept
# 		#stop()
# 		z = z - delta_z #Subtract delta z
# 		tot_delta_z = tot_delta_z + delta_z #Store total change in z direction
# 		fit_y = linregress(y, z)
# 		delta_z = y*fit_y.slope + fit_y.intercept
# 		#stop()
# 		z = z - delta_z #Subtract delta z
# 		tot_delta_z = tot_delta_z + delta_z #Store total change in z direction
# 	stop()
# 	return(tot_delta_z) #Return the z value of the surface



def import_black_and_van_dishoeck(): #Read in line intensities for model 14 from Black & van Dishoeck (1987) table 3, then set column densities to that model
	file_name = 'data/black_and_van_dishoeck_1987_table3.dat' #Name of electronic table
	labels = loadtxt(file_name, usecols=(0,), dtype='str', unpack=True, delimiter='\t') #Read in H2 line labels
	intensities = loadtxt(file_name, usecols=(1,), dtype='float', unpack=True, delimiter='\t') #Read in intensities of each line for model 14
	model = make_line_list() #Create object
	model.read_model(labels, intensities) #Stick intenities into this line list object
	model.calculate_column_density() #Calculate column densities from model 14
	model.normalize()
	return model #Return object

#Read in an ascii file in the format line \t flux \t sigma, normalize, and caclulate column densities from the fluxs given
def import_data(file_name, normalize_to='5-3 O(3)'):
	labels = loadtxt(file_name, usecols=(0,), dtype='str', unpack=True, delimiter='\t') #Read in H2 line labels
	flux, sigma = loadtxt(file_name, usecols=(1,2,), dtype='float', unpack=True, delimiter='\t') #Read in line fluxes and uncertainities
	h = make_line_list() #Create object
	h.read_data(labels, flux, sigma) #Stick fluxes and uncertainities into this line list object
	h.calculate_column_density(normalize=False) #Calculate column densities from data
	h.normalize(label=normalize_to) #Normalize to the line defined above
	return h #Return object

#This defintion reads in the data from Takahashi & Uehara 2001 and creates an H2 transitions object storing the data, this is for comparing
#the IGRINS data to formation pumping models
def read_takahashi_uehara_2001_model():
	labels = loadtxt('h2_models/takahashi_uehara_2001.dat', unpack=True, dtype='str', delimiter='\t', usecols=(0,)) #Read in line list wavelengths
	data_ice_A, data_ice_B, data_Si_A, data_Si_B, data_C_A, data_C_B = loadtxt('h2_models/takahashi_uehara_2001.dat', unpack=True, dtype='f', delimiter='\t', usecols=(1,2,3,4,5,6,))
	ice_A = make_line_list() #Make line lists for icy mantel, Si, and Carbonacious dust types from the models
	ice_B = make_line_list()
	Si_A = make_line_list()
	Si_B = make_line_list()
	C_A = make_line_list()
	C_B = make_line_list()
	for i in range(len(labels)): #Loop through each line in the table
		match = ice_A.label == labels[i] #Match H2 transition objects to the line and set the flux to the intensity in the table
		ice_A.F[match] = data_ice_A[i] #Take intensity from table and paint it onto the flux for the respective lines for the respective models
		ice_B.F[match] = data_ice_B[i]
		Si_A.F[match] = data_Si_A[i]
		Si_B.F[match] = data_Si_B[i]
		C_A.F[match] = data_C_A[i]
		C_B.F[match] = data_C_B[i]
	ice_A.calculate_column_density() #Given the intensities, now convert to column densities
	ice_B.calculate_column_density()
	Si_A.calculate_column_density()
	Si_B.calculate_column_density()
	C_A.calculate_column_density()
	C_B.calculate_column_density()
	for i in range(len(labels)): #Loop through each line in the table
		match = ice_A.label == labels[i] #Match H2 transition objects to the line and set the column density N to what is in the table
		same_upper_level = (ice_A.V.u == ice_A.V.u[match]) & (ice_A.J.u == ice_A.J.u[match]) #Find all lines from the same upper state
		ice_A.N[same_upper_level] = ice_A.N[match]
		ice_B.N[same_upper_level] = ice_B.N[match]
		Si_A.N[same_upper_level] = Si_A.N[match]
		Si_B.N[same_upper_level] = Si_B.N[match]
		C_A.N[same_upper_level] = C_A.N[match]
		C_B.N[same_upper_level] = C_B.N[match]
	return(ice_A, ice_B, Si_A, Si_B, C_A, C_B) #Return objects


#def multi_temp_function(x, c1, c2, c3, c4, c5, T1, T2, T3, T4, T5): #Function of 5 temperatures and coefficients for fitting boltzmann diagrams of gas with multiple thermal components
#	return c1*exp(-x/T1) + c2*exp(-x/T2) + c3*exp(-x/T3) + c4*exp(-x/T4) + c5*exp(-x/T5) 


def multi_temp_func(x,b, c1, c2, c3, T1, T2, T3): #Function of 3 temperatures and coefficients for fitting boltzmann diagrams of gas with multiple thermal components
    return b + log(c1*e**(-x/T1) + c2*e**(-x/T2)+ c3*e**(-x/T3))

def single_temp_func(x,b,T): #Function of a single temperature for fitting noltzmann diagrams for gas with a single thermal component
	return b - (x/T)


def linear_function(x, m, b): #Define a linear function for use with scipy.optimize curve_fit, for fitting rotation temperatures
	return m*x + b

#Since scipy sucks, find uncertainity in slope for just two points
def two_point_slope_uncertainity(x,y,sig_y):
	#slope = (y[1]-y[0])/(x[1]-x[0]) #get slope through two points
	extreme_1 = (y[1]+sig_y[1]-y[0]-sig_y[0])/(x[1]-x[0]) #Get one extreme slope
	extreme_2 = (y[1]-sig_y[1]-y[0]+sig_y[0])/(x[1]-x[0]) #Get other extreme slope
	sig_slope = abs(extreme_1 - extreme_2) / 2.0 #Average two extremes together
	return sig_slope


def make_line_list():
	#Read in molecular data
	# level_V, level_J = loadtxt(energy_table, usecols=(0,1), unpack=True, dtype='int', skiprows=1) #Read in data for H2 ground state rovibrational energy levels
	# level_E = loadtxt(energy_table, usecols=(2,), unpack=True, dtype='float', skiprows=1)
	# trans_Vu, trans_Ju, trans_Vl, trans_Jl =  loadtxt(transition_table, usecols=(1,2,4,5), unpack=True, dtype='int', skiprows=1) #Read in data for the transitions (ie. spectral lines which get created by the emission of a photon)
	# trans_A =  loadtxt(transition_table, usecols=(6,), unpack=True, dtype='float', skiprows=1) #Read in data for the transitions (ie. spectral lines which get created by the emission of a photon)
	# n_transitions = len(trans_Vu) #Number of transitions

	#Organize molecular data into objects storing J, V, Energy, and A values
	# J_obj = J(trans_Ju, trans_Jl) #Create object storing upper and lower J levels for each transition
	# V_obj = V(trans_Vu, trans_Vl) #Create object storing upper and lower V levels for each transition
	# A = trans_A
	# E_u = zeros(n_transitions)
	# E_l = zeros(n_transitions)
	# for i in range(n_transitions):
	# 	E_u[i] = level_E[ (level_V == trans_Vu[i]) & (level_J == trans_Ju[i]) ]
	# 	E_l[i] = level_E[ (level_V == trans_Vl[i]) & (level_J == trans_Jl[i]) ]
	# E_obj = E(E_u, E_l) #Create object for storing energies of upper and lower rovibrational levels for each transition
	#Create and return the transitions object which stores all the information for each transition
	t = ascii.read(roueff_2019_table, data_start=3)
	J_obj = J(t['Ju'].data, t['Jl'].data) #Create object storing upper and lower J levels for each transition
	V_obj = V(t['vu'].data, t['vl'].data) #Create object storing upper and lower V levels for each transition
	E_obj = E(36118.0695+t['Eu'], 36118.0695+t['Eu']-t['sigma'])  #Create object for storing energies of upper and lower rovibrational levels for each transition
	A = t['A'] #Grab transition probabilities
	transitions = h2_transitions(J_obj, V_obj, E_obj, A) #Create main transitions object
	return transitions #Return transitions object


#Calculate a weighted mean for extinction (A_V)
def calculate_exctinction(transitions, use_Av = [0.0,50.0]):
	A_lambda = array([ 0.482,  0.282,  0.175,  0.112,  0.058]) #(A_lambda / A_V) extinction curve from Rieke & Lebofsky (1985) Table 3
	l = array([ 0.806,  1.22 ,  1.63 ,  2.19 ,  3.45 ]) #Wavelengths for extinction curve from Rieke & Lebofsky (1985)
	extinction_curve = interp1d(l, A_lambda, kind='quadratic') #Create interpolation object for extinction curve from Rieke & Lebofsky (1985)
	n_doubles_found = 0 #Count doubles (pair from same upper state)
	n_trips_found = 0 #Count trips
	i = (transitions.F != 0.0) & (transitions.s2n > 3.0) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
	J_upper_found = unique(transitions.J.u[i]) #Find J for all (detected) transition upper states
	V_upper_found = unique(transitions.V.u[i]) #Find V for all (detected) transition upper states
	lines_found = []
	Avs = [] #Store Avs found
	sigma_Avs = [] #store uncertainity in Avs 
	for V in V_upper_found: #Check each upper V for pairs		
		for J in J_upper_found: #Check each upper J for pairs
			match_upper_states = (transitions.J.u[i] == J) & (transitions.V.u[i] == V) #Find all transitions from the same upper J and V state
			waves = transitions.wave[i][match_upper_states] #Store wavelengths of all found transitions
			s = argsort(waves) #sort by wavelength
			waves = waves[s]
			labels = transitions.label[i][match_upper_states][s]
			if len(waves) == 2 and abs(waves[0]-waves[1]) > wave_thresh: #If a single pair of lines from the same upper state are found, calculate differential extinction for this single pair
				print('For '+labels[0]+'/'+labels[1]+'    '+str(waves[0])+'/'+str(waves[1])+':')
				ratio_of_ratios = transitions.flux_ratio(labels[0], labels[1], sigma=True) /  transitions.intrinsic_ratio(labels[0], labels[1])
				Av =  -2.5*log10(ratio_of_ratios[0][0])/(extinction_curve(waves[0])-extinction_curve(waves[1])) #Calculate exctinction in Av
				sigma_Av = abs(-2.5 * (ratio_of_ratios[1][0])/(Av*log(10.0))) #Calculate uncertainity in extinction in Av
				print('Observed/intrinsic = %4.2f' % ratio_of_ratios[0][0] + ' +/- %4.2f' % (ratio_of_ratios[1][0]))
				print('Calculated A_V = ', Av)
				lines_found.append(labels[0]) #Store line labels of lines found
				lines_found.append(labels[1])
				if Av > use_Av[0] and Av < use_Av[1]: #If Av is within a reasonable range
					Avs.append(Av) #Store Avs
					sigma_Avs.append(sigma_Av) #Store sigma Avs
			elif len(waves) == 3: #If three liens are found from the same upper state, calculate differential extinction from differences between all three lines
				lines_found.append(labels[0]) #Store line labels of lines found
				lines_found.append(labels[1])
				lines_found.append(labels[2])
				#Pair 1
				if abs(waves[0] - waves[1]) > wave_thresh: #check if pair of lines are far enough apart
					print('For '+labels[0]+'/'+labels[1]+'    '+str(waves[0])+'/'+str(waves[1])+':')
					ratio_of_ratios = transitions.flux_ratio(labels[0], labels[1], sigma=True) /  transitions.intrinsic_ratio(labels[0], labels[1])
					Av =  -2.5*log10(ratio_of_ratios[0][0])/(extinction_curve(waves[0])-extinction_curve(waves[1])) #Calculate exctinction in Av
					sigma_Av = abs(-2.5 * (ratio_of_ratios[1][0])/(Av*log(10.0))) #Calculate uncertainity in extinction in Av
					print('Observed/intrinsic = %4.2f' % ratio_of_ratios[0][0] + ' +/- %4.2f' % (ratio_of_ratios[1][0]))
					print('Calculated A_V = ', Av)
					if Av > use_Av[0] and Av < use_Av[1]: #If Av is within a reasonable range
						Avs.append(Av) #Store Avs
						sigma_Avs.append(sigma_Av) #Store sigma Avs
				#Pair 2
				if abs(waves[0] - waves[2]) > wave_thresh: #check if pair of lines are far enoug7h apart
					print('For '+labels[0]+'/'+labels[2]+'    '+str(waves[0])+'/'+str(waves[2])+':')
					ratio_of_ratios = transitions.flux_ratio(labels[0], labels[2], sigma=True) /  transitions.intrinsic_ratio(labels[0], labels[2])
					Av =  -2.5*log10(ratio_of_ratios[0][0])/(extinction_curve(waves[0])-extinction_curve(waves[2])) #Calculate exctinction in Av
					sigma_Av = abs(-2.5 * (ratio_of_ratios[1][0])/(Av*log(10.0))) #Calculate uncertainity in extinction in Av
					print('Observed/intrinsic = %4.2f' % ratio_of_ratios[0][0] + ' +/- %4.2f' % (ratio_of_ratios[1][0]))
					print('Calculated A_V = ', Av)
					if Av > use_Av[0] and Av < use_Av[1]: #If Av is within a reasonable range
						Avs.append(Av) #Store Avs
						sigma_Avs.append(sigma_Av) #Store sigma Avs
				#Pair 3
					print('For '+labels[1]+'/'+labels[2]+'    '+str(waves[1])+'/'+str(waves[2])+':')
					ratio_of_ratios = transitions.flux_ratio(labels[1], labels[2], sigma=True) /  transitions.intrinsic_ratio(labels[1], labels[2])
					Av =  -2.5*log10(ratio_of_ratios[0][0])/(extinction_curve(waves[1])-extinction_curve(waves[2])) #Calculate exctinction in Av
					sigma_Av = abs(-2.5 * (ratio_of_ratios[1][0])/(Av*log(10.0))) #Calculate uncertainity in extinction in Av
					print('Observed/intrinsic = %4.2f' % ratio_of_ratios[0][0] + ' +/- %4.2f' % (ratio_of_ratios[1][0]))
					print('Calculated A_V = ', Av)
					if Av > use_Av[0] and Av < use_Av[1]: #If Av is within a reasonable range
						Avs.append(Av) #Store Avs
						sigma_Avs.append(sigma_Av) #Store sigma Avs
				if abs(waves[1] - waves[2]) > wave_thresh: #check if pair of lines are far enough apart
					n_trips_found += 1
	print('Number of pairs from same upper state = ', n_doubles_found)
	print('Number of tripples from same upper state = ', n_trips_found)
	Avs = array(Avs) #Convert to numpy arrays to do vector math to figure out weighted mean
	sigma_Avs = array(sigma_Avs)
	weights = sigma_Avs**-2
	summed_weights = nansum(weights)
	weighted_mean_Av = nansum(Avs * weights) / summed_weights
	weighted_sigma_Av = sqrt(1.0 / summed_weights)
	print('Weighted mean Av = %4.2f' %  weighted_mean_Av + ' +/- %4.2f' % weighted_sigma_Av)
	return weighted_mean_Av, weighted_sigma_Av

#Simple algorithim to vary alpha (exctinction curve power law) and A_k and do a chi sq
#minimization to find the best fit for the observed - intrinsic line ratios
def find_best_extinction_correction(h_in, s2n_cut=1.0):
	#First find all the line pairs and store their indicies
	pair_a = [] #Store index numbers of one of a set of line pairs from the same upper state
	pair_b = [] #Store index numbers of the other of a set of line pairs from the same upper state
	i = (h_in.F != 0.0) & (h_in.s2n > 0.5) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
	J_upper_found = unique(h_in.J.u[i]) #Find J for all (detected) transition upper states
	V_upper_found = unique(h_in.V.u[i]) #Find V for all (detected) transition upper states
	for V in V_upper_found: #Check each upper V for pairs		
		for J in J_upper_found: #Check each upper J for pairs
			i = (h_in.F != 0.0) & (h_in.s2n > s2n_cut) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
			match_upper_states = (h_in.J.u[i] == J) & (h_in.V.u[i] == V) #Find all transitions from the same upper J and V state
			waves = h_in.wave[i][match_upper_states] #Store wavelengths of all found transitions
			s = argsort(waves) #sort by wavelength
			waves = waves[s]
			labels = h_in.label[i][match_upper_states][s]
			if len(waves) == 2 and abs(waves[0]-waves[1]) > wave_thresh: #If a single pair of lines from the same upper state are found, calculate observed vs. intrinsic ratio
				pair_a.append(where(h_in.wave == waves[0])[0][0])
				pair_b.append(where(h_in.wave == waves[1])[0][0])
			elif len(waves) == 3: #If three liens are found from the same upper state, calculate differential extinction from differences between all three lines
				#Pair 1
				if abs(waves[0] - waves[1]) > wave_thresh:
					pair_a.append(where(h_in.wave == waves[0])[0][0])
					pair_b.append(where(h_in.wave == waves[1])[0][0])
				#Pair 2
				if abs(waves[0] - waves[2]) > wave_thresh: #check if pair of lines are far enoug7h apart
					pair_a.append(where(h_in.wave == waves[0])[0][0])
					pair_b.append(where(h_in.wave == waves[2])[0][0])
				if abs(waves[1] - waves[2]) > wave_thresh: #check if pair of lines are far enough apart
					pair_a.append(where(h_in.wave == waves[1])[0][0])
					pair_b.append(where(h_in.wave == waves[2])[0][0])
	pair_a = array(pair_a) #Turn lists of indicies into arrays of indicies
	pair_b = array(pair_b)
	chisqs = [] #Store chisq for each possible extinction and extinction law
	alphas = [] #Store alphas for each possible exctinction and extinction law
	A_Ks = [] #Store extinctions for each possible exctinction and exctinction law
	for a in arange(0.5,3.0,0.1): #Loop through different exctinction law powers
		for A_K in arange(0.0,5.0,0.01): #Loop through different possible K band exctinctions
			h = copy.deepcopy(h_in) #Make a copy of the input h2 line object
			A_lambda = A_K * h.wave**(-a) / lambda0**(-a) #Calculate an extinction correction
			h.F *= 10**(0.4*A_lambda) #Apply extinction correction
			h.calculate_column_density() #Calculate column densities from each transition, given the guess at extinction correction
			chisq = nansum((h.N[pair_a] - h.N[pair_b])**2 /  h.N[pair_b]) #Calculate chisq from all line pairs that arise from same upper states
			chisqs.append(chisq) #Store chisq and corrisponding variables for extinction correction
			alphas.append(a)
			A_Ks.append(A_K)
	chisqs = array(chisqs) #Convert lists to arrays
	alphas = array(alphas)
	A_Ks = array(A_Ks)
	best_fit = chisqs == nanmin(chisqs) #Find the minimum chisq and best fit alpha and A_K
	best_fit_A_K = A_Ks[best_fit]
	best_fit_alpha = alphas[best_fit]
	print('Best fit alpha =', best_fit_alpha) #Print results so user can see
	print('Best fit A_K = ', best_fit_A_K)
	A_lambda = best_fit_A_K * h_in.wave**(-best_fit_alpha) / lambda0**(-best_fit_alpha) #Calculate an extinction correction
	h_in.F *= 10**(0.4*A_lambda) #Apply extinction correction
	h_in.calculate_column_density() #Calculate column densities from each transition, given the new extinction correction





#Test extinction correction by animating stepping through alpha and A_K space and making v_plots of the results
def animate_extinction_correction(h_in):
	with PdfPages('animate_extinction_correction.pdf') as pdf:
		for a in [0.5,1.0,1.5,2.0,2.5,3.0]:
			for A_K in arange(0.0,3.0,0.1):
				h = copy.deepcopy(h_in)
				A_lambda = A_K * h.wave**(-a) / lambda0**(-a)
				h.F *= 10**(0.4*A_lambda)
				h.calculate_column_density()
				h.v_plot()
				suptitle('alpha = '+str(a)+'    A_K = '+str(A_K))
				pdf.savefig()


#Iterate adding extinction curves until you are satisfied
def iterate_extinction_curve(transitions):
	a = input('What value of alpha do you want to use? ') #Prompt from user what alpha should be
	A_K = input('What value of A_K do you want to use? ') #Prompt user for A_K
	while a != 0.0:  #Loop until user is satisfied with extinction correction
		fit_extinction_curve(transitions, a=a, A_K=A_K) #Try fitting previously inputted extinction curve, plot results
		a = input('What value of alpha do you want to use? (0. to stop iteration) ') #Prompt from user what alpha should be
		A_K = input('What value of A_K do you want to use? ') #Prompt user for A_K


#Definition that takes all the H2 lines with determined column densities and calculates as many differential extinctions as it can between
#pairs of lines that come from the same upper state, and fits an extinction curve (power law here) to them
def fit_extinction_curve(transitions, a=0.0, A_K=0.0):
	figure(1)
	clf() #Clear plot field
	figure(2)
	clf() #Clear plot field
	figure(3)
	clf() #Clear plot field

	n_doubles_found = 0 #Count doubles (pair from same upper state)
	n_trips_found = 0 #Count trips
	#i = (transitions.N != 0.0) & (transitions.s2n > 10.0) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
	i = (transitions.F != 0.0) & (transitions.s2n > 0.5) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
	J_upper_found = unique(transitions.J.u[i]) #Find J for all (detected) transition upper states
	V_upper_found = unique(transitions.V.u[i]) #Find V for all (detected) transition upper states
	pairs = [] #Set up array of line pairs for measuring the differential extinction A_lamba1-lambda2
	observed_to_intrinsic = []
	wave_sets = []
	for V in V_upper_found: #Check each upper V for pairs		
		for J in J_upper_found: #Check each upper J for pairs
			match_upper_states = (transitions.J.u[i] == J) & (transitions.V.u[i] == V) #Find all transitions from the same upper J and V state
			waves = transitions.wave[i][match_upper_states] #Store wavelengths of all found transitions
			#N = transitions.N[i][match_upper_states] #Store all column densities for found transitions
			F = transitions.F[i][match_upper_states] 
			Fsigma = transitions.sigma[i][match_upper_states] 
			intrinsic_constants =  (transitions.g[i][match_upper_states] * transitions.E.diff()[i][match_upper_states] * transitions.A[i][match_upper_states]) #Get constants for calculating the intrinsic ratios
			#Nsigma = transitions.Nsigma[i][match_upper_states] #Grab uncertainity in column densities
			if len(waves) == 2 and abs(waves[0]-waves[1]) > wave_thresh: #If a single pair of lines from the same upper state are found, calculate differential extinction for this single pair
				A_delta_lambda = -2.5*log10((F[0]/F[1]) / (intrinsic_constants[0]/intrinsic_constants[1])) #Calculate differential extinction between two H2 lines
				sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[0]/F[0])**2 + (Fsigma[1]/F[1])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
				pair = differential_extinction([waves[0], waves[1]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
				pairs.append(pair) #Save a single pair
				wave_sets.append(waves)
				n_doubles_found = n_doubles_found + 1
				observed_to_intrinsic.append((F[0]/F[1]) / (intrinsic_constants[0]/intrinsic_constants[1]))
			elif len(waves) == 3: #If three liens are found from the same upper state, calculate differential extinction from differences between all three lines
				#Pair 1
				if abs(waves[0] - waves[1]) > wave_thresh: #check if pair of lines are far enough apart
					A_delta_lambda = -2.5*log10((F[0]/F[1]) / (intrinsic_constants[0]/intrinsic_constants[1])) #Calculate differential extinction between two H2 lines
					sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[0]/F[0])**2 + (Fsigma[1]/F[1])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
					pair = differential_extinction([waves[0], waves[1]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
					pairs.append(pair) #Save a single pair
					observed_to_intrinsic.append((F[0]/F[1]) / (intrinsic_constants[0]/intrinsic_constants[1]))
				#Pair 2
				if abs(waves[0] - waves[2]) > wave_thresh: #check if pair of lines are far enoug7h apart
					A_delta_lambda = -2.5*log10((F[0]/F[2]) / (intrinsic_constants[0]/intrinsic_constants[2])) #Calculate differential extinction between two H2 lines
					sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[0]/F[0])**2 + (Fsigma[2]/F[2])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
					pair = differential_extinction([waves[0], waves[2]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
					pairs.append(pair) #Save a single pair
					observed_to_intrinsic.append((F[0]/F[2]) / (intrinsic_constants[0]/intrinsic_constants[2]))
				#Pair 3
				if abs(waves[1] - waves[2]) > wave_thresh: #check if pair of lines are far enough apart
					A_delta_lambda = -2.5*log10((F[1]/F[2]) / (intrinsic_constants[1]/intrinsic_constants[2])) #Calculate differential extinction between two H2 lines
					sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[1]/F[1])**2 + (Fsigma[2]/F[2])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
					pair = differential_extinction([waves[1], waves[2]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
					pairs.append(pair) #Save a single pair
					observed_to_intrinsic.append((F[1]/F[2]) / (intrinsic_constants[1]/intrinsic_constants[2]))
				wave_sets.append(waves)
				n_trips_found += 1

			for pair in pairs: #Loop through each pair
				if pair.s2n > 3.0:
					pair.fit_curve()
					figure(1)
					plot(alpha, pair.A_K, color=color_list[V], label = 'V = '+str(V) + ' J = ' + str(J))
					plot(alpha, pair.A_K + pair.sigma_A_K, '--', color=color_list[V])
					plot(alpha, pair.A_K - pair.sigma_A_K, '--', color=color_list[V])
					f = interp1d(alpha, pair.A_K)
					g = interp1d(alpha, pair.sigma_A_K)
					print('V = ', str(V), 'J = ', str(J),' at alpha=2, A_K = ', f(2.0), '+/-', g(2.0))
					print('for pair at waves', str(pair.waves[0]), ' & ', str(pair.waves[1]), ' A_delta_lambda=', str(pair.A))
			pairs  = []
		xlabel('Alpha')
		ylabel('$A_K$')
		legend()
		ylim([0,20])
	#show()
	#figure(2)
	#clf()
	#for pair in pairs: #Loop through each pair
	#	plot(pair.waves, [0,10**(0.4*pair.A)])
	##show()
	#print('V=', V)
	#pairs = []
	stop()

	if a == 0.0: #If user does not specify alpha to use
		a = input('What value of alpha do you want to use? ') #Prompt from user what alpha should be
	if A_K == 0.0: #If user doesn onot specify what the K-band extinction A_K should be
		A_K = input('What value of A_K do you want to use? ') #Prompt user for A_K
	A_lambda = A_K * transitions.wave**(-a) / lambda0**(-a)
	transitions.F *= 10**(0.4*A_lambda)
	transitions.sigma *= 10**(0.4*A_lambda)
	#suptitle('V = ' + str(V))

	#stop()
	print('Number of pairs from same upper state = ', n_doubles_found)
	print('Number of tripples from same upper state = ', n_trips_found)



#Test printing intrinsic ratios, for debugging/diagnosing extinction
def test_intrinsic_ratios(transitions):
	A_lambda = array([ 0.482,  0.282,  0.175,  0.112,  0.058]) #(A_lambda / A_V) extinction curve from Rieke & Lebofsky (1985) Table 3
	l = array([ 0.806,  1.22 ,  1.63 ,  2.19 ,  3.45 ]) #Wavelengths for extinction curve from Rieke & Lebofsky (1985)
	extinction_curve = interp1d(l, A_lambda, kind='quadratic') #Create interpolation object for extinction curve from Rieke & Lebofsky (1985)
	clf()
	n_doubles_found = 0 #Count doubles (pair from same upper state)
	n_trips_found = 0 #Count trips
	#i = (transitions.N != 0.0) & (transitions.s2n > 10.0) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
	i = (transitions.F != 0.0) & (transitions.s2n > 0.5) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
	J_upper_found = unique(transitions.J.u[i]) #Find J for all (detected) transition upper states
	V_upper_found = unique(transitions.V.u[i]) #Find V for all (detected) transition upper states
	lines_found = []
	for V in V_upper_found: #Check each upper V for pairs		
		for J in J_upper_found: #Check each upper J for pairs
			match_upper_states = (transitions.J.u[i] == J) & (transitions.V.u[i] == V) #Find all transitions from the same upper J and V state
			waves = transitions.wave[i][match_upper_states] #Store wavelengths of all found transitions
			s = argsort(waves) #sort by wavelength
			#N = transitions.N[i][match_upper_states] #Store all column densities for found transitions
			#F = transitions.F[i][match_upper_states] 
			#Fsigma = transitions.sigma[i][match_upper_states] 
			#intrinsic_constants =  (transitions.g[i][match_upper_states] * transitions.E.diff()[i][match_upper_states] * transitions.A[i][match_upper_states]) #Get constants for calculating the intrinsic ratios
			waves = waves[s]
			labels = transitions.label[i][match_upper_states][s]
			#Nsigma = transitions.Nsigma[i][match_upper_states] #Grab uncertainity in column densities
			if len(waves) == 2 and abs(waves[0]-waves[1]) > wave_thresh: #If a single pair of lines from the same upper state are found, calculate differential extinction for this single pair
				print('For '+labels[0]+'/'+labels[1]+'    '+str(waves[0])+'/'+str(waves[1])+':')
				#print('     Observed ratio: ', transitions.flux_ratio(labels[0], labels[1], sigma=True))
				#print('     Intrinsic ratio:',  transitions.intrinsic_ratio(labels[0], labels[1]))
				ratio_of_ratios = transitions.flux_ratio(labels[0], labels[1], sigma=True) /  transitions.intrinsic_ratio(labels[0], labels[1])
				print('Observed/intrinsic = %4.2f' % ratio_of_ratios[0][0] + ' +/- %4.2f' % (ratio_of_ratios[1][0]))
				print('Calculated A_V = ', -2.5*log10(ratio_of_ratios)/(extinction_curve(waves[0])-extinction_curve(waves[1])))
				plot([waves[0],waves[1]], [ratio_of_ratios[0], 1.])
				lines_found.append(labels[0]) #Store line labels of lines found
				lines_found.append(labels[1])
			elif len(waves) == 3: #If three liens are found from the same upper state, calculate differential extinction from differences between all three lines
				lines_found.append(labels[0]) #Store line labels of lines found
				lines_found.append(labels[1])
				lines_found.append(labels[2])
				#Pair 1
				if abs(waves[0] - waves[1]) > wave_thresh: #check if pair of lines are far enough apart
					print('For '+labels[0]+'/'+labels[1]+'    '+str(waves[0])+'/'+str(waves[1])+':')
					#print('     Observed ratio: ', transitions.flux_ratio(labels[0], labels[1], sigma=True))
					#print('     Intrinsic ratio:',  transitions.intrinsic_ratio(labels[0], labels[1]))
					ratio_of_ratios = transitions.flux_ratio(labels[0], labels[1], sigma=True) /  transitions.intrinsic_ratio(labels[0], labels[1])
					print('Observed/intrinsic = %4.2f' % ratio_of_ratios[0][0] + ' +/- %4.2f' % (ratio_of_ratios[1][0]))
					print('Calculated A_V = ', -2.5*log10(ratio_of_ratios)/(extinction_curve(waves[0])-extinction_curve(waves[1])))
					plot([waves[0],waves[1]], [ratio_of_ratios[0], 1.])

				#Pair 2
				if abs(waves[0] - waves[2]) > wave_thresh: #check if pair of lines are far enoug7h apart
					print('For '+labels[0]+'/'+labels[2]+'    '+str(waves[0])+'/'+str(waves[2])+':')
					#print('     Observed ratio: ', transitions.flux_ratio(labels[0], labels[2], sigma=True))
					#print('     Intrinsic ratio:',  transitions.intrinsic_ratio(labels[0], labels[2]))
					ratio_of_ratios = transitions.flux_ratio(labels[0], labels[2], sigma=True) /  transitions.intrinsic_ratio(labels[0], labels[2])
					print('Observed/intrinsic = %4.2f' % ratio_of_ratios[0][0] + ' +/- %4.2f' % (ratio_of_ratios[1][0]))
					print('Calculated A_V = ', -2.5*log10(ratio_of_ratios)/(extinction_curve(waves[0])-extinction_curve(waves[2])))
					plot([waves[0],waves[2]], [ratio_of_ratios[0], 1.])
				#Pair 3
					print('For '+labels[1]+'/'+labels[2]+'    '+str(waves[1])+'/'+str(waves[2])+':')
					#print('     Observed ratio: ', transitions.flux_ratio(labels[1], labels[2], sigma=True))
					#print('     Intrinsic ratio:',  transitions.intrinsic_ratio(labels[1], labels[2]))
					ratio_of_ratios = transitions.flux_ratio(labels[1], labels[2], sigma=True) /  transitions.intrinsic_ratio(labels[1], labels[2])
					print('Observed/intrinsic = %4.2f' % ratio_of_ratios[0][0] + ' +/- %4.2f' % (ratio_of_ratios[1][0]))
					print('Calculated A_V = ', -2.5*log10(ratio_of_ratios)/(extinction_curve(waves[1])-extinction_curve(waves[2])))
					plot([waves[1],waves[2]], [ratio_of_ratios[0], 1.])
				if abs(waves[1] - waves[2]) > wave_thresh: #check if pair of lines are far enough apart
					n_trips_found += 1
	#stop()
	print('Number of pairs from same upper state = ', n_doubles_found)
	print('Number of tripples from same upper state = ', n_trips_found)
	#return lines_found

##Store differential extinction between two transitions from the same upper state
class differential_extinction:
	def __init__(self, waves, A, sigma): #Input the lambda, flux, and sigma of two different lines as paris [XX,XX]
		self.waves = waves  #Store the wavleneghts as lamda[0] and lambda[1]
		self.A = A #Store differential extinction
		self.sigma = sigma #Store uncertainity in differential extinction A
		self.s2n = A / sigma
	def fit_curve(self):
		constants = lambda0**alpha / ( self.waves[0]**(-alpha) - self.waves[1]**(-alpha) ) #Calculate constants to mulitply A_delta_lambda by to get A_K
		self.A_K = self.A * constants #calculate extinction for a given power law alpha
		self.sigma_A_K = self.sigma * constants #calculate extinction for a given power law alpha


	
def import_cloudy(model=''): #Import cloudy model from cloudy directory
	h = make_line_list() #Make H2 transitions object
	# paths = open(cloudy_dir + 'process_model/input.dat') #Read in current model from process_model/input.dat
	# input_model = paths.readline().split(' ')[0]
	# distance = float(paths.readline().split(' ')[0])
	# inner_radius = float(paths.readline().split(' ')[0])
	# slit_area = float(paths.readline().split(' ')[0])
	# data_dir =  paths.readline().split(' ')[0]
	# plot_dir = paths.readline().split(' ')[0]
	# table_dir =  paths.readline().split(' ')[0]
	# paths.close()
	# if model == '': #If no model is specified by the user, read in model set in process_model/input.dat
	# 	model = input_model
	#READ IN LEVEL COLUMN DENSITY FILE
	# filename = data_dir+model+".h2col" #Name of file to open
	# v, J, E, N, N_over_g, LTE_N, LTE_N_over_g = loadtxt(filename, skiprows=4, unpack=True) #Read in H2 column density file
	# for i in range(len(v)): #Loop through each rovibrational energy level
	# 	found_transitions = (h.V.u == v[i]) & (h.J.u == J[i]) #Find all rovibrational transitions that match the upper v and J
	# 	h.N[found_transitions] = N[i] #Set column density of transitions
	#READ IN LINE EMISSION FILE AND CONVERT LINE EMISSION TO COLUMN DENSITIES
	#filename = data_dir+model+'.h2.lines'
	filename = cloudy_dir + '/run/' +model+'.h2.lines'
	line, wl_lab = loadtxt(filename, unpack=True, dtype='S', delimiter='\t', usecols=(0,8))
	Ehi, Vhi, Jhi, Elo, Vlo, Jlo = loadtxt(filename, unpack=True, dtype='int', delimiter='\t', usecols=(1,2,3,4,5,6))
	wl_mic, log_L, I_ratio, Excit, gu_h_nu_aul =  loadtxt(filename, unpack=True, dtype='float', delimiter='\t', usecols=(7,9,10,11,12))
	L=10**log_L #Convert log luminosity to linear units
	for i in range(len(L)): #Loop through each transition
		h.F[(h.V.u == Vhi[i]) & (h.V.l == Vlo[i]) & (h.J.u == Jhi[i]) & (h.J.l == Jlo[i])] = L[i] #Find current transition in h2 transitions object for list of H2 lines cloudy outputs and set flux to be equal to the luminosity of the line outputted by cloudy
	h.calculate_column_density()
	h.normalize() #Normalize to the 5-3 O(3) line
	return(h)

def combine_models(model1, model2, weight1, weight2): #Combine two models scaling each by their given weights
	combined_model = make_line_list() #Make new object for combination of both nmodels
	i = (model1.N > 0.) & (model2.N > 0.) #Use only models that include the same lines in each
	combined_model.N[i] = model1.N[i] * weight1 + model2.N[i] * weight2 #Combine the level column densities weighted by the given weights
	return combined_model #Return the single combined model

def import_emissivity(x_range=[4.25e17, 4.5e17], dr=5e15): #Import Cloudy model emmisivity adn integrate using given range
	paths = open(cloudy_dir + 'process_model/input.dat') #Read in current model
	model = paths.readline().split(' ')[0]
	distance = float(paths.readline().split(' ')[0])
	inner_radius = float(paths.readline().split(' ')[0])
	slit_area = float(paths.readline().split(' ')[0])
	data_dir =  paths.readline().split(' ')[0]
	plot_dir = paths.readline().split(' ')[0]
	table_dir =  paths.readline().split(' ')[0]
	paths.close()
	filename = data_dir+model+".line_emiss"
	#read_labels = loadtxt(filename, dtype=str, comments='$', delimiter='\t', unpack=True) #Read labels
	read_data = loadtxt(filename, dtype=float, comments='#', delimiter='\t', unpack=True, skiprows=1) #Read data
	read_H_band_wavelengths = loadtxt(data_dir+model+'.100lines_hband.waves', dtype=float) #Read H and K band wavelengths
	read_K_band_wavelengths = loadtxt(data_dir+model+'.100lines_kband.waves', dtype=float)
	read_H_band_labels = loadtxt(data_dir+model+'.100lines_hband.lines', dtype=str, delimiter='~') #Read H and K band kabeks
	read_K_band_labels = loadtxt(data_dir+model+'.100lines_kband.lines', dtype=str, delimiter='~')
	read_wavelengths = air_to_vac( concatenate([read_H_band_wavelengths, read_K_band_wavelengths]) ) #Combine H & K band wavelengths into one array and convert from air to vaccume waves
	#read_wavelengths = concatenate([read_H_band_wavelengths, read_K_band_wavelengths])
	emiss = emissivity(concatenate([read_H_band_labels, read_K_band_labels]), read_data[0,:], read_wavelengths, 10**read_data[1:,:]) #Put everything in an emissivity object
	emiss.set_H2_labels() #Set H2 labels to proper spectroscopic notation
	f = emiss.integrate_slab(x_range[0], x_range[1], dr=dr) #Integrate slab
	h = make_line_list()
	h.read_model(emiss.labels, f) #Convert model into H2 object
	#return emiss #Return the emissivity object to the user
	h.calculate_column_density()
	return h #Return H2 object storing the integrated model line emissivity


#Convert wavelenghts in air (outputted by Cloudy) into Vacuum (what IGRINS sees)
#Based on APOGEE Technical Note "Conversion from vacuum to standard air wavelengths" by Prieto (2011)
def air_to_vac(l):
    a, b1, b2, c1, c2 = [0.0, 5.792105e-2, 1.67917e-3, 238.0185, 57.362] #Coeffecients from Ciddor (1996)
    n = 1 + a + (b1 / (c1-l**-2)) + (b2 / (c2-l**-2))
    l_vac = l * n
    return(l_vac)
	
		    
#Class for storing all the data from a cloudy emissivity file
class emissivity:
    def __init__(self, labels, depth, waves, flux): #When first runnnig this class...
        self.labels = labels #Store line labels under "labels"
        self.depth = depth #Store depth (or radius) into cloud under "depth"
        self.waves = waves #Store wavelengths of lines under "waves"
        self.flux = flux #Store fluxes as a function of depth into cloud
    def get(self, label):  #return depth and flux (unpacked) for a chosen line
        index = self.labels == label #Find line
        return array([self.depth, self.flux[index,:][0]]) #Return depth and flux of line
    def plot(self, label): #Simple plot of a given line emissivity vs. depth
        line = self.get(label) #Grab depth, emissivity of specified line
        plot(line[0], line[1]) #Plot depth vs. emissivity
        show() #Show plot
    def set_H2_labels(self): #Set H2 labels to standard spectroscopic notation
        h2_line_labels =  loadtxt(cloudy_dir+'process_model/IGRINS_H2_line_list.dat', usecols=(1,), delimiter="\t", dtype='string') #Load spectroscopic notation for H2 lines
        h2_line_wave = loadtxt(cloudy_dir+'process_model/IGRINS_H2_line_list.dat', usecols=(0,), delimiter="\t") #Load
        for i in  range(len(self.labels)):
            if self.labels[i].astype('|S4') == 'H2  ':
                wave_diff = abs(h2_line_wave - self.waves[i])
                j = wave_diff == min(wave_diff) #Find index of nearest H2 line
                self.labels[i] = h2_line_labels[j][0] #Replace Cloudy H2 label with proper spectroscopic notation label
    def integrate_slab(self, inner_radius, outer_radius, dr=1e12): #Integrate up emission from a slab between inner and outer radii
        #goodpix = (self.depth >= inner_radius) & (self.depth <= outer_radius) #Find pixels in radii range
        #interp_emissivity = interp1d(self.depth[goodpix], self.flux[:,goodpix], axis=1, bounds_error=False, kind='linear') #Interpolate over all lines
        interp_emissivity = interp1d(self.depth, self.flux, axis=1, bounds_error=False, kind='linear')
        r = arange(inner_radius, outer_radius, dr) #Set up radius grid to interpolate over
        interp_flux = nansum(interp_emissivity(r)*dr, axis=1) #Sum up all line fluxes
        return interp_flux
    def slice(self, radius): #grab line emissivities at slice through cloud
        interp_emissivity = interp1d(self.depth, self.flux, axis=1, bounds_error=False, kind='linear') #Interpolate over all lines
        interp_flux = interp_emissivity(radius) #Grab fluxes and specific radius
        return interp_flux  
    # def excitation_diagrams(self, xstep=1e16, y_range=[-6,2], fname=plot_dir+model+'slices_excitation_diagrams.pdf'):  #Make a PDF where each page is an excitation diagram is an integrated slice of a Cloudy model
    #     h = h2.make_line_list() #Set up object for storing H2 transitions
    #     #interp_emissivity = interp1d(self.depth, self.flux, axis=1, bounds_error=False, kind='linear')
    #     #r = arange(inner_radius, outer_radius, dr) #Set up radius grid to interpolate over
    #     with PdfPages(fname) as pdf: #Set up saving a PDF
    #         for i, x in enumerate(self.depth):
    #         #for x in arange(0., max(self.depth)-xstep, xstep): #Loop through each slice of the model up to the maximum depth
    #            # stop()
    #             #f = self.integrate_slab(x, x+xstep, dr=xstep/1e3) #Integrate up flux in each slice from line emissivities
    #             #dr = xstep / 1e3
    #             #r = arange(x, x+xstep, dr) #Set up radius grid to interpolate over
    #             #f = nansum(interp_emissivity(r)*dr, axis=1) #Sum up all line fluxes
    #             h.read_model(self.labels, self.flux[:,i]) #Read fluxes from slices into H2 object
    #             h.calculate_column_density() #Calcualte column density for H2 rovibrational lines
    #             h.v_plot(s2n_cut=-1.0, show_labels=True, savepdf=False, show_legend=False, y_range=y_range)
    #             title('Slice = ' + str(x) + ' cm') #Set title show show what depth we are at in the model
    #             pdf.savefig() #Output page of pdf




					

#Class to store information on H2 transition, with flux can calculate column density
class h2_transitions:
	def __init__(self, J, V, E, A):
		s = argsort(E.u) #Sort all arrays by energy for easy plotting later
		J.sort(s)
		V.sort(s)
		E.sort(s)
		A = A[s]
		n_lines = len(A) #Number of lines
		self.n_lines = n_lines
		self.J = J #Read in J object to store upper and lower J states
		self.V = V #Read in V object to store upper and lower V states
		self.E = E #Read in E object to store energy of upper state
		self.A = A #Read in Einstein A coeff. (transition probability) of line
		self.F = zeros(n_lines) #Set initial flux of line
		self.N = zeros(n_lines) #Set initial column density for each line
		self.Nsigma = zeros(n_lines) #Set uncertainity in the column demnsity for each line
		self.sigma = zeros(n_lines) #Set initial sigma (uncertainity) for each line
		self.s2n = zeros(n_lines) #Set initial signal-to-noise ratio for each line
		self.label = self.makelabel() #Make label of spectroscpic notation
		g_ortho_para = 1 + 2 * (J.u % 2 == 1) #Calculate the degenearcy for ortho or para hydrogen
		self.g = g_ortho_para * (2*J.u+1) #Store degeneracy
		self.T = E.u / k #Store "temperature" of the energy of the upper state
		self.wave = E.getwave() #Store wavelength of transitions
		self.path = '' #Store path for saving excitation diagram and other files, read in when reading in region with definition set_Flux
		self.rot_T = zeros(n_lines)  #Store rotation temperature from fit
		self.model_ratio = zeros(n_lines) #store ratio to model, if a model fit is performed
		self.sig_rot_T = zeros(n_lines) #Store uncertainity in rotation temperature fit
		self.res_rot_T =  zeros(n_lines) #Store residuals from offset of line fitting rotation temp
		self.sig_res_rot_T =  zeros(n_lines) #Store uncertainity in residuals from fitting rotation temp (e.g. using covariance matrix)
	def tin(self, v, J): #Find and return indicies of transitions into a given level defined by v and J
		return where((self.V.l == v) & (self.J.l == J))
	def tout(self, v, J): #FInd and return indicies of transitions out of a given level defined by v and J(self.V.u == v) & (self.J.u == J)
		return where((self.V.u == v) & (self.J.u == J))
	def calculate_column_density(self, normalize=True): #Calculate the column density and uncertainity for a line's given upper state from the flux and appropriate constants
		##self.N = self.F / (self.g * self.E.u * h * c * self.A)
		##self.Nsigma = self.sigma /  (self.g * self.E.u * h * c * self.A)
		#self.N = self.F / (self.g * self.E.diff() * h * c * self.A)
		#self.Nsigma = self.sigma /  (self.g * self.E.diff() * h * c * self.A)
		self.N = self.F / (self.E.diff() * h * c * self.A)
		self.Nsigma = self.sigma /  (self.E.diff() * h * c * self.A)
		if normalize: #By default normalize to the 1-0 S(1) line, set normalize = False if using absolute flux calibrated data
			self.normalize()
			#N_10_S1 = self.N[self.label == '1-0 S(1)'] #Grab column density derived from 1-0 S(1) line
			#self.N = self.N / N_10_S1 #Normalize column densities
			#self.Nsigma = self.Nsigma / N_10_S1 #Normalize uncertainity
	def calculate_flux(self): #Calculate flux for a given calculated column density (ie. if you set it to thermalize)
		#self.F = self.N * self.g * self.E.diff() * h * c * self.A
		self.F = self.N * self.E.diff() * h * c * self.A
	def generate_synthetic_spectrum(self, wave_range=[1.45,2.45], pixel_size=1e-5, line_fwhm=7.5, centroid=0.): #Generate a synthetic 1D spectrum based on stored flux values in this object, can be used to synthesize spectra from Cloudy models, or thermal gas generated by the "thermalize" command
		#n_pixels = (wave_range[1] - wave_range[0])/pixel_size #Calcualte number of pixels in 1D sythetic spectrum
		#velocity_grid = arange(-500,500,0.01) #Create velocity grid
		c_km_s = c / 1e5 #Get speed of light in km/s
		sigma = line_fwhm / 2.0*sqrt(2.0*log(2.0)) #Convert FWHM into sigma for a gaussian
		alpha = 2.0*sigma**2 #Calcualte alpha for gaussian
		beta =  (1.0/sqrt(pi*alpha)) #Calculate another part (here called beta) for the gaussian profile
		#line_profile =  beta *  exp(-((velocity_grid-centroid)**2/(alpha))) #Calculate normalizeable line profile in velocity space
		wave = arange(wave_range[0], wave_range[1], pixel_size) #Create wavelength array for 1D synthetic spectrum
		flux = zeros(len(wave)) #Create flux array for 1D synthetic spectrum
		for i in range(len(self.wave)):
			current_wavelength = self.wave[i]
			if (current_wavelength > wave_range[0]) and (current_wavelength < wave_range[1]):
				#Interpolate line profile into wavelength space
				#velocity_grid = c_km_s * ((wave/current_wavelength) - 1.0) #Create velocity grid from wavelength grid
				line_profile =  beta *  exp(-((c_km_s * ((wave/current_wavelength) - 1.0)-centroid)**2/(alpha))) #Calculate gaussian line profile in wavelength space
				flux = flux + self.F[i]*line_profile #Build up line on flux grid
		return wave, flux #Return wavlelength and flux grids
	def normalize(self, label='5-3 O(3)'):
		i = self.label == label
		if self.N[i] > 0.: #Check if line even exists
			normalize_by_this = self.N[i] / self.g[i]#Grab column density of line to normalize by
			self.N /= normalize_by_this #Do the normalization
			self.Nsigma /= normalize_by_this #Ditto on the uncertainity
		else:
			print("ERROR: Attempted to normalize by the " + label + " line, but it appears to not exist.  No normalization done.  Try a different line?")
	def thermalize(self, temperature): #Set all column densities to be thermalized at the specified temperature, normalized to the 1-0 S(1) line
		exponential = self.g * exp(-self.T/temperature) #Calculate boltzmann distribution for user given temperature, used to populate energy levels
		boltzmann_distribution = exponential / nansum(exponential) #Create a normalized boltzmann distribution
		self.N = boltzmann_distribution #Set column densities to the boltzmann distribution
		#self.normalize() #Normalize to the 1-0 S(1) line
		self.calculate_flux() #Calculate flux of new lines after thermalization
	def makelabel(self): #Make labels for each transition in spectroscopic notation.
		labels = []
		for i in range(self.n_lines):
			labels.append(self.V.label[i] + ' ' + self.J.label[i])
		return array(labels)
	def intrinsic_ratio(self, line_label_1, line_label_2): #Return the intrinsic flux ratio of two transitions that arise from the same upper state
		line_1 = self.label == line_label_1 #Find index to transition 1
		line_2 = self.label == line_label_2 #Find index to transition 2
		if (self.V.u[line_1] != self.V.u[line_2]) or (self.J.u[line_1] != self.J.u[line_2]): #Test if both transitions came from the upper state and catch error if not
			print("ERROR: Both of these transitions do not arise from the same upper state.")
			return(0.0) #Return 0 if the transitions do not arise from the same upper state
		ratio = (self.E.diff()[line_1] * self.A[line_1]) / (self.E.diff()[line_2] * self.A[line_2]) #Calculate intrinsic ratio of the two transitions
		return(ratio) #return the intrinsic ratio
	def flux_ratio(self, line_label_1, line_label_2, sigma=False): #Return flux ratio of any two lines
		line_1 = self.label == line_label_1 #Find index to transition 1
		line_2 = self.label == line_label_2 #Find index to transition 2
		ratio = self.F[line_1] / self.F[line_2]
		if sigma: #If user specifies they want the uncertainity returned, return the ratio, and the uncertainity
			uncertainity = sqrt(ratio**2 * ((self.sigma[line_1]/self.F[line_1])**2 + (self.sigma[line_2]/self.F[line_2])**2) )
			return ratio, uncertainity
		else:
			return self.F[line_1] / self.F[line_2]
		return ratio #Return observed line flux ratio only (no uncertainity)
	def upper_state(self, label, wave_range = [0,999999999.0]): #Given a label in spectroscopic notation, list transitions with same upper state (and a wavelength range if specified)
		i = self.label == label
		Ju = self.J.u[i]
		Vu = self.V.u[i]
		found_transitions = (self.wave > wave_range[0]) & (self.wave < wave_range[1]) & (self.J.u == Ju) & (self.V.u == Vu)
		label_subset = self.label[found_transitions]
		wave_subset = self.wave[found_transitions]
		for i in range(len(label_subset)):
			print(label_subset[i] + '\t' + str(wave_subset[i]))
		#print(self.label[found_transitions])#Find all matching transitions in the specified wavelength range with a matching upper J and V state
		#print(self.wave[found_transitions])
	def set_flux(self, region): #Set the flux of a single line or multiple lines given the label for it, e.g. h2.set_flux('1-0 S(1)', 456.156)
		self.path = region.path #Set path to 
		n = len(region.label)
		if n == 1: #If only a single line
			matched_line = (self.label == region.label)
			if any(matched_line): #If any matches are found...
				self.F[matched_line] = region.flux #Set flux for a single line
				self.s2n[matched_line] = region.s2n #Set S/N for a single line
				self.sigma[matched_line] = region.sigma #Set sigma (uncertainity) for a single line
		else: #if multiple lines
			for i in range(n): #Loop through each line\
				matched_line = (self.label == region.label[i])
				if any(matched_line): #If any matches are found...
					self.F[matched_line] = region.flux[i] #And set flux
					self.s2n[matched_line] = region.s2n[i] #Set S/N for a single line
					self.sigma[matched_line] = region.sigma[i] #Set sigma (uncertainity) for a single line
	def read_model(self, labels, flux): #Read in fluxes from model
		for i in range(len(labels)): #Loop through each line
			matched_line = (self.label == labels[i]) #Match to H2 line
			self.F[matched_line] = flux[i] #Set flux to flux from model
	def read_data(self, labels, flux, sigma):
		for i in range(len(labels)): #Loop through each line
			matched_line = (self.label == labels[i]) #Match to H2 line
			self.F[matched_line] = flux[i] #Set flux to flux from data
			self.sigma[matched_line] = sigma[i] #Set uncertainity to uncertainity from data
			self.s2n[matched_line] = flux[i] / sigma[i] #Calculate S/N
	def quick_plot(self): #Create quick boltzmann diagram for previewing and testing purposes
		nonzero = self.N != 0.0
		clf()
		plot(self.T[nonzero], log(self.N[nonzero]), 'o')
		ylabel("Column Density   log$_e$(N/g) [cm$^{-2}$]", fontsize=18)
		xlabel("Excitation Energy     (E/k)     [K]", fontsize=18)
		show()
	def make_latex_table(self, output_filename, s2n_cut = 3.0, normalize_to='5-3 O(3)'): #Output a latex table of column densities for each H2 line
		lines = []
		#lines.append(r"\begin{table}")  #Set up table header
		lines.append(r"\begin{longtable}{llrrrrr}")
		lines.append(r"\caption{\htwo{} rovibrational state column densities}{} \label{tab:coldens} \\")
		#lines.append("\begin{scriptsize}")
		#lines.append(r"\begin{tabular}{cccc}")
		lines.append(r"\hline")
		lines.append(r"$\lambda_{\mbox{\tiny vacuum}}$ & \htwo{} line ID & $v_u$ & $J_u$ & $E_u/k$ & $\log_{10}\left(A_{ul}\right)$ & $\ln \left(N_u/g_u\right) - \ln\left(N_{\mbox{\tiny "+normalize_to+r"}}/g_{\mbox{\tiny "+normalize_to+r"}}\right)$ \\")
		lines.append(r"\hline\hline")
		lines.append(r"\endfirsthead")
		lines.append(r"\hline")
		lines.append(r"$\lambda_{\mbox{\tiny vacuum}}$ & \htwo{} line ID & $v_u$ & $J_u$ & $E_u/k$ & $\log_{10}\left(A_{ul}\right)$ & $\ln \left(N_u/g_u\right) - \ln\left(N_{\mbox{\tiny "+normalize_to+r"}}/g_{\mbox{\tiny "+normalize_to+r"}}\right)$ \\")
		lines.append(r"\hline\hline")
		lines.append(r"\endhead")
		lines.append(r"\hline")
		lines.append(r"\endfoot")
		lines.append(r"\hline")
		lines.append(r"\endlastfoot")
		if any(self.V.u[self.s2n > s2n_cut]): #Error catching
			highest_v = max(self.V.u[self.s2n > s2n_cut]) #Find highest V level
			for v in range(1,highest_v+1): #Loop through each rotation ladder
				i = (self.V.u == v) & (self.s2n > s2n_cut) #Find all lines in the current ladder
				s = argsort(self.J.u[i]) #Sort by upper J level
				labels = self.label[i][s] #Grab line labels
				J =  self.J.u[i][s] #Grab upper J
				N = self.N[i][s] / self.g[i][s] #Grab column density N/g
				E = self.T[i][s]
				A = self.A[i][s]
				wave = self.wave[i][s]
				sig_N =  self.Nsigma[i][s] / self.g[i][s] #Grab uncertainity in N
				for j in range(len(labels)):
					#lines.append(labels[j] + " & " + str(v) + " & " + str(J[j]) + " & " + "%1.2e" % N[j] + " $\pm$ " + "%1.2e" %  sig_N[j] + r" \\") 
					lines.append(r"%1.6f" % wave[j] + " &  " + labels[j] + " & " + str(v) + " & " + str(J[j]) + " & %5.0f" % E[j] + " & %1.2f" %  log10(A[j]) +  
						 " & $" + "%1.2f" % log(N[j]) + r"^{+%1.2f" % (-log(N[j]) + log(N[j]+sig_N[j]))   +r"}_{%1.2f" % (-log(N[j]) + log(N[j]-sig_N[j])) +r"} $ \\") 
		#lines.append(r"\hline\hline")
		#lines.append(r"\end{tabular}")
		lines.append(r"\end{longtable}")
		#lines.append(r"\end{table}")
		savetxt(output_filename, lines, fmt="%s") #Output table
	def save_table(self): #Output ascii table with data for making an excitation diagram
		lines = [] #Set up array for saving lines for text file
		lines.append('#H2 Line\twavelength [um]\tortho/para\tv_u\tJ_u\tE_u\tlog(N/g)-log(N/g)_1-0S(1)\t+sigma\t-sigma') #Header of text file listing all the columns
		if any(self.V.u[self.s2n > 0.0]): #Error catching
			highest_v = max(self.V.u[self.N > 0.0]) #Find highest V level
			ortho_para = ['para' ,'ortho']
			for v in range(1,highest_v+1): #Loop through each rotation ladder
				i = (self.V.u == v) & (self.N > 0.0)  #Find all lines in the current ladder
				s = argsort(self.J.u[i]) #Sort by upper J level
				labels = self.label[i][s] #Grab line labels
				J =  self.J.u[i][s] #Grab upper J
				N = self.N[i][s] / self.g[i][s] #Grab column density N\
				E = self.T[i][s]
				sig_N =  self.Nsigma[i][s] / self.g[i][s] #Grab uncertainity in N
				wave = self.wave[i][s] #Grab wavelength of line
				for j in range(len(labels)): #Loop through each rotation ladder
					lines.append(labels[j] + '\t%1.5f' % wave[j] + '\t' + ortho_para[J[j]%2] + '\t' + str(v) + '\t' + str(J[j])+ '\t%1.1f' % E[j] + 
						 '\t%1.3f'  % log(N[j]) + '\t%1.3f' %  (-log(N[j]) + log(N[j]+sig_N[j])) + '\t%1.3f' % (-log(N[j]) + log(N[j]-sig_N[j])) )
		savetxt(self.path + '_H2_column_densities.dat', lines, fmt="%s") #Output table
	def fit_rot_temp(self, T, log_N, y_error_bars, s2n_cut = 1., color='black', dotted_line=False, rot_temp_energy_limit=0., show=True): #Fit rotation temperature to a given ladder in vibration
		log_N_sigma = nanmax(y_error_bars, 0) #Get largest error in log space
		if rot_temp_energy_limit > 0.: #If user specifies to cut rotation temp fit, use that....
			usepts = (T < rot_temp_energy_limit) & isfinite(log_N) 
			print('debug time! Log_N[usepts]=', log_N[usepts])
			fit, cov = curve_fit(linear_function, T[usepts], log_N[usepts], sigma=log_N_sigma[usepts], absolute_sigma=False) #Do weighted linear regression fit
		else: #Else fit all points
			fit, cov = curve_fit(linear_function, T, log_N, sigma=log_N_sigma, absolute_sigma=False) #Do weighted linear regression fit
		slope = fit[0]#[0]
		sigma_slope = sqrt(abs(cov[0,0]))
		if dotted_line:
			linestyle=':'
		else:
			linestyle='-'
		#y = polyval(fit, T) #Get y positions of rotation temperature fit
		y = linear_function(T, fit[0], fit[1]) #Get y positions of rotation temperature fit
		y_sigma = sqrt(cov[0,0]*T**2 + 2.0*cov[0,1]*T + cov[1,1]) #Grab uncertainity in fit for a given y value and the covariance matrix, see Pg 125 of the Math Methods notes
		if show: #If user wants to plot lines
			plot(T, y, color=color, linestyle=linestyle) #Plot T rot fit
		#plot(T, y+y_sigma, color=color, linestyle='--') #Plot uncertainity in T rot fit
		#plot(T, y-y_sigma, color=color, linestyle='--')
		rot_temp = -1.0/slope #Calculate the rotation taemperature
		sigma_rot_temp = rot_temp * (sigma_slope/abs(slope)) #Calculate uncertainity in rotation temp., basically just scale fractional error
		print('rot_temp = ', rot_temp,'+/-',sigma_rot_temp)
		#residuals = e**log_N - e**y #Calculate residuals in fit, but put back in linear space
		#sigma_residuals = sqrt( (e**(y + y_sigma) - e**y)**2 + (e**(log_N + log_N_sigma)-e**log_N)**2 ) #Calculate uncertainity in residuals from adding uncertainity in fit and data points together in quadarature
		residuals = e**(log_N-y)
		sigma_residuals = sqrt(log_N_sigma**2 + y_sigma**2)
		return rot_temp, sigma_rot_temp, residuals, sigma_residuals
	def compare_model(self, h2_model_input, name='compare_model_excitation_diagrams', figsize=[17.0,13], x_range=[0.0,55000.0], y_range=array([-6.25,5.5]), ratio_y_range=[1e-1,1e1],
		plot_residual_temp=False, residual_temp=default_single_temp, residual_temp_y_intercept=default_single_temp_y_intercept, multi_temp_fit=False,
		take_ratio=False, s2n_cut=3.0, makeplot=True): #Make a Boltzmann diagram comparing a model (ie. Cloudy) to data, and show residuals, show even and odd vibration states for clarity
		fname = self.path + '_'+name+'.pdf'
		h2_model = copy.deepcopy(h2_model_input) #Copy h2 model obj so not to modify the original
		show_these_v  = [] #Set up a blank vibration array to automatically fill 
		for v in range(14): #Loop through and check each set of states of constant v
			in_this_v = self.V.u == v
			if any(self.s2n[self.V.u == v] >= s2n_cut): #If anything is found to be plotted in the data
				show_these_v.append(v) #store this vibration state for later plotting
				max_J = max(self.J.u[in_this_v & (self.s2n >= s2n_cut)])
				if max_J > 6: #If data probes in this rotation ladder beyond J of six`
					h2_model.N[in_this_v & (self.J.u > max_J+1)] = 0.  #Blank out model where > J + 1 max
				else:
					h2_model.N[in_this_v & (self.J.u > 7)] = 0.
			else:
				h2_model.N[in_this_v] = 0. #Blank out model if no datapoints are in this rotation ladder
		self.model_ratio = self.N / h2_model.N #Calulate and store ratio of data/model for later use to make tables or whatever the user wants to script up
		ratio = copy.deepcopy(self)
		if take_ratio: #If user actually wants to take a ratio
			ratio.N = (self.N / h2_model.N) #Take a ratio, note we are multiplying by the degeneracy
			ratio.Nsigma = self.Nsigma  /  h2_model.N
			chi_sq = nansum(log10(ratio.N[ratio.s2n > s2n_cut])**2) #Calculate chisq from ratios
			print('Compare model for ' + name + ' sum(log10(ratios)**2) = ', chi_sq)
		else: #If user doesn ot specifiy acutally taking a ratio
			ratio.N = self.N - h2_model.N
			ratio.Nsigma = self.Nsigma
		#ratio.Nsigma = (self.Nsigma /h2_model.N)
		if makeplot:
			with PdfPages(fname) as pdf: #Make a pdf
				### Set up subplotting
				subplots(2, sharex="col") #Set all plots to share the same x axis
				tight_layout(rect=[0.03, 0.00, 1.0, 1.0]) #Try filling in white space
				fig = gcf()#Adjust aspect ratio
				fig.set_size_inches(figsize) #Adjust aspect ratio
				subplots_adjust(hspace=0.037, wspace=0) #Set all plots to have no space between them vertically
				gs = GridSpec(2, 1, height_ratios=[1, 1]) #Set up grid for unequal sized subplots
				### Left side
				subplot(gs[0])
				h2_model.v_plot(V=show_these_v, orthopara_fill=False, empty_fill=True, show_legend=False, savepdf=False, show_labels=False, line=True,y_range=y_range, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=True) #Plot model points as empty symbols
				self.v_plot(V=show_these_v, orthopara_fill=False, full_fill=True, show_legend=True, savepdf=False, y_range=y_range, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=False, s2n_cut=s2n_cut)
				ylabel("Column Density   ln(N$_u$/g$_u$)-ln(N$_{r}$/g$_{r}$)", fontsize=18)
				V = range(1,14)
				frame = gca() #Turn off axis number labels
				setp(frame.get_xticklabels(), visible=False)
				#subplot(gs[3])
				subplot(gs[1])
				plot([0,100000],[1,1], linestyle='--', color='gray')
				ratio.v_plot(V=show_these_v, orthopara_fill=False, full_fill=True,  show_legend=False, savepdf=False, no_zero_x=True, x_range=x_range,  clear=False, show_axis_labels=False, no_legend_label=True,
					plot_single_temp=plot_residual_temp, single_temp=residual_temp, single_temp_y_intercept=residual_temp_y_intercept, multi_temp_fit=multi_temp_fit, show_ratio=True, s2n_cut=s2n_cut, y_range=ratio_y_range)
				if take_ratio:
					#ylabel("Data/Model ratio  ln(N$_u$/g$_u$)-ln(N$_{m}$/g$_{m}$)", fontsize=18)
					ylabel("Data/Model Ratio", fontsize=18)
				else:
					ylabel("Data-Model  ln((N$_u$-$N_m$)/g$_u$)-ln(N$_{r}$/g$_{r}$)", fontsize=18)
				xlabel("Excitation Energy     (E$_u$/k)     [K]", fontsize=18)
				pdf.savefig()
		return(chi_sq) #Return chisq value to quantify the goodness of fit
	def v_plot_with_model(self, h2_model_input, x_range=[0.0,55000.0], y_range=array([-6.25,5.5]), s2n_cut=3.0, **args): #Do a vplot with a model overlayed, a simple form of def compare_model for making multipaneled plots and things with your own scripts
		h2_model = copy.deepcopy(h2_model_input) #Copy h2 model obj so not to modify the original
		show_these_v  = [] #Set up a blank vibration array to automatically fill 
		for v in range(14): #Loop through and check each set of states of constant v
			in_this_v = self.V.u == v
			if any(self.s2n[self.V.u == v] >= s2n_cut): #If anything is found to be plotted in the data
				show_these_v.append(v) #store this vibration state for later plotting
				max_J = max(self.J.u[in_this_v & (self.s2n >= s2n_cut)])
				if max_J > 6: #If data probes in this rotation ladder beyond J of six`
					h2_model.N[in_this_v & (self.J.u > max_J+1)] = 0.  #Blank out model where > J + 1 max
				else:
					h2_model.N[in_this_v & (self.J.u > 7)] = 0.
			else:
				h2_model.N[in_this_v] = 0.  #Blank out model if no datapoints are in this rotation ladder
		#tight_layout(rect=[0.03, 0.00, 1.0, 1.0]) #Try filling in white space
		h2_model.v_plot(V=show_these_v, orthopara_fill=False, empty_fill=True, show_legend=False, savepdf=False, show_labels=False, line=True,y_range=y_range, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=True) #Plot model points as lines
		self.v_plot(V=show_these_v, orthopara_fill=False, full_fill=True, show_legend=False, savepdf=False, y_range=y_range, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=False, s2n_cut=s2n_cut, **args)
		#ylabel("Column Density   ln(N$_u$/g$_u$)-ln(N$_{r}$/g$_{r}$)", fontsize=18)
	def v_plot_ratio_with_model(self, h2_model, x_range=[0.0,55000.0], y_range=array([1e-1,1e1]), s2n_cut=3.0, y_label=r'N$_{obs}$/N$_{model}$', **args):
		show_these_v  = [] #Set up a blank vibration array to automatically fill 
		for v in range(14): #Loop through and check each set of states of constant v
			if any(self.s2n[self.V.u == v] >= s2n_cut): #If anything is found to be plotted in the data
				show_these_v.append(v) #store this vibration state for later plotting
		self.model_ratio = self.N / h2_model.N #Calulate and store ratio of data/model for later use to make tables or whatever the user wants to script up
		ratio = copy.deepcopy(self)
		ratio.N = (self.N / h2_model.N) #Take a ratio, note we are multiplying by the degeneracy
		ratio.Nsigma = self.Nsigma  /  h2_model.N
		#chi_sq = nansum(log10(ratio.N[ratio.s2n > s2n_cut])**2) #Calculate chisq from ratios
		plot([0,100000],[1,1], linestyle='--', color='gray')
		ratio.v_plot(V=show_these_v, orthopara_fill=False, full_fill=True,  show_legend=False, savepdf=False, no_zero_x=True, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=False,
					show_ratio=True, s2n_cut=s2n_cut, y_range=y_range, **args)
		#ylabel(y_label, fontsize=18)
		#xlabel("Excitation Energy     (E$_u$/k)     [K]", fontsize=18)
	def plot_individual_ladders(self, x_range=[0.,0.0], s2n_cut = 0.0): #Plot set of individual ladders in the excitation diagram
		fname = self.path + '_invidual_ladders_excitation_diagrams.pdf'
		with PdfPages(fname) as pdf: #Make a pdf
			V = range(0,14)
			for i in V:
				if any((self.V.u == i) & isfinite(self.N) & (self.N > 0.0)):
					self.v_plot(V=[i], show_upper_limits=False, show_labels=True, rot_temp=False, show_legend=True, savepdf=False, s2n_cut=s2n_cut, no_zero_x=True)
					pdf.savefig()
	def plot_rot_temp_fit(self, s2n_cut = 3.0, figsize=[21.0,15], x_range=[0.0,55000.0], y_range=array([-5.25,15.25])): #Fit and plot rotation temperatures then show their residuals
		fname = self.path + '_rotation_temperature_fits_and_residuals_all.pdf' #Set filename
		with PdfPages(fname) as pdf: #Make a pdf
			### Set up subplotting
			subplots(2, sharex="col") #Set all plots to share the same x axis
			tight_layout(rect=[0.03, 0.00, 1.0, 1.0]) #Try filling in white space
			fig = gcf()#Adjust aspect ratio
			fig.set_size_inches(figsize) #Adjust aspect ratio
			subplots_adjust(hspace=0, wspace=0) #Set all plots to have no space between them vertically
			gs = GridSpec(2, 1, height_ratios=[1, 1]) #Set up grid for unequal sized subplots
			### Left side
			subplot(gs[0])
			V = range(0,15)
			self.v_plot(V=V, orthopara_fill=False, full_fill=True, show_legend=True, savepdf=False, y_range=y_range, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=False,
					rot_temp=True, rot_temp_residuals=False, s2n_cut=s2n_cut)
			ylabel("Column Density   ln(N$_u$/g$_u$)-ln(N$_{r}$/g$_{r}$)", fontsize=18)
			frame = gca() #Turn off axis number labels
			setp(frame.get_xticklabels(), visible=False)
			#subplot(gs[3])
			subplot(gs[1])
			self.v_plot(V=V, orthopara_fill=False, full_fill=True, show_legend=False, savepdf=False, y_range=y_range, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=False,
					rot_temp=False, rot_temp_residuals=True, s2n_cut=s2n_cut)
			ylabel("Column Density Ratio of Data to Model   ln(N$_u$/g$_u$)-ln(N$_{m}$/g$_{m}$)]", fontsize=18)
			xlabel("Excitation Energy     (E$_u$/k)     [K]", fontsize=18)
			pdf.savefig()
			### Middle
			# V=[1,3,5,7,9,11,13]
			# subplot(gs[1])
			# self.v_plot(V=V, orthopara_fill=False, full_fill=True, show_legend=True, savepdf=False, y_range=y_range, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=False,
			# 		rot_temp=True, rot_temp_residuals=False, s2n_cut=s2n_cut)			
			# frame = gca() #Turn off axis number labels
			# setp(frame.get_xticklabels(), visible=False)
			# setp(frame.get_yticklabels(), visible=False)
			# subplot(gs[4])
			# frame = gca() #Turn off axis number labels
			# setp(frame.get_yticklabels(), visible=False)
			# self.v_plot(V=V, orthopara_fill=False, full_fill=True, show_legend=False, savepdf=False, y_range=y_range, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=False,
			# 		rot_temp=False, rot_temp_residuals=True, s2n_cut=s2n_cut)
			# xlabel("Excitation Energy     (E$_u$/k)     [K]", fontsize=18)
			# ### Right side
			# V=[0,2,4,6,8,10,12,14]
			# subplot(gs[2])
			# self.v_plot(V=V, orthopara_fill=False, full_fill=True, show_legend=True, savepdf=False, y_range=y_range, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=False,
			# 		rot_temp=True, rot_temp_residuals=False, s2n_cut=s2n_cut)			
			# frame = gca() #Turn off axis number labels
			# setp(frame.get_xticklabels(), visible=False)
			# setp(frame.get_yticklabels(), visible=False)
			# subplot(gs[5])
			# frame = gca() #Turn off axis number labels
			# setp(frame.get_yticklabels(), visible=False)
			# self.v_plot(V=V, orthopara_fill=False, full_fill=True, show_legend=False, savepdf=False, y_range=y_range, x_range=x_range, clear=False, show_axis_labels=False, no_legend_label=False,
			# 		rot_temp=False, rot_temp_residuals=True, s2n_cut=s2n_cut)
			# xlabel("Excitation Energy     (E$_u$/k)     [K]", fontsize=18)
			#pdf.savefig()

		#OLD VERSION
		# with PdfPages(fname) as pdf: #Make a pdf
		# 	self.v_plot(V=V, show_labels=False, rot_temp=True, rot_temp_residuals=False, savepdf=False, s2n_cut=s2n_cut)
		# 	pdf.savefig()
		# for i in V:
		# 	fname = self.path + '_rotation_temperature_fits_V'+str(i)+'.pdf'
		# 	with PdfPages(fname) as pdf: #Make a pdf
		# 		title('V = '+str(i))
		# 		self.v_plot(V=[i], show_labels=True, rot_temp=True, rot_temp_residuals=False, savepdf=False, s2n_cut=s2n_cut, no_zero_x=True) #Plot single rotation ladder + rot temp fit
		# 		pdf.savefig()
		# 	fname = self.path + '_rotation_temperature_residuals_V'+str(i)+'.pdf'
		# 	with PdfPages(fname) as pdf: #Make a pdf
		# 		title('V = '+str(i)+' residuals')
		# 		self.v_plot(V=[i], show_labels=True, rot_temp=False, rot_temp_residuals=True, savepdf=False, s2n_cut=s2n_cut, ignore_x_range=True, show_legend=False) #Plot residuals
		# 		pdf.savefig()
	# WORK IN PROGRESS, NEED TO ALLOW FITTING OF VIBRATIONAL TEMPERATURES
	# def plot_vib_temp_fit(self, s2n_cut = 3.0, V = range(0,14)): #Fit and plot rotation temperatures then show their residuals
	# 	fname = self.path + '_rotation_temperature_fits_and_residuals_all.pdf' #Set filename
	# 	with PdfPages(fname) as pdf: #Make a pdf
	# 		self.v_plot(V=V, show_labels=False, rot_temp=True, rot_temp_residuals=False, savepdf=False, s2n_cut=s2n_cut)
	# 		pdf.savefig()
	# 	for i in V:
	# 		fname = self.path + '_rotation_temperature_fits_V'+str(i)+'.pdf'
	# 		with PdfPages(fname) as pdf: #Make a pdf
	# 			title('V = '+str(i))
	# 			self.v_plot(V=[i], show_labels=True, rot_temp=True, rot_temp_residuals=False, savepdf=False, s2n_cut=s2n_cut, no_zero_x=True) #Plot single rotation ladder + rot temp fit
	# 			pdf.savefig()
	# 		fname = self.path + '_rotation_temperature_residuals_V'+str(i)+'.pdf'
	# 		with PdfPages(fname) as pdf: #Make a pdf
	# 			title('V = '+str(i)+' residuals')
	# 			self.v_plot(V=[i], show_labels=True, rot_temp=False, rot_temp_residuals=True, savepdf=False, s2n_cut=s2n_cut, no_zero_x=True, show_legend=False) #Plot residuals
	# 			pdf.savefig()
	#Make simple plot first showing all the different rotational ladders for a constant V
	def v_plot(self, plot_single_temp = False, show_upper_limits = False, nocolor = False, V=[-1], s2n_cut=-1.0, normalize=True, savepdf=False, orthopara_fill=True, 
		empty_fill =False, full_fill=False, show_labels=False, x_range=[0.,0.], y_range=[0.,0.], rot_temp=False, show_legend=True, rot_temp_energy_limit=100000., 
		rot_temp_residuals=False, fname='', clear=True, legend_fontsize=14, line=False, subtract_single_temp = False, single_temp=default_single_temp, no_legend_label=False,
		single_temp_y_intercept=default_single_temp_y_intercept, no_zero_x = False, show_axis_labels=True, ignore_x_range=False, label_J=False, label_V=False, multi_temp_fit=False, single_temp_fit=False,
		single_color='none', show_ratio=False, symbsize = 9, single_temp_use_sigma=False):
		if fname == '':
			fname=self.path + '_excitation_diagram.pdf'
		with PdfPages(fname) as pdf: #Make a pdf
			nonzero = self.N != 0.0
			if clear: #User can specify if they want to clear the plot
				clf()
			labelsize = 18 #Size of text for labels
			if orthopara_fill:  #User can specify how they want symbols to be filled
				orthofill = 'full' #How symbols on excitation diagram are filled, 'full' vs 'none'
				parafill = 'none'
			elif empty_fill:
				orthofill = 'none' #How symbols on excitation diagram are filled, 'full' vs 'none'
				parafill = 'none'
			else:
				orthofill = 'full' #How symbols on excitation diagram are filled, 'full' vs 'none'
				parafill = 'full'
			if V == [-1]: #If user does not specify a specific set of V states to plot...
				use_upper_v_states = unique(self.V.u) #plot every one found
			else: #or else...
				use_upper_v_states = V #Plot upper V s tates specified by the user
			if subtract_single_temp: #If user wants to subtract the single temperature
				x = arange(0,200000, 10) #Set up an ax axis 
				interp_single_temp = interp1d(x, single_temp_y_intercept - (x / single_temp), kind='linear') #create interpolation object for the single temperature
				data_single_temp = interp_single_temp(self.T) #Create array of the single temperature for subtraction from the column density later on
				#log_N = log(self.N/self.g) - data_single_temp #Log of the column density
				log_N = log((self.N/self.g) - exp(data_single_temp))
				#plus_one_sigma = abs(log_N  + data_single_temp - log((self.N + self.Nsigma)/self.g) )
				#minus_one_sigma = abs(log_N + data_single_temp - log((self.N + self.Nsigma)/self.g) )
				#upper_limits = log(self.Nsigma*3.0/self.g) - data_single_temp
				plus_one_sigma = abs(log_N  - log((self.N + self.Nsigma)/self.g) )
				minus_one_sigma = abs(log_N - log((self.N - self.Nsigma)/self.g) )
				upper_limits = log((self.Nsigma*3.0/self.g) - exp(data_single_temp))
			elif show_ratio: #If user is plotting column density ratios, keep things in linear form and use linear axes on a log scale
				log_N = self.N #Not really log N but you get the idea
				log_N[log_N<=0.] = nan #Nan out zero and negative values, since they are essentialy meaningless anyway and won't plot on a log plot
				plus_one_sigma = self.Nsigma #Set error bars in linear space
				minus_one_sigma = self.Nsigma
				find_negative_sigma = log_N < minus_one_sigma #Find negative sigma
				minus_one_sigma[find_negative_sigma] = 1e-1 * log_N[find_negative_sigma] #Just make negative error bars 1/10 of data so it can still be plotted on log plot
				semilogy() #Set y axis to be semi log
			elif rot_temp_residuals: #If user has previously calculated rotation temperatures for each ladder, here they can show the residuals after subtracting the linear fits 
				log_N = log(self.res_rot_T)
				plus_one_sigma = abs(log_N  - log(self.res_rot_T + (self.Nsigma/self.g)))
				minus_one_sigma = abs(log_N - log(self.res_rot_T - (self.Nsigma/self.g)))
				upper_limits = log(self.Nsigma*3.0/self.g)
			else: #Default to simply plotting the column densities and their error bars
				log_N = log(self.N/self.g) #Log of the column density
				plus_one_sigma = abs(log_N  - log((self.N + self.Nsigma)/self.g) )
				minus_one_sigma = abs(log_N - log((self.N - self.Nsigma)/self.g) )
				upper_limits = log(self.Nsigma*3.0/self.g)
			#plus_one_sigma = abs(log_N - data_single_temp - log(self.N - exp(data_single_temp) + self.Nsigma)) #Upper 1 sigma errors in log space
			#minus_one_sigma = abs(log_N - data_single_temp  - log(self.N - exp(data_single_temp) - self.Nsigma)) #Lower 1 sigma errors in log space




			for i in use_upper_v_states:
				if single_color != 'none': #If user specifies a specific color, use that single color
					current_color = single_color
					current_symbol = 'o'
				elif nocolor: #If user specifies no color,
					current_color = 'gray'
					current_symbol = symbol_list[i]
				else: #Or else by default use colors from the color list defined at the top of the code
					current_color = color_list[i]
					current_symbol = 'o'
				if line: #if user specifies using lines
					#current_symbol = current_symbol + '-' #Draw a line between each symbol
					current_symbol = '-'
				data_found = (self.V.u == i) & (self.s2n > s2n_cut) & (self.N > 0.) #Search for data in this vibrational state
				#if any(data_found) and not show_ratio: #If any data is found in this vibrational state, add a line on the legend for this state
				if any(data_found): #If any data is found in this vibrational state, add a line on the legend for this state
					if no_legend_label:
						use_label = '_nolegend_'
					else:
						use_label = ' '
					errorbar([nan], [nan], yerr=1.0, fmt=current_symbol,  color=current_color, label=use_label, capthick=3, elinewidth=2, markersize=symbsize, fillstyle=orthofill)  #Do empty plot to fill legend
				ortho = (self.J.u % 2 == 1) &  (self.V.u == i) & (self.s2n > s2n_cut) & (self.N > 0.) #Select only states for ortho-H2, which has the proton spins aligned so J can only be odd (1,3,5...)
				ortho_upperlimit = (self.J.u % 2 == 1) &  (self.V.u == i) & (self.s2n <= s2n_cut) & (self.N > 0.)  #Select ortho-H2 lines where there is no detection (e.g. S/N <= 1)
				if any(ortho): #If datapoints are found...
					if nansum(self.s2n[ortho]) == 0.:
						plot(self.T[ortho], log_N[ortho], current_symbol,  color=current_color, markersize=symbsize, fillstyle=orthofill)  #Plot data + error bars
					else:
						y_error_bars = [minus_one_sigma[ortho], plus_one_sigma[ortho]] #Calculate upper and lower ends on error bars
						errorbar(self.T[ortho], log_N[ortho], yerr=y_error_bars, fmt=current_symbol,  color=current_color, capthick=3, elinewidth=2, markersize=symbsize, fillstyle=orthofill)  #Plot data + error bars
						if show_upper_limits:
							test = errorbar(self.T[ortho_upperlimit], upper_limits[ortho_upperlimit], yerr=1.0, fmt=current_symbol,  color=current_color, capthick=3, elinewidth=2,uplims=True, markersize=symbsize, fillstyle=orthofill) #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
					if show_labels: #If user wants to show labels for each of the lines
						for j in range(len(log_N[ortho])): #Loop through each point to label
							if  y_range[1] == 0 or (log_N[ortho][j] > y_range[0] and log_N[ortho][j] < y_range[1]): #check to make sure label is in plot y range
								text(self.T[ortho][j], log_N[ortho][j], '        '+self.label[ortho][j], fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black')  #Label line with text
					if label_J: #If user specifies labels for J
						for j in range(len(log_N[ortho])): #Loop through each point to label
							if  y_range[1] == 0 or (log_N[ortho][j] > y_range[0] and log_N[ortho][j] < y_range[1]): #check to make sure label is in plot y range
								text(self.T[ortho][j], log_N[ortho][j], '    '+str(self.J.u[ortho][j]), fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black')  #Label line with J upper level
					#print('For ortho v=', i)
					if rot_temp and len(log_N[ortho][isfinite(log_N[ortho])]) > 1: #If user specifies fit rotation temperature
						#stop()
						rt, srt, residuals, sigma_residuals = self.fit_rot_temp(self.T[ortho], log_N[ortho], y_error_bars, s2n_cut=s2n_cut, color=current_color, dotted_line=False, rot_temp_energy_limit=rot_temp_energy_limit) #Fit rotation temperature
						self.rot_T[ortho] = rt #Save rotation temperature for individual lines
						self.sig_rot_T[ortho] = srt #Save rotation tempreature uncertainity for individual lines
						self.res_rot_T[ortho] = residuals #Save residuals for individual data points from the rotation tmeperature fit
						self.sig_res_rot_T[ortho] = sigma_residuals #Save the uncertainity in the residuals from the rotation temp fit (point uncertainity and fit uncertainity added in quadrature)
			for i in use_upper_v_states:
				if single_color != 'none': #If user specifies a specific color, use that single color
					current_color = single_color
					current_symbol = '^'
				elif nocolor:
					current_color = 'Black'
					current_symbol = symbol_list[i]
				else: #Or else by default use colors from the color list defined at the top of the code
					current_color = color_list[i]
					current_symbol = '^'
				if line: #if user specifies using lines
					#current_symbol = current_symbol + ':' #Draw a line between each symbol
					current_symbol = ':'
				data_found = (self.V.u == i) & (self.s2n > s2n_cut) & (self.N > 0.) #Search for data in this vibrational state
				#if any(data_found) and not show_ratio: #If any data is found in this vibrational state, add a line on the legend for this state
				if any(data_found): #If any data is found in this vibrational state, add a line on the legend for this state
					if no_legend_label: #Check if user wants to use legend labes, if not ignore the label
						use_label = '_nolegend_'
					else:
						use_label = 'v='+str(i)
					errorbar([nan], [nan], yerr=1.0, fmt=current_symbol,  color=current_color, label=use_label, capthick=3, elinewidth=2, markersize=symbsize, fillstyle=parafill)  #Do empty plot to fill legend
				para = (self.J.u % 2 == 0) & (self.V.u == i) & (self.s2n > s2n_cut) & (self.N > 0.) #Select only states for para-H2, which has the proton spins anti-aligned so J can only be even (0,2,4,...)
				para_upperlimit =  (self.J.u % 2 == 0) & (self.V.u == i) & (self.s2n <= s2n_cut) & (self.N > 0.) #Select para-H2 lines where there is no detection (e.g. S/N <= 1)
				if any(para): #If datapoints are found...
					if nansum(self.s2n[para]) == 0.:
						plot(self.T[para], log_N[para], current_symbol,  color=current_color, markersize=symbsize, fillstyle=parafill)  #Plot data + error bars
					else:
						y_error_bars = [minus_one_sigma[para], plus_one_sigma[para]] #Calculate upper and lower ends on error bars
						#errorbar(self.T[para], log_N, yerr=y_error_bars, fmt=current_symbol,  color=current_color, label='v='+str(i), capthick=3, markersize=symbsize, fillstyle=parafill)  #Plot data + error bars
						errorbar(self.T[para], log_N[para], yerr=y_error_bars, fmt=current_symbol,  color=current_color, capthick=3, elinewidth=2,markersize=symbsize, fillstyle=parafill)  #Plot data + error bars
						if show_upper_limits:
							test = errorbar(self.T[para_upperlimit], upper_limits[para_upperlimit], yerr=1.0, fmt=current_symbol,  color=current_color, capthick=3, elinewidth=2, uplims=True, markersize=symbsize, fillstyle=parafill) #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
					if show_labels: #If user wants to show labels for each of the lines
						for j in range(len(log_N[para])): #Loop through each point to label
							if  y_range[1] == 0 or (log_N[para][j] > y_range[0] and log_N[para][j] < y_range[1]): #check to make sure label is in plot y range
								text(self.T[para][j], log_N[para][j], '        '+self.label[para][j], fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black')  #Label line with text
					if label_J: #If user specifies labels for J
						for j in range(len(log_N[para])): #Loop through each point to label
							if  y_range[1] == 0 or (log_N[para][j] > y_range[0] and log_N[para][j] < y_range[1]): #check to make sure label is in plot y range
								text(self.T[para][j], log_N[para][j], '    '+str(self.J.u[para][j]), fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black')  #Label line with J upper level
					#print('For para v=', i)
					if rot_temp and len(log_N[para][isfinite(log_N[para])]) > 1: #If user specifies fit rotation temperature
						rt, srt, residuals, sigma_residuals = self.fit_rot_temp(self.T[para], log_N[para], y_error_bars, s2n_cut=s2n_cut, color=current_color, dotted_line=True, rot_temp_energy_limit=rot_temp_energy_limit) #Fit rotation temperature
						self.rot_T[para] = rt #Save rotation temperature for individual lines
						self.sig_rot_T[para] = srt #Save rotation tempreature uncertainity for individual lines
						self.res_rot_T[para] = residuals #Save residuals for individual data points from the rotation tmeperature fit
						self.sig_res_rot_T[para] = sigma_residuals #Save the uncertainity in the residuals from the rotation temp fit (point uncertainity and fit uncertainity added in quadrature)						
					elif rot_temp and len(log_N[para][isfinite(log_N[para])]) <= 1:
						self.rot_T[para] = 0 #Save rotation temperature for individual lines
						self.sig_rot_T[para] = 0 #Save rotation tempreature uncertainity for individual lines
						self.res_rot_T[para] = ones(len(log_N[para])) #Save residuals for individual data points from the rotation tmeperature fit
						self.sig_res_rot_T[para] = self.Nsigma[para]/self.g[para] #Save the uncertainity in the residuals from the rotation temp fit (point uncertainity and fit uncertainity added in quadrature)						
			tick_params(labelsize=14) #Set tick mark label size
			if show_axis_labels: #By default print the axis labels, but the user can turn these off if so desired (replacing them with custom labels if needed)
				if normalize: #If normalizing to the 1-0 S(1) line
					ylabel("Column Density   ln(N$_u$/g$_u$)-ln(N$_{r}$/g$_{r}$)", fontsize=labelsize)
				else:  #If using absolute flux calibrated data
					ylabel("Column Density   ln(N$_u$/g$_u$) [cm$^{-2}$]", fontsize=labelsize)
				xlabel("Excitation Energy     (E$_u$/k)     [K]", fontsize=labelsize, labelpad=4)
			if x_range[1] == 0.0: #If user does not specifiy a range for the x-axis
				if any(self.T[self.s2n >= s2n_cut]) and not no_zero_x: #Catch error
					goodpix = (self.s2n >= s2n_cut) 
					xlim([0,1.4*max(self.T[goodpix])]) #Autoscale with left side of x set to zero
				elif any(self.T[self.s2n >= s2n_cut]) and no_zero_x: #If user does not want left side of x set to zero
					goodpix = (self.s2n >= s2n_cut) & (self.V.u == V[0])
					xlim([0.9*min(self.T[goodpix]), 1.1*max(self.T[goodpix])]) #Autoscale with left side of x not fixed at zero
				elif ignore_x_range:
					print('') #Do nothing, we are ignoring the xrange here
				else: #If no points are acutally found just set the limit here.
					xlim([0,70000.0])
			else: #Else if user specifies range
				xlim(x_range) #Use user specified range for x-axis
			if y_range[1] != 0.0: #If user specifies a y axis limit, use it
				ylim(y_range) #Use user specified y axis range
			if label_V: #Loop through and label every vibration level (rotation ladder), if the user sets label_V =  True
				for i in use_upper_v_states:
					if single_color != 'none': #If user specifies a specific color, use that single color
						current_color = single_color
					elif nocolor: #If user specifies no color,
						current_color = 'gray'
					else: #Or else by default use colors from the color list defined at the top of the code
						current_color = color_list[i]
					data_found = (self.V.u == i) & (self.s2n > s2n_cut) & (self.N > 0.) #Search for data in this vibrational state
					if sum(data_found) > 1: #Only add label if any data is found
						xposition = self.T[data_found][0]
						#yposition = log_N[data_found][0]
						text(xposition, y_range[1]*0.9, 'v='+str(i), fontsize=12, verticalalignment='top', horizontalalignment='left', color=current_color, rotation=90)  #Label line with J upper level
			if show_legend: #If user does not turn off showing the legend
				legend( ncol=2, fontsize=legend_fontsize, numpoints=1, columnspacing=-0.5, title = 'ortho  para  ladder', loc="upper right", bbox_to_anchor=(1,1))
			if plot_single_temp: #Plot a single temperature line for comparison, if specified
				x = arange(0,20000, 10)
				plot(x, single_temp_y_intercept - (x / single_temp), linewidth=2, color='orange')
				midpoint = size(x)/2
				text(0.7*x[int(midpoint)], 0.7*(single_temp_y_intercept - (x[int(midpoint)] / single_temp)), "T = "+str(single_temp)+" K", color='orange')
			if multi_temp_fit: #If user specifies they want to fit a multi temperature gas
				goodpix = (self.s2n > 5.0) & (self.N > 0.)
				x = self.T[goodpix]
				y = log(self.N[goodpix]/self.g[goodpix])
				vary_y_intercept = 7.0
				vary_temp = 3000.0
				vary_coeff = 0.6
				guess = array([15.0, 0.85, 0.15, 1e-8, 350.0, 650.0, 5500.0])
				upper_bound = guess + array([vary_y_intercept, vary_coeff, vary_coeff, vary_coeff, vary_temp, vary_temp, vary_temp])
				lower_bound = guess - array([vary_y_intercept, vary_coeff, vary_coeff, vary_coeff, vary_temp, vary_temp, vary_temp])
				fit, cov = curve_fit(multi_temp_func, x, y, guess, bounds=[lower_bound, upper_bound])
				b = fit[0]
				c = [fit[1], fit[2], fit[3]]
				T = [fit[4], fit[5], fit[6]]
				x = arange(0.0,70000.0,0.1)
				plot(x, b+log(c[0]*e**(-x/T[0]) + c[1]*e**(-x/T[1])+ c[2]*e**(-x/T[2])),'--', color='Black', linewidth=2)# + c[3]*e**(-x/T[3]) + c[4]*e**(-x/T[4]) + + c[5]*e**(-x/T[5])))
				print('Results from temperature fit to Boltzmann diagram data:')
				print('b = ', b)
				print('c = ', c)
				print('T = ', T)
			if single_temp_fit: #If user specifies they want to do a single temperature fit (ie. for shocks) 
				goodpix = (self.s2n > s2n_cut) & (self.N > 0.)
				x = self.T[goodpix]
				y = log(self.N[goodpix]/self.g[goodpix])
				sigma = plus_one_sigma[goodpix]
				vary_y_intercept = 7.0
				vary_temp = 3000.0
				guess = array([10.0, 1000.0])
				upper_bound = guess + array([vary_y_intercept, vary_temp])
				lower_bound = guess - array([vary_y_intercept, vary_temp])
				if single_temp_use_sigma: #If user specifies using the statistical 1 sigma uncertainity in the fit
					fit, cov = curve_fit(single_temp_func, x, y, guess, sigma, bounds=[lower_bound, upper_bound])
				else: #Don't use statistical sigma in fit
					fit, cov = curve_fit(single_temp_func, x, y, guess, bounds=[lower_bound, upper_bound])
				b, T = fit
				b_err, T_err = sqrt(diag(cov))
				x = arange(min(x)-1000.0,max(x)+1000.0,0.1)
				plot(x, b-x/T,'--', color='Black', linewidth=2)
				print('Results from temperature fit to Boltzmann diagram data:')
				print('b = ', b, ' +/- ', b_err)
				print('T = ', T, ' +/- ', T_err)
				#stop()
				self.model_ratio = self.N /  (self.g*exp(b-self.T/T)) #Calcualte and store ratio of 
			#show()
			draw()
			if savepdf:
				pdf.savefig() #Add in the pdf
			#stop()
			if single_temp_fit:
				return T, T_err
	#Plot 
	def rotation_plot(self, show_upper_limits = True, nocolor = False, V=[-1], s2n_cut=-1.0, normalize=True, savepdf=True, orthopara_fill=True, empty_fill =False, full_fill=False,
   				show_labels=False, x_range=[0.,0.], y_range=[0.,0.], show_legend=True, fname='', clear=True, legend_fontsize=14):
		if fname == '':
			fname = self.path + '_rotation_diagram.pdf'
		with PdfPages(fname) as pdf: #Make a pdf
			nonzero = self.N != 0.0
			if clear: #User can specify if they want to clear the plot
				clf()
			symbsize = 7 #Size of symbols on excitation diagram
			labelsize = 18 #Size of text for labels
			if orthopara_fill:  #User can specify how they want symbols to be filled
				orthofill = 'full' #How symbols on excitation diagram are filled, 'full' vs 'none'
				parafill = 'none'
			elif empty_fill:
				orthofill = 'none' #How symbols on excitation diagram are filled, 'full' vs 'none'
				parafill = 'none'
			else:
				orthofill = 'full' #How symbols on excitation diagram are filled, 'full' vs 'none'
				parafill = 'full'
			if V == [-1]: #If user does not specify a specific set of V states to plot...
				use_upper_v_states = unique(self.V.u) #plot every one found
			else: #or else...
				use_upper_v_states = V #Plot upper V s tates specified by the user
			for i in use_upper_v_states:
				if nocolor: #If user specifies no color,
					current_color = 'gray'
					current_symbol = symbol_list[i]
				else: #Or else by default use colors from the color list defined at the top of the code
					current_color = color_gradient[i]
					current_symbol = 'o'
				ortho = (self.J.u % 2 == 1) &  (self.V.u == i) & (self.s2n > s2n_cut) & (self.N > 0.) #Select only states for ortho-H2, which has the proton spins aligned so J can only be odd (1,3,5...)
				ortho_upperlimit = (self.J.u % 2 == 1) &  (self.V.u == i) & (self.s2n <= s2n_cut) & (self.N > 0.)  #Select ortho-H2 lines where there is no detection (e.g. S/N <= 1)
				if any(ortho): #If datapoints are found...
					log_N = log(self.N[ortho]/self.g[ortho]) #Log of the column density
					if nansum(self.s2n[ortho]) == 0.:
						plot(self.J.u[ortho], log_N, current_symbol,  color=current_color, label=' ', markersize=symbsize, fillstyle=orthofill)  #Plot data + error bars
					else:
						y_error_bars = [abs(log_N - log((self.N[ortho]-self.Nsigma[ortho])/self.g[ortho])), abs(log_N - log((self.N[ortho]-self.Nsigma[ortho])/self.g[ortho]))] #Calculate upper and lower ends on error bars
						errorbar(self.J.u[ortho], log_N, yerr=y_error_bars, fmt=current_symbol,  color=current_color, label=' ', capthick=3, elinewidth=2, markersize=symbsize, fillstyle=orthofill)  #Plot data + error bars
						if show_upper_limits:
							test = errorbar(self.J.u[ortho_upperlimit], log(self.Nsigma[ortho_upperlimit]*3.0/self.g[ortho_upperlimit]), yerr=1.0, fmt=current_symbol,  color=current_color, capthick=3, elinewidth=2, uplims=True, markersize=symbsize, fillstyle=orthofill) #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
					if show_labels: #If user wants to show labels for each of the lines
						for j in range(len(log_N)): #Loop through each point to label
							text(self.J.u[ortho][j], log_N[j], '        '+self.label[ortho][j], fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black')  #Label line with text
					#print('For ortho v=', i)
				else: #Else if no datapoints are found...
					errorbar([nan], [nan], yerr=1.0, fmt=current_symbol,  color=current_color, label=' ', capthick=3,  elinewidth=2, markersize=symbsize, fillstyle=orthofill)  #Do empty plot to fill legend
				
			for i in use_upper_v_states:
				if nocolor: #If user specifies no color,
					current_color = 'Black'
					current_symbol = symbol_list[i]
				else: #Or else by default use colors from the color list defined at the top of the code
					current_color = color_gradient[i]
					current_symbol = '^'
				para = (self.J.u % 2 == 0) & (self.V.u == i) & (self.s2n > s2n_cut) & (self.N > 0.) #Select only states for para-H2, which has the proton spins anti-aligned so J can only be even (0,2,4,...)
				para_upperlimit =  (self.J.u % 2 == 0) & (self.V.u == i) & (self.s2n <= s2n_cut) & (self.N > 0.) #Select para-H2 lines where there is no detection (e.g. S/N <= 1)
				if any(para): #If datapoints are found...
					log_N = log(self.N[para]/self.g[para]) #Log of the column density
					if nansum(self.s2n[para]) == 0.:
						plot(self.J.u[para], log_N, current_symbol,  color=current_color, label='v='+str(i), markersize=symbsize, fillstyle=parafill)  #Plot data + error bars
					else:
						y_error_bars = [abs(log_N - log((self.N[para]-self.Nsigma[para])/self.g[para])), abs(log_N - log((self.N[para]-self.Nsigma[para])/self.g[para]))] #Calculate upper and lower ends on error bars
						errorbar(self.J.u[para], log_N, yerr=y_error_bars, fmt=current_symbol,  color=current_color, label='v='+str(i), capthick=3, elinewidth=2, markersize=symbsize, fillstyle=parafill)  #Plot data + error bars
						if show_upper_limits:
							test = errorbar(self.J.u[para_upperlimit], log(self.Nsigma[para_upperlimit]*3.0/self.g[para_upperlimit]), yerr=1.0, fmt=current_symbol,  color=current_color, capthick=3, elinewidth=2, uplims=True, markersize=symbsize, fillstyle=parafill) #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
					if show_labels: #If user wants to show labels for each of the lines
						for j in range(len(log_N)): #Loop through each point to label
							text(self.J.u[para][j], log_N[j], '        '+self.label[para][j], fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black')  #Label line with text
					#print('For para v=', i)
				else: #Else if no datapoints are found...
					errorbar([nan], [nan], yerr=1.0, fmt=current_symbol,  color=current_color, label='v='+str(i), capthick=3, elinewidth=2, markersize=symbsize, fillstyle=parafill)  #Do empty plot to fill legend
			tick_params(labelsize=14) #Set tick mark label size
			if normalize: #If normalizing to the 1-0 S(1) line
				ylabel("Column Density   ln(N$_u$/g$_u$)-ln(N$_{r}$/g$_{r}$)", fontsize=labelsize)
			else:  #If using absolute flux calibrated data
				ylabel("Column Density   pn(N$_u$/g$_u$) [cm$^{-2}$]", fontsize=labelsize)
			xlabel("Upper Rotation State J$_u$", fontsize=labelsize, labelpad=4)
			if x_range[1] == 0.0: #If user does not specifiy a range for the x-axis
				xlim([0,1.4*max(self.J.u[self.s2n >= s2n_cut])]) #Autoscale
			else: #Else if user specifies range
				xlim(x_range) #Use user specified range for x-axis
			if y_range[1] != 0.0: #If user specifies a y axis limit, use it
				ylim(y_range) #Use user specified y axis range
			if show_legend: #If user does not turn off showing the legend
				legend(loc=1, ncol=2, fontsize=legend_fontsize, numpoints=1, columnspacing=-0.5, title = 'ortho  para  ladder')
			#show()
			draw()
			if savepdf:
				pdf.savefig() #Add in the pdf
			#stop()
	def vibration_plot(self, show_upper_limits = True, nocolor = False, J=[-1], s2n_cut=-1.0, normalize=True, savepdf=True, empty_fill =False, full_fill=False,
   				show_labels=False, x_range=[0.,0.], y_range=[0.,0.], show_legend=True, fname='', clear=True, legend_fontsize=14):
		if fname == '':
			fname = self.path + '_vibration_diagram.pdf'
		with PdfPages(fname) as pdf: #Make a pdf
			nonzero = self.N != 0.0
			if clear: #User can specify if they want to clear the plot
				clf()
			symbsize = 7 #Size of symbols on excitation diagram
			labelsize = 18 #Size of text for labels
			if empty_fill:
				fill = 'none'
			else:
				fill = 'full'
			if J == [-1]: #If user does not specify a specific set of V states to plot...
				use_upper_j_states = unique(self.J.u) #plot every one found
			else: #or else...
				use_upper_j_states = J #Plot upper V s tates specified by the user
			for i in use_upper_j_states:
				if nocolor: #If user specifies no color,
					current_color = 'Black'
					current_symbol = symbol_list[i]
				else: #Or else by default use colors from the color list defined at the top of the code
					current_color = color_gradient[i]
					current_symbol = 'o'
				found = (self.J.u == i) & (self.s2n > s2n_cut) & (self.N > 0.) #Select only states for para-H2, which has the proton spins anti-aligned so J can only be even (0,2,4,...)
				upperlimit =  (self.J.u == i) & (self.s2n <= s2n_cut) & (self.N > 0.) #Select para-H2 lines where there is no detection (e.g. S/N <= 1)
				if any(found): #If datapoints are found...
					log_N = log(self.N[found]/self.g[found]) #Log of the column density
					if nansum(self.s2n[found]) == 0.:
						plot(self.V.u[found], log_N, current_symbol,  color=current_color, label='v='+str(i), markersize=symbsize, fillstyle=fill)  #Plot data + error bars
					else:
						y_error_bars = [abs(log_N - log((self.N[found]-self.Nsigma[found])/self.g[found]) ), abs(log_N - log((self.N[found]-self.Nsigma[found])/self.g[found]) )] #Calculate upper and lower ends on error bars
						errorbar(self.V.u[found], log_N, yerr=y_error_bars, fmt=current_symbol,  color=current_color, label='J='+str(i), capthick=3, elinewidth=2, markersize=symbsize, fillstyle=fill)  #Plot data + error bars
						if show_upper_limits:
							test = errorbar(self.V.u[upperlimit], log(self.Nsigma[upperlimit]*3.0/self.g[upperlimit]), yerr=1.0, fmt=current_symbol,  color=current_color, capthick=3, elinewidth=2, uplims=True, markersize=symbsize, fillstyle=fill) #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
					if show_labels: #If user wants to show labels for each of the lines
						for j in range(len(log_N)): #Loop through each point to label
							text(self.V.u[found][j], log_N[j], '        '+self.label[found][j], fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black')  #Label line with text
				else: #Else if no datapoints are found...
					errorbar([nan], [nan], yerr=1.0, fmt=current_symbol,  color=current_color, label='J='+str(i), capthick=3, markersize=symbsize, fillstyle=fill)  #Do empty plot to fill legend
			tick_params(labelsize=14) #Set tick mark label size
			if normalize: #If normalizing to the 1-0 S(1) line
				ylabel("Column Density   ln(N$_u$/g$_u$)-ln(N$_{r}$/g$_{r}$)", fontsize=labelsize)
			else:  #If using absolute flux calibrated data
				ylabel("Column Density   ln(N$_u$/g$_u$) [cm$^{-2}$]", fontsize=labelsize)
			xlabel("Upper Vibration State v$_u$", fontsize=labelsize, labelpad=4)
			if x_range[1] == 0.0: #If user does not specifiy a range for the x-axis
				xlim([0,1.4*max(self.J.u[self.s2n >= s2n_cut])]) #Autoscale
			else: #Else if user specifies range
				xlim(x_range) #Use user specified range for x-axis
			if y_range[1] != 0.0: #If user specifies a y axis limit, use it
				ylim(y_range) #Use user specified y axis range
			if show_legend: #If user does not turn off showing the legend
				legend(loc=1, ncol=1, fontsize=legend_fontsize, numpoints=1, columnspacing=-0.5)
			#show()
			draw()
			if savepdf:
				pdf.savefig() #Add in the pdf
	def test_3D_plot(self, s2n_cut=-1.0, wireframe=False, surface=False, extra=[], x_range=[-1.0,15.0], y_range=[-1.0,15.0]):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		i = (self.s2n > s2n_cut) & (self.N > 0.) 
		#ax.scatter(self.V.u[i], self.J.u[i], log(self.N[i]))
		if surface: #By default plot as a surface plot
			ax.plot_trisurf(self.V.u[i], self.J.u[i], log(self.N[i]), cmap=cm.jet, alpha=0.3)
		#elif wireframe:
		#	ax.plot_wireframe(self.V.u[i], self.J.u[i], log(self.N[i]), rstride=2, cstride=2, cmap=cm.jet)
		else: #Else plot as a scatter plot
			ax.scatter(self.V.u[i], self.J.u[i], log(self.N[i]/self.g[i]))
		for more_surfaces in extra: #Plot more H2 surfaces (ie. models or thermalized populations)
			i = (more_surfaces.N > 0.) 
			ax.plot_trisurf(more_surfaces.V.u[i], more_surfaces.J.u[i], log(more_surfaces.N[i]), cmap=cm.jet, alpha=0.3)
		xlim(x_range)
		ylim(y_range)
		ax.set_xlabel('v$_u$')
		ax.set_ylabel('J$_u$')
		ax.set_zlabel('Column Density   ln(N$_i$/g$_i$)-ln(N$_{r}$/g$_{r}$)')
		draw()
		stop()
	def correct_extinction(self, s2n_cut=3.0, alpha_range=arange(0.5,3.0,0.1), A_K_range=arange(0.0,5.0,0.01)): 
		#First find all the line pairs and store their indicies
		pair_a = [] #Store index numbers of one of a set of line pairs from the same upper state
		pair_b = [] #Store index numbers of the other of a set of line pairs from the same upper state
		i = (self.F != 0.0) & (self.s2n > s2n_cut) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
		J_upper_found = unique(self.J.u[i]) #Find J for all (detected) transition upper states
		V_upper_found = unique(self.V.u[i]) #Find V for all (detected) transition upper states
		for V in V_upper_found: #Check each upper V for pairs		
			for J in J_upper_found: #Check each upper J for pairs
				#i = (self.F != 0.0) & (self.s2n > s2n_cut) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
				match_upper_states = (self.J.u[i] == J) & (self.V.u[i] == V) #Find all transitions from the same upper J and V state
				waves = self.wave[i][match_upper_states] #Store wavelengths of all found transitions
				s = argsort(waves) #sort by wavelength
				waves = waves[s]
				labels = self.label[i][match_upper_states][s]
				if len(waves) == 2 and abs(waves[0]-waves[1]) > wave_thresh: #If a single pair of lines from the same upper state are found, calculate observed vs. intrinsic ratio
					pair_a.append(where(self.wave == waves[0])[0][0])
					pair_b.append(where(self.wave == waves[1])[0][0])
				elif len(waves) == 3: #If three liens are found from the same upper state, calculate differential extinction from differences between all three lines
					#Pair 1
					if abs(waves[0] - waves[1]) > wave_thresh:
						pair_a.append(where(self.wave == waves[0])[0][0])
						pair_b.append(where(self.wave == waves[1])[0][0])
					#Pair 2
					if abs(waves[0] - waves[2]) > wave_thresh: #check if pair of lines are far enoug7h apart
						pair_a.append(where(self.wave == waves[0])[0][0])
						pair_b.append(where(self.wave == waves[2])[0][0])
					if abs(waves[1] - waves[2]) > wave_thresh: #check if pair of lines are far enough apart
						pair_a.append(where(self.wave == waves[1])[0][0])
						pair_b.append(where(self.wave == waves[2])[0][0])
		pair_a = array(pair_a) #Turn lists of indicies into arrays of indicies
		pair_b = array(pair_b)
		chisqs = [] #Store chisq for each possible extinction and extinction law
		alphas = [] #Store alphas for each possible exctinction and extinction law
		A_Ks = [] #Store extinctions for each possible exctinction and exctinction law
		for a in alpha_range: #Loop through different exctinction law powers
			for A_K in A_K_range: #Loop through different possible K band exctinctions
				h = copy.deepcopy(self) #Make a copy of the input h2 line object
				A_lambda = A_K * h.wave**(-a) / lambda0**(-a) #Calculate an extinction correction
				h.F *= 10**(0.4*A_lambda) #Apply extinction correction
				h.calculate_column_density() #Calculate column densities from each transition, given the guess at extinction correction
				chisq = nansum((h.N[pair_a] - h.N[pair_b])**2 /  h.N[pair_b]) #Calculate chisq from all line pairs that arise from same upper states
				chisqs.append(chisq) #Store chisq and corrisponding variables for extinction correction
				alphas.append(a)
				A_Ks.append(A_K)
		chisqs = array(chisqs) #Convert lists to arrays
		alphas = array(alphas)
		A_Ks = array(A_Ks)
		best_fit = chisqs == nanmin(chisqs) #Find the minimum chisq and best fit alpha and A_K
		best_fit_A_K = A_Ks[best_fit]
		best_fit_alpha = alphas[best_fit]
		print('Best fit alpha =', best_fit_alpha) #Print results so user can see
		print('Best fit A_K = ', best_fit_A_K)
		A_lambda = best_fit_A_K * self.wave**(-best_fit_alpha) / lambda0**(-best_fit_alpha) #Calculate an extinction correction
		self.F *= 10**(0.4*A_lambda) #Apply extinction correction
		self.calculate_column_density() #Calculate column densities from each transition, given the new extinction correction
		self.A_K = best_fit_A_K #Store extinction paramters in case user wants to inspect or tabulate them later
		self.alpha = best_fit_alpha
	# def find_cascade(v_u, j_u, v_l, j_l): #Find all possible paths between two levels
	# 	found_transitions = self.tout(v_up, j_up) #Find all transitions out of the upper level

	# 	for i in range(len(v_l_trans)):
	# 		if v_l_trans == v_l and j_l_trans == j_l
	# 	for found_transition in found_transitions: #Loop through each transition found
	# 		if self.v.l[found_transition] == v_l and 
	# 		XXXXX.append(find_cascade
	def gbar_approx(self): #Estimate collision rate coeffs based on the g-bar approximatino done in Shaw et al. (2005) Section 2.2.1 and Table 2
		y0 = array([-9.9265, -8.281, -10.0357, -8.6213, -9.2719])#Coeffs from Table 2 of Shaw et al (2005 for H0, He, H2(ortho), H2(para), & H+)
		a = array([-0.1048, -0.1303, -0.0243, -0.1004, -0.0001])
		b = array([0.456, 0.4931, 0.67, 0.5291, 1.0391])
		E_trans = self.E.diff() #Grab energy of transitions (in wavenumber cm^-1)
		E_trans[E_trans < 100.0] = 100.0 #Set max(sigma, 100) sen in Eq. 1 of Shaw et al. (2005)
		k_total = zeros(len(E_trans)) #Store total collisional coeffs (cm^3 s^-1)
		for i in range(5): #Loop through g-bar approx for  H0, He, H2(ortho), H2(para), & H+ and total up the collisional coeffs k for each transition
			k_total += exp( y0[i] + a[i]*(E_trans**b[i]) ) #Eq 1 in Shaw et al. (2005)
		#k_total[k_total < 0.] = 0. #Ignore negative coefficients
		self.k = k_total #Store the estimated collisional reate coeffs for each transition



class transition_node:
	def __init__(self, h2_obj, v, J, itercount=0, wave_range=[0.,0.]):
		#if itercount < 50:
			print('itercont = ', itercount, '   v = ', v , ' J = ', J)
			touts = h2_obj.tout(v, J)
			n  = size(touts)
			if n > 0:
				children = []
				v_out, J_out, wave = h2_obj.V.l[touts], h2_obj.J.l[touts], h2_obj.wave[touts]
				for i in range(n):
					if (wave_range[0] == 0. and wave_range[1] == 0.) or (wave[i] >= wave_range[0] and wave[i] <= wave_range[1]):
						print('found transition ', h2_obj.label[touts][i], ' wavelength=', h2_obj.wave[touts][i])
						children.append(transition_node(h2_obj, v_out[i], J_out[i], itercount + 1, wave_range=wave_range))
				self.v = v
				self.J = J
				self.i = touts
				self.children = children
				self.last = False
			else:
				self.last = True
				print('Looks like that is the last of one set of tranistions.')


class density_surface(): #Fit surface in v and J space, save object to store surface
	def __init__(self, h2_obj, s2n_cut=-1.0):
		i = (h2_obj.s2n > s2n_cut) & (h2_obj.N > 0.) #Filter out low S/N and unused points
		v = h2_obj.V.u[i] #Grabe the relavent variables to fit
		J = h2_obj.J.u[i]
		log_N = log(h2_obj.N[i])
		v_fit = linregress(v, log_N) #Fit v and J functions seperately with a simple linear regression
		J_fit = linregress(J, log_N)
		self.v_slope = v_fit.slope
		self.v_intercept = v_fit.intercept
		self.J_slope = J_fit.slope
		self.J_intercept = J_fit.intercept









#Store upper and lower J (rotational) states
class J:
	def __init__(self, u, l):
		self.u = u #Store upper J state
		self.l = l #Store lower J state
		self.label = self.makelabel() #Store portion of spectroscopic notation for J
	def diff(self): #Get difference between J upper and lower levels
		return self.u - self.l
	def makelabel(self): #Make spectroscopic notation label
		delta_J = self.diff() #Grab difference between J upper and lower levels
		n = len(delta_J) #Number of transitions
		J_labels = []
		for i in range(n):
			if delta_J[i] == -2:
				J_labels.append('O(' + str(self.l[i]) + ')')  #Create label O(J_l) for transitions where delta-J = -2
			elif delta_J[i] == 0:
				J_labels.append('Q(' + str(self.l[i]) + ')') #Create label Q(J_l) for transitions where delta-J = 0
			elif delta_J[i] == 2:
				J_labels.append('S(' + str(self.l[i]) + ')') #Create label S(J_l) for transitions where delta-J = +2
		return array(J_labels)
	def sort(self, sort_object): #Sort both upper and lower levels for a given sorted object fed to this function (e.g. argsort)
		self.u = self.u[sort_object] #Sort upper states
		self.l = self.l[sort_object] #Sort lower states
		self.label = self.label[sort_object] #Sort labels


#Store upper and lower V (vibrational) states
class V:
	def __init__(self, u, l):
		self.u = u #Store upper V state
		self.l = l #Store lower V state
		self.label = self.makelabel()  #Store portion of spectroscopic notation for V
	def diff(self): #Get difference between V upper and lower levels
		return self.u - self.l
	def makelabel(self):
		n = len(self.u) #Number of transitions
		V_labels = []
		for i in range(n):
			V_labels.append( str(self.u[i]) + '-' + str(self.l[i]) ) #Create label for V transitions of V_u-V_l
		return array(V_labels)
	def sort(self, sort_object): #Sort both upper and lower levels for a given sorted object fed to this function (e.g. argsort)
		self.u = self.u[sort_object] #Sort upper states
		self.l = self.l[sort_object] #Sort lower states
		self.label = self.label[sort_object] #Sort labels

#Store upper and lower E (energies) of the states
class E:
	def __init__(self, u, l):
		self.u = u #Store upper J state
		self.l = l #Store lower J state
		#self.wave = self.getwave() #Store wavelength for each line
	def diff(self): #Get difference between J upper and lower levels
		return self.u - self.l
	def getwave(self): #Get wavelength from difference in energy levels
		return self.diff()**(-1) * 1e4 #Get wavelength of line from energy [cm ^-1] and convert to um
	def sort(self, sort_object): #Sort both upper and lower levels for a given sorted object fed to this function (e.g. argsort)
		self.u = self.u[sort_object] #Sort upper states
		self.l = self.l[sort_object] #Sort lower states


#@jit
def run_cascade(iterations, time, N, trans_A, upper_states, lower_states, pure_rot_states, rovib_states_per_J, J, V, collisions=False, scale_factor=1e-10): #Speed up radiative cascade with numba
	transition_amount = trans_A*time
	#para = J%1==0
	#ortho = J%1==1
	#ground_J1 = (J==1) & (V==0)
	#ground_J0 = (J==0) & (V==0)
	#pure_rot_states = V == 0
	#rovib_states = V > 1
	#J_pure_rot_states = J[pure_rot_states]
	n_states = len(N)
	n_lines = len(trans_A)
	time_x_scale_factor = time * scale_factor
	for k in range(iterations): #loop through however many iterations user specifies
		#N[para] += distribution[para]*(N[ground_J0] + 0.5*(1.0-sum(N)))
		#N[ortho] += distribution[ortho]*(N[ground_J1] + 0.5*(1.0-sum(N)))
		#N += distribution * scale_factor*time
		if scale_factor > 0.:
			for current_J in range(32): #Loop through each J
				#current_rovib_states = (V > 1) & (J==current_J) #Grab index of current pure rotation state
				#current_pure_rot_state = (V == 0) & (J==current_J) #Grab indicies of current rovibration states with the same J as the current pure rotation state
				#current_pure_rot_state = pure_rot_states[current_J]
				#current_rovib_states = rovib_states_per_J[current_J]
				#num_current_rovib_states = len(N[current_rovib_states]) #Count number of rovibrational states we are going to redistribute the popluations from v=0 into
				delta_N = N[pure_rot_states[current_J]] * time_x_scale_factor#How many molecules out of the pure rotation state to redistribute to higher v
				N[rovib_states_per_J[current_J]] += delta_N / len(N[rovib_states_per_J[current_J]]) #num_current_rovib_states  #Redistribute fraction of pure rotation state molecules to higher v
				N[pure_rot_states[current_J]] -= delta_N #Remove molecules in pure rotation state that have now been redistributed
			#N[para] += scale_factor*distribution[para]#*N[ground_J0] #+ 0.5*(1.0-sum(N)))
			#N[ortho] += scale_factor*distribution[ortho]#*N[ground_J1] #+ 0.5*(1.0-sum(N)))	
			#N[ground_J0] = 0.
			#N[ground_J1] = 0.
		#N[pure_rot_states] = pure_rot_pops*sum(N[pure_rot_states]) / sum(pure_rot_pops) #Thermalize pure rotation states

		store_delta_N = zeros(n_states)  #Set up array to store all the changes in N 
		for i in range(n_lines):
			delta_N = N[upper_states[i]]*transition_amount[i] 
			store_delta_N[upper_states[i]] -= delta_N
			store_delta_N[lower_states[i]] += delta_N
		N += store_delta_N #Modfiy level populations after the effects of all the transitions have been summed up
		if collisions: #If user specifies to use collisions
			N -= 0.01*N*time*V  #Apply this very crude approximation of collisional de-excitation, which favors high V


	return N



#Object for storing column densities of individual levels, and performing calculations upon them
class states:
	def __init__(self, max_J=99):
		ion() #Set up plotting to be interactive
		show() #Open a plotting window
		V, J = loadtxt(energy_table, usecols=(0,1), unpack=True, dtype='int')#, skiprows=1) #Read in data for H2 ground state rovibrational energy levels
		E = loadtxt(energy_table, usecols=(2,), unpack=True, dtype='float')#, skiprows=1)
		if max_J < 99:  #If user specifies a maximum J, use only states where J <= max_J
			use_these_states = J <= max_J
			V = V[use_these_states]
			J = J[use_these_states]
			E = E[use_these_states]
		self.n_states = len(V) #Number of levels
		self.V = V #Array to store vibration level
		self.J = J #Array to store rotation level
		self.T = E / k #Excited energy above the ground rovibrational state in units of Kelvin
		self.N = zeros(self.n_states) #Array for storing column densities
		g_ortho_para = 1 + 2 * (J % 2 == 1) #Calculate the degenearcy for ortho or para hydrogen
		self.g = g_ortho_para * (2*J+1) #Store degeneracy
		self.tau = zeros(self.n_states) #array for storing radiative lifetime
		self.Q = zeros(self.n_states)
		self.A_tot_in = zeros(self.n_states) #A tots for radiative transitions
		self.A_tot_out = zeros(self.n_states)
		self.k_tot_out = zeros(self.n_states) #Estimated k tot (collision rate coeff.) for collisional transitions
		self.transitions = make_line_list() #Create transitions list 
		self.transitions.upper_states = zeros(self.transitions.n_lines, dtype=int) #set up index to upper states
		self.transitions.lower_states = zeros(self.transitions.n_lines, dtype=int) #Set up index to lower states
		self.transitions.gbar_approx() #Estiamte collisional rate coeffs based on section 2.1.1 of Shaw et al. (2005)
		for i in range(self.transitions.n_lines):
			if self.transitions.J.u[i] <= max_J and self.transitions.J.l[i] <= max_J:
				self.transitions.upper_states[i] = where((J == self.transitions.J.u[i]) & (V == self.transitions.V.u[i]))[0][0] #Find index of upper states 
				self.transitions.lower_states[i] = where((J == self.transitions.J.l[i]) & (V == self.transitions.V.l[i]))[0][0] #Find index of lower states 
		for i in range(self.n_states): #Calculate relative lifetime of each level (inverse sum of transition probabilities), see Black & Dalgarno (1976) Eq. 4
			transitions_out_of_this_state = (self.transitions.J.l == J[i]) & (self.transitions.V.l == V[i])  #Find transitions out of this state
			transitions_into_this_state =  (self.transitions.J.u == J[i]) & (self.transitions.V.u == V[i]) 
			self.tau[i] = sum(self.transitions.A[transitions_out_of_this_state])**-1 #Black & Dalgarno (1976) Eq. 4
			self.Q[i] = sum(self.transitions.A[transitions_into_this_state])**-1 
			self.A_tot_out[i] = sum(self.transitions.A[transitions_out_of_this_state]) #Black & Dalgarno (1976) Eq. 4
			self.A_tot_in[i] = sum(self.transitions.A[transitions_into_this_state])
			self.k_tot_out[i] = sum(self.transitions.k[transitions_out_of_this_state])
		self.ncr = self.A_tot_out / self.k_tot_out #Estimate critical densities
		self.test_n = self.Q * self.tau
		self.start_cascade = False #Flag if cascade has started or not
		#self.convergence = [] #Set up python list that will hold convergence of cascade
		#UV pumping from Black & Dalgarno (1976) 
		self.BD76_cloud_boundary_pumping = array([1.78e-11, 1.32e-11, 1.32e-11, 8.88e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1e-11, 7.77e-12, 7.81e-12, 5.1e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.06e-11, 7.02e-12, 7.23e-12, 4.64e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		  0.0, 0.0, 1.07e-11, 6.85e-12, 7.18e-12, 4.55e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.05e-11, 6.65e-12,
		   6.96e-12, 4.4e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.01e-11, 6.43e-12, 6.7e-12, 4.24e-12, 0.0, 0.0, 0.0, 0.0, 
		   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.71e-12, 5.95e-12, 6.01e-12, 3.91e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 7.47e-12, 5.47e-12, 5.38e-12, 3.58e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.66e-12, 4.86e-12, 4.53e-12, 3.17e-12,
		     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.62e-12, 4.6e-12, 4.1e-12, 2.99e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.36e-12, 
		     3.89e-12, 3.07e-12, 2.51e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.39e-12, 3.66e-12, 2.77e-12, 2.41e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.45e-12, 3.8e-12, 
		     2.88e-12, 2.49e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.36e-12, 3.55e-12, 2.69e-12, 2.29e-12, 0.0, 0.0, 0.0, 0.0, 8.71e-13, 2.29e-12, 1.58e-12, 1.27e-12])
		self.BD76_formation_pumping = array([1.12e-14, 1.14e-13, 5.41e-12, 2.53e-13, 9.08e-14, 3.66e-13, 1.19e-13, 4.43e-13, 1.36e-13, 4.83e-13, 1.42e-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.91e-15, 8.09e-14, 3.84e-14, 1.8e-13, 6.46e-14, 2.59e-13, 8.43e-14, 3.14e-13, 9.6e-14, 3.43e-13, 1.01e-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.73e-15, 5.86e-14, 2.78e-14, 1.3e-13, 4.68e-14, 1.88e-13, 6.09e-14, 2.27e-13, 6.95e-14, 2.47e-13, 7.27e-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.21e-15, 4.31e-14, 2.05e-14, 9.6e-14, 3.44e-14, 1.38e-13, 4.49e-14, 1.67e-13, 5.12e-14, 1.83e-13, 5.37e-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.17e-15, 3.24e-14, 1.54e-14, 7.19e-14, 2.59e-14, 1.04e-13, 3.36e-14, 1.25e-13, 3.85e-14, 1.37e-13, 4.03e-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.42e-15, 2.47e-14, 1.18e-14, 5.5e-14, 1.98e-14, 7.94e-14, 2.58e-14, 9.6e-14, 2.94e-14, 1.05e-13, 3.09e-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		     0.0, 0.0, 0.0, 0.0, 1.89e-15, 1.93e-14, 9.17e-15, 4.29e-14, 1.54e-14, 6.18e-14, 2.01e-14, 7.48e-14, 2.29e-14, 8.16e-14, 2.4e-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		     1.5e-15, 1.53e-14, 7.27e-15, 3.41e-14, 1.23e-14, 4.91e-14, 1.59e-14, 5.95e-14, 1.83e-14, 6.49e-14, 1.91e-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.22e-15, 1.25e-14, 5.9e-15, 2.76e-14,
		      9.95e-15, 3.99e-14, 1.3e-14, 4.83e-14, 1.48e-14, 5.26e-15, 1.55e-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-15, 1.03e-14, 4.88e-15, 2.28e-14, 8.22e-15, 3.3e-14, 1.07e-14, 3.99e-14, 1.22e-14,
		       4.35e-14, 1.28e-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.5e-16, 8.74e-15, 4.13e-15, 1.93e-14, 6.95e-15, 2.79e-14, 9.08e-15, 3.37e-14, 1.03e-14, 3.68e-14, 1.08e-14, 0.0, 0.0, 0.0, 0.0, 7.37e-16, 
		       7.53e-15, 3.57e-15, 1.68e-14, 6.02e-15, 2.41e-14, 7.85e-15, 2.92e-14, 9e-15, 3.18e-14, 9.43e-15, 0.0, 0.0, 6.55e-16, 6.7e-15, 3.18e-15, 
			1.49e-14, 5.35e-15, 2.15e-14, 6.97e-15, 2.6e-14, 7.97e-15, 2.84e-14, 8.35e-15, 6.01e-16, 6.15e-15, 2.92e-15, 1.37e-14, 4.91e-15, 1.94e-14, 6.4e-15, 2.39e-14, 7.31e-15, 2.6e-14, 7.66e-15, 5.7e-16])
		#self.BD76_cloud_center_pumping = 
		pure_rot_states = [] #Save indicies of pure rotation states
		rovib_states_per_J = [] #Save indicies of each set of rovib states of constant J
		for current_J in range(32): #Loop through each rotation level
			pure_rot_states.append((J == current_J) & (V == 0)) #Store indicies for a given J for pure rotation state
			rovib_states_per_J.append((J == current_J) & (V > 0)) #Store indicies for a given J for all rovib. states where v>0
		self.pure_rot_states = pure_rot_states
		self.rovib_states_per_J = rovib_states_per_J

	def thermalize(self, temperature, N_tot=1.0): #Set populations to be thermal at the user supplied temperature
		exponential = self.g * exp(-self.T/temperature) #Calculate boltzmann distribution for user given temperature, used to populate energy levels
		boltzmann_distribution = exponential / nansum(exponential) #Create a normalized boltzmann distribution
		self.N = boltzmann_distribution * N_tot #Set column densities to the boltzmann distribution
	def generate_synthetic_spectrum(self, wave_range=[1.45,2.45], pixel_size=1e-5, line_fwhm=7.5, centroid=0.): #Generate a synthetic 1D spectrum based on stored flux values in this object, can be used to synthesize spectra from Cloudy models, or thermal gas generated by the "thermalize" command
		self.set_transition_column_densities() #Set column densities from the states class object here
		self.transitions.calculate_flux()
		w, f = self.transitions.generate_synthetic_spectrum(wave_range=wave_range, pixel_size=pixel_size, line_fwhm=line_fwhm, centroid=centroid)
		return w, f #Send the wavelength and flux arrays back to you
	def total_N(self): #Grab total column density N and return it
		return nansum(self.N) #Return total column density of H2
	def cascade(self, time=1.0, temp=250.0, quick=0, showplot=True, iterations=1, scale_factor=1e-10, collisions=False): #Do a step in the radiative cascade
		V = self.V #Assign variables to speed up loop
		J = self.J
		N = self.N
		g = self.g
		#trans_V_u = self.transitions.V.u
		#trans_V_l = self.transitions.V.l
		#trans_J_u = self.transitions.J.u
		#trans_J_l = self.transitions.J.l
		trans_A = self.transitions.A
		upper_states = self.transitions.upper_states
		lower_states = self.transitions.lower_states
		pure_rot_states = self.pure_rot_states
		rovib_states_per_J = self.rovib_states_per_J
		#if quick != 0: #If user wants to speed up the cascade
		#	maxthresh = -partsort(-trans_A,quick)[quick-1] #Find threshold for maximum
		#else:
		#	maxthresh = 0. #E
		exponential = g * exp(-self.T/temp) #Calculate boltzmann distribution for user given temperature, used to populate energy levels
		boltmann_distribution = exponential / nansum(exponential)
		# ground_J1 = (J==1) & (V==0)
		# ground_J0 = (J==0) & (V==0)
		if not self.start_cascade: #If cascade has not started yet 
			N = zeros(self.n_states) #Start off with everything  = 0.0
			N = boltmann_distribution #preset the population to be the boltzmann distribution 
			# pure_rotation_states = V == 0
			# const = 1e-3 #Fraction to populate other states at v > 0 to match the J levels of v=0
			# N[pure_rotation_states] = boltmann_distribution[pure_rotation_states] #preset the population to be the boltzmann distribution for only the pure rotation states
			# for current_J in J[pure_rotation_states]: #loop through each possible rotation state
			# 	other_vibrational_states_with_same_J = (J==current_J) & (~pure_rotation_states) #Find states at higher V with same J
			# 	if any(other_vibrational_states_with_same_J): #If any J states in v>0 matches the current J
			# 		N[other_vibrational_states_with_same_J] = const * N[pure_rotation_states & (J==current_J)] #Set level populations
			# #N = (1.-(J/10.)) + (1.-(V/14.0)) #Set populations based on V and J
			# #N[(V==14) & (J==1)] = 1.0
			# #N = self.BD76_formation_pumping
			# #N = self.BD76_cloud_boundary_pumping
			# #N = exp(-(0.5*(J+1)+0.3*(V+1)))
	 	# 	#N[V>-1] = exp(-(0.25*(J[V>-1]-1)+0.2*(V[V>-1]-1)))
	 	# 	#N[ground_J0] = 0.
	 	# 	#N[ground_J1] = 0.
			# #N = exp(-J.astype(float)-V.astype(float))
			# #N = ones(self.n_states)
			# N[V==0] == 0.
			N = N/sum(N) #Normalize
			self.distribution = copy.deepcopy(N)
			self.start_cascade = True #Then flip the flag so that the populations stay as they are
		#old_N = copy.deepcopy(self.N)

		N = run_cascade(iterations, time, N, trans_A, upper_states, lower_states, pure_rot_states, rovib_states_per_J, J, V, collisions=collisions, scale_factor=scale_factor) #Test cascade with numba
		# transition_amount = trans_A*time
		# para = J%1==0
		# ortho = J%1==1
		# for k in range(iterations): #loop through however many iterations user specifies
		# 	#delta_N = N*trans_A[upper_states]*time
		# 	#store_delta_N -= delta_N
		# 	#store_delta_N += delta_N
		# 	#for i in range(self.n_states):
		# 	#	u = upper_states==i
		# 	#	l = lower_states==i
		# 	#	store_delta_N[i] -= N[i]*sum(trans_A[u])*time
		# 	#	store_delta_N[i] += sum(N[l]*trans_A[l])*time
		# 	store_delta_N = zeros(self.n_states)  #Set up array to store all the changes in N 
		# 	delta_N = N[upper_states]*transition_amount #Move this much H2 around with this transition
		# 	for i in range(self.n_states):
		# 		store_delta_N[i] = nansum(delta_N[lower_states == i]) - nansum(delta_N[upper_states == i])
		# 	#store_delta_N[upper_states] -= delta_N
		# 	#store_delta_N[lower_states] += delta_N
		# 	#for i in range(self.transitions.n_lines): #Loop through each transition
		# 	# 	#if trans_A[i] > maxthresh: #Select only certain transitions below a certain A to be important, to optimize code
		# 	# 		#Ju = self.transitions.J.u[i] #Grab upper and lower J levels for this transition
		# 	# 		#Jl = self.transitions.J.l[i]
		# 	# 		#Vu = self.transitions.V.u[i] #Grab upper and lower V levels for this transition
		# 	# 		#Vl = self.transitions.V.l[i]
		# 	# 		#upper_state = logical_and(V == trans_V_u[i], J == trans_J_u[i])#Finder upper state of transition
		# 	# 		#lower_state = logical_and(V == trans_V_l[i], J == trans_J_l[i])#Find loer state of transition
		# 	# 		#upper_state = (V == trans_V_u[i]) & (J == trans_J_u[i])#Finder upper state of transition
		# 	# 		#lower_state = (V == trans_V_l[i]) & (J == trans_J_l[i])#Find loer state of transition
			 		
		# 	 		#store_delta_N[upper_states[i], lower_states[i]] += [-delta_N, delta_N] #Try some vectorization
		# 			#print('delta_N=', delta_N)
		# 	# 		#delta_N = self.N[upper_state]*(1.0 - exp(-self.transitions.A[i]*time) )#Move this much H2 around with this transition
		# 	# 		store_delta_N[upper_states[i]] -= delta_N[i] #Store change in N taken out of upper state by this transition
		# 	# 		store_delta_N[lower_states[i]] += delta_N[i] #Store change in N put into lower state by this transition
		# 	N += store_delta_N #Modfiy level populations after the effects of all the transitions have been summed up
		# 	#self.N[1:] = self.N[1:] + self.N[0] / (float(self.n_states)-1.0) #Crudely redistribute everything in the ground state back to all other states
	 # 		#N[1:] = N[1:] + boltmann_distribution[1:]*N[0]
	 # 		#N[j] = N[j] + boltmann_distribution*N[0]
	 # 		#stop()
	 # 		N[para] += self.distribution[para]*(N[ground_J0] + 0.5*(1.0-nansum(N)))
	 # 		N[ortho] += self.distribution[ortho]*(N[ground_J1] + 0.5*(1.0-nansum(N)))
	 # 		#N[J%1==0] += boltmann_distribution[J%1==0]*N[ground_J0]
	 # 		#N[J%1==1] += boltmann_distribution[J%1==1]*N[ground_J1]
	 # 		N[ground_J0] = 0.
	 # 		N[ground_J1] = 0.
	 # 		#stop()
	 # 		#N[V>0] += self.distribution*sum(N[V==0])
	 # 		#N[V==0] = 0.

			
		#N[285] = N[285] + N[0] #Test just dumping everything into the final level and let everything cascade out of it.
		#N[0]  = 0.0 #Empty out ground state after redistributing all the molecules in the ground
		#convergence_measurement = (nansum((N-old_N)))**2
		#print('convergence = ', convergence_measurement)
		#self.convergence.append(convergence_measurement)#Calculate convergence from one step to the 
		self.N = N
		if showplot:
			self.set_transition_column_densities()
			self.transitions.v_plot(s2n_cut=-1.0, savepdf=False)
		#stop()
	def set_transition_column_densities(self): #Put column densities into 
		for i in range(self.n_states): #Loop through each level
			upper_states = (self.transitions.J.u == self.J[i]) & (self.transitions.V.u == self.V[i]) #Find all transitions with upper states in a level
			self.transitions.N[upper_states] = self.N[i] #Set new column densities to the transitions object
