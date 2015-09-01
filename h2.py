from pdb import set_trace as stop #Use stop() for debugging
#from scipy import *
from pylab import *
from matplotlib.backends.backend_pdf import PdfPages  #For outputting a pdf with multiple pages (or one page)
#from astropy.modeling import models, fitting #Import astropy models and fitting for fitting linear functions for tempreatures (e.g. rotation temp)
from scipy.optimize import curve_fit
import copy
from bottleneck import *


#Global variables, modify
single_temp = 1500.0 #K
single_temp_y_intercept = 22.0
alpha = arange(0.0, 10.0, 0.01) #Save range of power laws to fit extinction curve [A_lambda = A_lambda0 * (lambda/lambda0)^alpha
lambda0 = 2.12 #Wavelength in microns for normalizing the power law exctinoction curve, here it is set to the K-badn at 2.12 um
wave_thresh = 0.0 #Set wavelength threshold (here 0.1 um) for trying to measure extinction, we need the line pairs to be far enough apart we can get a handle on the extinction

#Global variables, do not modify
cloudy_dir = '/Users/kfkaplan/Dropbox/cloudy/'
cloudy_h2_data_dir = 'data/' #Directory where H2 data is stored for cloudy
energy_table = cloudy_h2_data_dir + 'energy_X.dat' #Name of table where Cloudy stores data on H2 electronic ground state rovibrational energies
transition_table = cloudy_h2_data_dir + 'transprob_X.dat' #Name of table where Cloudy stores data on H2 transition probabilities (Einstein A coeffs.)
k = 0.69503476 #Botlzmann constant k in units of cm^-1 K^-1 (from http://physics.nist.gov/cuu/Constants/index.html)
h = 6.6260755e-27 #Plank constant in erg s, used for converting energy in wave numbers to cgs
c = 2.99792458e10 #Speed of light in cm s^-1, used for converting energy in wave numbers to cgs

#Make array of color names
color_list = ['black','gray','darkorange','blue','red','green','orange','magenta','darkgoldenrod','purple','deeppink','darkolivegreen', 'cyan','yellow','beige']
symbol_list = ['o','v','8','x','s','*','h','D','^','8','1','o','o','o','o','o','o','o'] #Symbol list for rotation ladders on black and white Boltzmann plot
#for c in matplotlib.colors.cnames:
    #color_list.append(c)

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
	for i in xrange(len(labels)): #Loop through each line in the table
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
	for i in xrange(len(labels)): #Loop through each line in the table
		match = ice_A.label == labels[i] #Match H2 transition objects to the line and set the column density N to what is in the table
		same_upper_level = (ice_A.V.u == ice_A.V.u[match]) & (ice_A.J.u == ice_A.J.u[match]) #Find all lines from the same upper state
		ice_A.N[same_upper_level] = ice_A.N[match]
		ice_B.N[same_upper_level] = ice_B.N[match]
		Si_A.N[same_upper_level] = Si_A.N[match]
		Si_B.N[same_upper_level] = Si_B.N[match]
		C_A.N[same_upper_level] = C_A.N[match]
		C_B.N[same_upper_level] = C_B.N[match]
	return(ice_A, ice_B, Si_A, Si_B, C_A, C_B) #Return objects

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
	level_V, level_J = loadtxt(energy_table, usecols=(0,1), unpack=True, dtype='int', skiprows=1) #Read in data for H2 ground state rovibrational energy levels
	level_E = loadtxt(energy_table, usecols=(2,), unpack=True, dtype='float', skiprows=1)
	trans_Vu, trans_Ju, trans_Vl, trans_Jl =  loadtxt(transition_table, usecols=(1,2,4,5), unpack=True, dtype='int', skiprows=1) #Read in data for the transitions (ie. spectral lines which get created by the emission of a photon)
	trans_A =  loadtxt(transition_table, usecols=(6,), unpack=True, dtype='float', skiprows=1) #Read in data for the transitions (ie. spectral lines which get created by the emission of a photon)
	n_transitions = len(trans_Vu) #Number of transitions
	#Organize molecular data into objects storing J, V, Energy, and A values
	J_obj = J(trans_Ju, trans_Jl) #Create object storing upper and lower J levels for each transition
	V_obj = V(trans_Vu, trans_Vl) #Create object storing upper and lower V levels for each transition
	A = trans_A
	E_u = zeros(n_transitions)
	E_l = zeros(n_transitions)
	for i in xrange(n_transitions):
		E_u[i] = level_E[ (level_V == trans_Vu[i]) & (level_J == trans_Ju[i]) ]
		E_l[i] = level_E[ (level_V == trans_Vl[i]) & (level_J == trans_Jl[i]) ]
	E_obj = E(E_u, E_l) #Create object for storing energies of upper and lower rovibrational levels for each transition
	#Create and return the transitions object which stores all the information for each transition
	transitions = h2_transitions(J_obj, V_obj, E_obj, A) #Create main transitions object
	return transitions #Return transitions object

#Definition that takes all the H2 lines with determined column densities and calculates as many differential extinctions as it can between
#pairs of lines that come from the same upper state, and fits an extinction curve (power law here) to them
def fit_extinction_curve(transitions):
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
				n_trips_found = n_trips_found + 1
	figure(1)
	clf() #Clear plot field
	for pair in pairs: #Loop through each pair
		pair.fit_curve()
		plot(alpha, pair.A_K)
	xlabel('Alpha')
	ylabel('$A_K$')
	ylim([0,20])
	figure(2)
	clf()
	for pair in pairs: #Loop through each pair
		plot(pair.waves, [0,pair.A])
	#suptitle('V = ' + str(V))
	a = input('What value of alpha do you want to use? ')
	A_K = input('What value of A_K do you want to use?') 
	A_lambda = A_K * transitions.wave**(-a) / lambda0**(-a)
	transitions.F = transitions.F * 10**(0.4*A_lambda)
	transitions.sigma = transitions.sigma * 10**(0.4*A_lambda)
	#stop()
	print 'Number of pairs from same upper state = ', n_doubles_found
	print 'Number of tripples from same upper state = ', n_trips_found
	
def import_cloudy(): #Import cloudy model from cloudy directory
	paths = open(cloudy_dir + 'process_model/input.dat') #Read in current model
	model = paths.readline().split(' ')[0]
	distance = float(paths.readline().split(' ')[0])
	inner_radius = float(paths.readline().split(' ')[0])
	slit_area = float(paths.readline().split(' ')[0])
	data_dir =  paths.readline().split(' ')[0]
	plot_dir = paths.readline().split(' ')[0]
	table_dir =  paths.readline().split(' ')[0]
	paths.close()
	filename = data_dir+model+".h2.coldens" #Name of file to open
	#stop()
	v, J, E, N, N_over_g, LTE_N, LTE_N_over_g = loadtxt(filename, skiprows=4, unpack=True) #Read in H2 column density file
	h2_transitions = make_line_list() #Make H2 transitions object
	for i in xrange(len(v)): #Loop through each rovibrational energy level
		found_transitions = (h2_transitions.V.u == v[i]) & (h2_transitions.J.u == J[i]) #Find all rovibrational transitions that match the upper v and J
		h2_transitions.N[found_transitions] = N_over_g[i] #Set column density of transitions
	h2_transitions.N = h2_transitions.N / h2_transitions.N[h2_transitions.label == '1-0 S(1)']
	return(h2_transitions)
	
		    
##Store differential extinction between two transitions from the same upper state
class differential_extinction:
	def __init__(self, waves, A, sigma): #Input the lambda, flux, and sigma of two different lines as paris [XX,XX]
		self.waves = waves  #Store the wavleneghts as lamda[0] and lambda[1]
		self.A = A #Store differential extinction
		self.sigma = sigma #Store uncertainity in differential extinction A
	def fit_curve(self):
		constants = lambda0**(-alpha) / ( self.waves[0]**(-alpha) - self.waves[1]**(-alpha) )
		#constants = lambda0**alpha / ( self.waves[0]**(-alpha) - self.waves[1]**(-alpha) ) #Calculate constants to mulitply A_delta_lambda by to get A_K
		self.A_K = self.A * constants #calculate extinction for a given power law alpha
		self.sigma_A_K = self.sigma * constants #calculate extinction for a given power law alpha
					

#Class to store information on H2 transition, with flux can calculate column density
class h2_transitions:
	def __init__(self, J, V, E, A):
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
		self.sig_rot_T = zeros(n_lines) #Store uncertainity in rotation temperature fit
		self.res_rot_T =  zeros(n_lines) #Store residuals from offset of line fitting rotation temp
		self.sig_res_rot_T =  zeros(n_lines) #Store uncertainity in residuals from fitting rotation temp (e.g. using covariance matrix)
	def calculate_column_density(self, normalize=True): #Calculate the column density and uncertainity for a line's given upper state from the flux and appropriate constants
		#self.N = self.F / (self.g * self.E.u * h * c * self.A)
		#self.Nsigma = self.sigma /  (self.g * self.E.u * h * c * self.A)
		self.N = self.F / (self.g * self.E.diff() * h * c * self.A)
		self.Nsigma = self.sigma /  (self.g * self.E.diff() * h * c * self.A)
		if normalize: #By default normalize to the 1-0 S(1) line, set normalize = False if using absolute flux calibrated data
			N_10_S1 = self.N[self.label == '1-0 S(1)'] #Grab column density derived from 1-0 S(1) line
			self.N = self.N / N_10_S1 #Normalize column densities
			self.Nsigma = self.Nsigma / N_10_S1 #Normalize uncertainity
	def makelabel(self): #Make labels for each transition in spectroscopic notation.
		labels = []
		for i in xrange(self.n_lines):
			labels.append(self.V.label[i] + ' ' + self.J.label[i])
		return array(labels)
	def upper_state(self, label, wave_range = [0,999999999.0]): #Given a label in spectroscopic notation, list transitions with same upper state (and a wavelength range if specified)
		i = self.label == label
		Ju = self.J.u[i]
		Vu = self.V.u[i]
		found_transitions = (self.wave > wave_range[0]) & (self.wave < wave_range[1]) & (self.J.u == Ju) & (self.V.u == Vu)
		label_subset = self.label[found_transitions]
		wave_subset = self.wave[found_transitions]
		for i in xrange(len(label_subset)):
			print label_subset[i] + '\t' + str(wave_subset[i])
		#print self.label[found_transitions]#Find all matching transitions in the specified wavelength range with a matching upper J and V state
		#print self.wave[found_transitions]
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
			for i in xrange(n): #Loop through each line\
				matched_line = (self.label == region.label[i])
				if any(matched_line): #If any matches are found...
					self.F[matched_line] = region.flux[i] #And set flux
					self.s2n[matched_line] = region.s2n[i] #Set S/N for a single line
					self.sigma[matched_line] = region.sigma[i] #Set sigma (uncertainity) for a single line
	def read_model(self, labels, flux): #Read in fluxes from model
		for i in xrange(len(labels)): #Loop through each line
			matched_line = (self.label == labels[i]) #Match to H2 line
			self.F[matched_line] = flux[i] #Set flux to flux from model

	def quick_plot(self): #Create quick boltzmann diagram for previewing and testing purposes
		nonzero = self.N != 0.0
		clf()
		plot(self.T[nonzero], log(self.N[nonzero]), 'o')
		ylabel("Column Density   log$_e$(N/g) [cm$^{-2}$]", fontsize=18)
   		xlabel("Excitation Energy     (E/k)     [K]", fontsize=18)
   		show()
   	def make_latex_table(self, output_filename, s2n_cut = 3.0): #Output a latex table of column densities for each H2 line
   		lines = []
   		#lines.append(r"\begin{table}")  #Set up table header
   		lines.append(r"\begin{longtable}{lrrr}")
   		lines.append(r"\caption{\htwo{} rovibrational state column densities}{} \label{tab:coldens} \\")
   		#lines.append("\begin{scriptsize}")
   		#lines.append(r"\begin{tabular}{cccc}")
   		lines.append(r"\hline")
   		lines.append(r"\htwo{} line & $v_u$ & $J_u$ &  $\log_{10} \left(N_i / N_{\mbox{\tiny 1-0 S(1)}} \right)$ \\")
   		lines.append(r"\hline\hline")
   		lines.append(r"\endfirsthead")
   		lines.append(r"\hline")
   		lines.append(r"\htwo{} line & $v_u$ & $J_u$ &  $\log_{10} \left(N_i / N_{\mbox{\tiny 1-0 S(1)}} \right)$ \\")
   		lines.append(r"\hline\hline")
   		lines.append(r"\endhead")
   		lines.append(r"\hline")
   		lines.append(r"\endfoot")
   		lines.append(r"\hline")
   		lines.append(r"\endlastfoot")
   		highest_v = max(self.V.u[self.s2n > s2n_cut]) #Find highest V level
   		for v in range(1,highest_v+1): #Loop through each rotation ladder
   			i = (self.V.u == v) & (self.s2n > s2n_cut) #Find all lines in the current ladder
   			s = argsort(self.J.u[i]) #Sort by upper J level
   			labels = self.label[i][s] #Grab line labels
   			J =  self.J.u[i][s] #Grab upper J
   			N = self.N[i][s] #Grab column density N
   			sig_N =  self.Nsigma[i][s] #Grab uncertainity in N
   			for j in xrange(len(labels)):
				#lines.append(labels[j] + " & " + str(v) + " & " + str(J[j]) + " & " + "%1.2e" % N[j] + " $\pm$ " + "%1.2e" %  sig_N[j] + r" \\") 
				lines.append(labels[j] + " & " + str(v) + " & " + str(J[j]) + " & $" + "%1.2f" % log10(N[j]) 
					+ r"^{+%1.2f" % (-log10(N[j]) + log10(N[j]+sig_N[j]))   +r"}_{%1.2f" % (-log10(N[j]) + log10(N[j]-sig_N[j])) +r"} $ \\") 
   		#lines.append(r"\hline\hline")
		#lines.append(r"\end{tabular}")
		lines.append(r"\end{longtable}")
		#lines.append(r"\end{table}")
		savetxt(output_filename, lines, fmt="%s") #Output table
	def fit_rot_temp(self, T, log_N, y_error_bars, s2n_cut = 1., color='black', dotted_line=False, rot_temp_energy_limit=0.): #Fit rotation temperature to a given ladder in vibration
		log_N_sigma = nanmax(y_error_bars, 0) #Get largest error in log space
		if rot_temp_energy_limit > 0.: #If user specifies to cut rotation temp fit, use that....
			usepts = T < rot_temp_energy_limit
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
		plot(T, y, color=color, linestyle=linestyle) #Plot T rot fit
		#plot(T, y+y_sigma, color=color, linestyle='--') #Plot uncertainity in T rot fit
		#plot(T, y-y_sigma, color=color, linestyle='--')
		rot_temp = -1.0/slope #Calculate the rotation taemperature
		sigma_rot_temp = rot_temp * (sigma_slope/abs(slope)) #Calculate uncertainity in rotation temp., basically just scale fractional error
		print 'rot_temp = ', rot_temp,'+/-',sigma_rot_temp
		residuals = e**log_N - e**y #Calculate residuals in fit, but put back in linear space
		sigma_residuals = sqrt( (e**(y + y_sigma) - e**y)**2 + (e**(log_N + log_N_sigma)-e**log_N)**2 ) #Calculate uncertainity in residuals from adding uncertainity in fit and data points together in quadarature
		return rot_temp, sigma_rot_temp, residuals, sigma_residuals
	def compare_model(self, h2_model): #Make a Boltzmann diagram comparing a model (ie. Cloudy) to data
		h2_model.v_plot(orthopara_fill=False, empty_fill=True, clear=True, show_legend=False, savepdf=False, show_labels=True) #Plot model points as empty symbols
		self.v_plot(orthopara_fill=False, full_fill=True, clear=False, show_legend=False, savepdf=False)
   	def v_plot(self, plot_single_temp = False, show_upper_limits = True, nocolor = False, V=[-1], s2n_cut=-1.0, normalize=True, savepdf=True, orthopara_fill=True, empty_fill =False, full_fill=False,
   				show_labels=False, x_range=[0.,0.], y_range=[0.,0.], rot_temp=False, show_legend=True, rot_temp_energy_limit=0., fname='', clear=True): #Make simple plot first showing all the different rotational ladders for a constant V
		if fname=='': #Automatically give file name if one is not specified by user
			fname = self.path + '_excitation_diagram.pdf'
		with PdfPages(fname) as pdf: #Make a pdf
			nonzero = self.N != 0.0
			if clear: #User can specify if they want to clear the plot
				clf()
			symbsize = 9 #Size of symbols on excitation diagram
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
					current_color = color_list[i]
					current_symbol = 'o'
				ortho = (self.J.u % 2 == 1) &  (self.V.u == i) & (self.s2n > s2n_cut)  #Select only states for ortho-H2, which has the proton spins aligned so J can only be odd (1,3,5...)
				ortho_upperlimit = (self.J.u % 2 == 1) &  (self.V.u == i) & (self.s2n <= s2n_cut) #Select ortho-H2 lines where there is no detection (e.g. S/N <= 1)
				if any(ortho): #If datapoints are found...
					log_N = log(self.N[ortho]) #Log of the column density
					if sum(self.s2n[ortho]) == 0.:
						plot(self.T[ortho], log_N, current_symbol,  color=current_color, label=' ', markersize=symbsize, fillstyle=orthofill)  #Plot data + error bars
					else:
						y_error_bars = [abs(log_N - log(self.N[ortho]-self.Nsigma[ortho])), abs(log_N - log(self.N[ortho]+self.Nsigma[ortho]))] #Calculate upper and lower ends on error bars
						errorbar(self.T[ortho], log_N, yerr=y_error_bars, fmt=current_symbol,  color=current_color, label=' ', capthick=3, markersize=symbsize, fillstyle=orthofill)  #Plot data + error bars
						if show_upper_limits:
							test = errorbar(self.T[ortho_upperlimit], log(self.Nsigma[ortho_upperlimit]*3.0), yerr=1.0, fmt=current_symbol,  color=current_color, capthick=3, uplims=True, markersize=symbsize, fillstyle=orthofill) #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
						if show_labels: #If user wants to show labels for each of the lines
							for j in xrange(len(log_N)): #Loop through each point to label
								text(self.T[ortho][j], log_N[j], '        '+self.label[ortho][j], fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black')  #Label line with text
						print 'For ortho v=', i
						if rot_temp and len(log_N) > 1: #If user specifies fit rotation temperature
							#stop()
							rt, srt, residuals, sigma_residuals = self.fit_rot_temp(self.T[ortho], log_N, y_error_bars, s2n_cut=s2n_cut, color=current_color, dotted_line=False, rot_temp_energy_limit=rot_temp_energy_limit) #Fit rotation temperature
							self.rot_T[ortho] = rt #Save rotation temperature for individual lines
							self.sig_rot_T[ortho] = srt #Save rotation tempreature uncertainity for individual lines
							self.res_rot_T[ortho] = residuals #Save residuals for individual data points from the rotation tmeperature fit
							self.sig_res_rot_T[ortho] = sigma_residuals #Save the uncertainity in the residuals from the rotation temp fit (point uncertainity and fit uncertainity added in quadrature)
				else: #Else if no datapoints are found...
					errorbar([nan], [nan], yerr=1.0, fmt=current_symbol,  color=current_color, label=' ', capthick=3, markersize=symbsize, fillstyle=orthofill)  #Do empty plot to fill legend
				
			for i in use_upper_v_states:
				if nocolor: #If user specifies no color,
					current_color = 'Black'
					current_symbol = symbol_list[i]
				else: #Or else by default use colors from the color list defined at the top of the code
					current_color = color_list[i]
					current_symbol = '^'
				para = (self.J.u % 2 == 0) & (self.V.u == i) & (self.s2n > s2n_cut) #Select only states for para-H2, which has the proton spins anti-aligned so J can only be even (0,2,4,...)
				para_upperlimit =  (self.J.u % 2 == 0) & (self.V.u == i) & (self.s2n <= s2n_cut) #Select para-H2 lines where there is no detection (e.g. S/N <= 1)
				if any(para): #If datapoints are found...
					log_N = log(self.N[para]) #Log of the column density
					if sum(self.s2n[para]) == 0.:
						plot(self.T[para], log_N, current_symbol,  color=current_color, label='v='+str(i), markersize=symbsize, fillstyle=parafill)  #Plot data + error bars
					else:
						y_error_bars = [abs(log_N - log(self.N[para]-self.Nsigma[para])), abs(log_N - log(self.N[para]+self.Nsigma[para]))] #Calculate upper and lower ends on error bars
						errorbar(self.T[para], log_N, yerr=y_error_bars, fmt=current_symbol,  color=current_color, label='v='+str(i), capthick=3, markersize=symbsize, fillstyle=parafill)  #Plot data + error bars
						if show_upper_limits:
							test = errorbar(self.T[para_upperlimit], log(self.Nsigma[para_upperlimit]*3.0), yerr=1.0, fmt=current_symbol,  color=current_color, capthick=3, uplims=True, markersize=symbsize, fillstyle=parafill) #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
						if show_labels: #If user wants to show labels for each of the lines
							for j in xrange(len(log_N)): #Loop through each point to label
								text(self.T[para][j], log_N[j], '        '+self.label[para][j], fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black')  #Label line with text
						print 'For para v=', i
						if rot_temp and len(log_N) > 1: #If user specifies fit rotation temperature
							rt, srt, residuals, sigma_residuals = self.fit_rot_temp(self.T[para], log_N, y_error_bars, s2n_cut=s2n_cut, color=current_color, dotted_line=True, rot_temp_energy_limit=rot_temp_energy_limit) #Fit rotation temperature
							self.rot_T[para] = rt #Save rotation temperature for individual lines
							self.sig_rot_T[para] = srt #Save rotation tempreature uncertainity for individual lines
							self.res_rot_T[para] = residuals #Save residuals for individual data points from the rotation tmeperature fit
							self.sig_res_rot_T[para] = sigma_residuals #Save the uncertainity in the residuals from the rotation temp fit (point uncertainity and fit uncertainity added in quadrature)						
				else: #Else if no datapoints are found...
					errorbar([nan], [nan], yerr=1.0, fmt=current_symbol,  color=current_color, label='v='+str(i), capthick=3, markersize=symbsize, fillstyle=parafill)  #Do empty plot to fill legend

			tick_params(labelsize=14) #Set tick mark label size
			if normalize: #If normalizing to the 1-0 S(1) line
				ylabel("Column Density   ln(N$_i$/g$_i$)-ln(N$_{r}$/g$_{r}$)", fontsize=labelsize)
			else:  #If using absolute flux calibrated data
				ylabel("Column Density   pn(N/g) [cm$^{-2}$]", fontsize=labelsize)
			xlabel("Excitation Energy     (E$_i$/k)     [K]", fontsize=labelsize, labelpad=4)
			if x_range[1] == 0.0: #If user does not specifiy a range for the x-axis
				xlim([0,1.4*max(self.T[self.s2n >= s2n_cut])]) #Autoscale
			else: #Else if user specifies range
				xlim(x_range) #Use user specified range for x-axis
			if y_range[1] != 0.0: #If user specifies a y axis limit, use it
				ylim(y_range) #Use user specified y axis range
			if show_legend: #If user does not turn off showing the legend
				legend(loc=1, ncol=2, fontsize=18, numpoints=1, columnspacing=-0.5, title = 'ortho  para  ladder')
			if plot_single_temp: #Plot a single temperature line for comparison, if specified
				x = arange(0,20000, 10)
				plot(x, single_temp_y_intercept - (x / single_temp), linewidth=2, color='orange')
				midpoint = size(x)/2
				text(0.7*x[midpoint], 0.7*(single_temp_y_intercept - (x[midpoint] / single_temp)), "T = "+str(single_temp)+" K", color='orange')
			#show()
			draw()
			if savepdf:
				pdf.savefig() #Add in the pdf
			#stop()



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
		for i in xrange(n):
			if delta_J[i] == -2:
				J_labels.append('O(' + str(self.l[i]) + ')')  #Create label O(J_l) for transitions where delta-J = -2
			elif delta_J[i] == 0:
				J_labels.append('Q(' + str(self.l[i]) + ')') #Create label Q(J_l) for transitions where delta-J = 0
			elif delta_J[i] == 2:
				J_labels.append('S(' + str(self.l[i]) + ')') #Create label S(J_l) for transitions where delta-J = +2
		return J_labels


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
		for i in xrange(n):
			V_labels.append( str(self.u[i]) + '-' + str(self.l[i]) ) #Create label for V transitions of V_u-V_l
		return V_labels

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

#Object for storing column densities of individual levels, and performing calculations upon them
class states:
	def __init__(self):
		ion() #Set up plotting to be interactive
		show() #Open a plotting window
		V, J = loadtxt(energy_table, usecols=(0,1), unpack=True, dtype='int', skiprows=1) #Read in data for H2 ground state rovibrational energy levels
		E = loadtxt(energy_table, usecols=(2,), unpack=True, dtype='float', skiprows=1)
		self.n_states = len(V) #Number of levels
		self.V = V #Array to store vibration level
		self.J = J #Array to store rotation level
		self.T = E / k #Excited energy above the ground rovibrational state in units of Kelvin
		self.N = zeros(self.n_states) #Array for storing column densities
		self.transitions = make_line_list() #Create transitions list 
		self.convergence = [] #Set up python list that will hold convergence of cascade
	def total_N(self): #Grab total column density N and return it
		return nansum(self.N) #Return total column density of H2
	def cascade(self, time=1.0, temp=100.0, quick=0, showplot=True): #Do a step in the radiative cascade
		store_delta_N = zeros(self.n_states)  #Set up array to store all the changes in N 
		V = self.V #Assign variables to speed up loop
		J = self.J
		N = self.N
		trans_V_u = self.transitions.V.u
		trans_V_l = self.transitions.V.l
		trans_J_u = self.transitions.J.u
		trans_J_l = self.transitions.J.l
		trans_A = self.transitions.A
		if quick != 0: #If user wants to speed up the cascade
			maxthresh = -partsort(-trans_A,quick)[quick-1] #Find threshold for maximum
		else:
			maxthresh = 0. #E
		for i in xrange(len(self.transitions.J.u)): #Loop through each transition
			if trans_A[i] > maxthresh:
				#Ju = self.transitions.J.u[i] #Grab upper and lower J levels for this transition
				#Jl = self.transitions.J.l[i]
				#Vu = self.transitions.V.u[i] #Grab upper and lower V levels for this transition
				#Vl = self.transitions.V.l[i]
				upper_state = logical_and(V == trans_V_u[i], J == trans_J_u[i])#Finder upper state of transition
				lower_state = logical_and(V == trans_V_l[i], J == trans_J_l[i])#Find loer state of transition
				delta_N = N[upper_state]*trans_A[i]*time #Move this much H2 around with this transition
				#print 'delta_N=', delta_N
				#delta_N = self.N[upper_state]*(1.0 - exp(-self.transitions.A[i]*time) )#Move this much H2 around with this transition
				store_delta_N[upper_state] = store_delta_N[upper_state] - delta_N #Store change in N taken out of upper state by this transition
				store_delta_N[lower_state] = store_delta_N[lower_state] + delta_N #Store change in N put into lower state by this transition
		old_N = copy.deepcopy(self.N)
		N = N + store_delta_N #Modfiy level populations after the effects of all the transitions have been summed up
		j = V == 12
		exponential = exp(-self.T[j]/temp) #Calculate boltzmann distribution for user given temperature, used to populate energy levels
		boltmann_distribution = exponential / nansum(exponential)
		#self.N[1:] = self.N[1:] + self.N[0] / (float(self.n_states)-1.0) #Crudely redistribute everything in the ground state back to all other states
 		#N[1:] = N[1:] + boltmann_distribution[1:]*N[0]
 		#N[j] = N[j] + boltmann_distribution*N[0]
 		N[j] = N[j] + boltmann_distribution*time
 		#N[285] = N[285] + N[0] #Test just dumping everything into the final level and let everything cascade out of it.
 		
 		#N[0]  = 0.0 #Empty out ground state after redistributing all the molecules in the ground
 		convergence_measurement = nansum((N-old_N)**2)
 		print 'convergence = ', convergence_measurement
 		self.convergence.append(convergence_measurement)#Calculate convergence from one step to the 
 		self.N = N
 		if showplot:
 			self.set_transition_column_densities()
	 		self.transitions.v_plot(s2n_cut=-1.0, savepdf=False)
 		#stop()
 	def set_transition_column_densities(self): #Put column densities into 
		for i in xrange(self.n_states): #Loop through each level
			upper_states = (self.transitions.J.u == self.J[i]) & (self.transitions.V.u == self.V[i]) #Find all transitions with upper states in a level
			self.transitions.N[upper_states] = self.N[i] #Set new column densities to the transitions object
