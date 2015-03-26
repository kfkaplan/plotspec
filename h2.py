from pdb import set_trace as stop #Use stop() for debugging
#from scipy import *
from pylab import *

#Global variables
cloudy_h2_data_dir = 'data/' #Directory where H2 data is stored for cloudy
energy_table = cloudy_h2_data_dir + 'energy_X.dat' #Name of table where Cloudy stores data on H2 electronic ground state rovibrational energies
transition_table = cloudy_h2_data_dir + 'transprob_X.dat' #Name of table where Cloudy stores data on H2 transition probabilities (Einstein A coeffs.)
k = 0.69503476 #Botlzmann constant k in units of cm^-1 K^-1 (from http://physics.nist.gov/cuu/Constants/index.html)
single_temp = 4000 #K
single_temp_y_intercept = 15.0
alpha = arange(0.0, -10.0, -0.01) #Save range of power laws to fit extinction curve [A_lambda = A_lambda0 * (lambda/lambda0)^alpha
lambda0 = 2.12 #Wavelength in microns for normalizing the power law exctinoction curve, here it is set to the K-badn at 2.12 um
#Make array of color names
color_list = []
for c in matplotlib.colors.cnames:
    color_list.append(c)


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
	
	i = (transitions.N != 0.0) & (transitions.s2n > 10.0) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
	J_upper_found = unique(transitions.J.u[i]) #Find J for all (detected) transition upper states
	V_upper_found = unique(transitions.V.u[i]) #Find V for all (detected) transition upper states
	for V in V_upper_found: #Check each upper V for pairs	
		pairs = [] #Set up array of line pairs for measuring the differential extinction A_lamba1-lambda2
		for J in J_upper_found: #Check each upper J for pairs
			match_upper_states = (transitions.J.u[i] == J) & (transitions.V.u[i] == V) #Find all transitions from the same upper J and V state
			waves = transitions.wave[i][match_upper_states] #Store wavelengths of all found transitions
			#N = transitions.N[i][match_upper_states] #Store all column densities for found transitions
			F = transitions.F[i][match_upper_states] 
			Fsigma = transitions.sigma[i][match_upper_states] 
			intrinsic_constants = 1.0 / (transitions.g[i][match_upper_states] * transitions.E.u[i][match_upper_states] * transitions.A[i][match_upper_states]) #Get constants for calculating the intrinsic ratios
			#Nsigma = transitions.Nsigma[i][match_upper_states] #Grab uncertainity in column densities
			if len(waves) == 2: #If a single pair of lines from the same upper state are found, calculate differential extinction for this single pair
				A_delta_lambda = -2.5*log10((F[0]/F[1]) / (intrinsic_constants[0]/intrinsic_constants[1])) #Calculate differential extinction between two H2 lines
				sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[0]/F[0])**2 + (Fsigma[1]/F[1])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
				pair = differential_extinction([waves[0], waves[1]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
				paris = pairs.append(pair) #Save a single pair
			elif len(waves) == 3: #If three liens are found from the same upper state, calculate differential extinction from differences between all three lines
				#Pair 1
				A_delta_lambda = -2.5*log10((F[0]/F[1]) / (intrinsic_constants[0]/intrinsic_constants[1])) #Calculate differential extinction between two H2 lines
				sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[0]/F[0])**2 + (Fsigma[1]/F[1])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
				pair = differential_extinction([waves[0], waves[1]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
				paris = pairs.append(pair) #Save a single pair
				#Pair 2
				A_delta_lambda = -2.5*log10((F[0]/F[2]) / (intrinsic_constants[0]/intrinsic_constants[2])) #Calculate differential extinction between two H2 lines
				sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[0]/F[0])**2 + (Fsigma[2]/F[2])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
				pair = differential_extinction([waves[0], waves[2]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
				paris = pairs.append(pair) #Save a single pair
				#Pair 3
				A_delta_lambda = -2.5*log10((F[1]/F[2]) / (intrinsic_constants[1]/intrinsic_constants[2])) #Calculate differential extinction between two H2 lines
				sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[1]/F[1])**2 + (Fsigma[2]/F[2])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
				pair = differential_extinction([waves[1], waves[2]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
				paris = pairs.append(pair) #Save a single pair
		clf() #Clear plot field
		for pair in pairs: #Loop through each pair
			pair.fit_curve()
			plot(alpha, pair.A_K)
		xlabel('Alpha')
		ylabel('$A_K$')
		ylim([-2,20])
		suptitle('V = ' + str(V))
		stop()
	
	
		    
##Store differential extinction between two transitions from the same upper state
class differential_extinction:
	def __init__(self, waves, A, sigma): #Input the lambda, flux, and sigma of two different lines as paris [XX,XX]
		self.waves = waves  #Store the wavleneghts as lamda[0] and lambda[1]
		self.A = A #Store differential extinction
		self.sigma = sigma #Store uncertainity in differential extinction A
	def fit_curve(self):
		constants = lambda0**alpha / ( self.waves[0]**(-alpha) - self.waves[1]**(-alpha) ) #Calculate constants to mulitply A_delta_lambda by to get A_K
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
	def calculate_column_density(self):
		self.N = self.F / (self.g * self.E.u * self.A)
		self.Nsigma = self.sigma /  (self.g * self.E.u * self.A)
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
	def quick_plot(self): #Create quick boltzmann diagram for previewing and testing purposes
		nonzero = self.N != 0.0
		clf()
		plot(self.T[nonzero], log(self.N[nonzero]), 'o')
		ylabel("Column Density   log$_e$(N/g) [cm$^{-2}$]", fontsize=18)
   		xlabel("Excitation Energy     (E/k)     [K]", fontsize=18)
   		show()
   	def v_plot(self, plot_single_temp = False): #Make simple plot first showing all the different rotational ladders for a constant V
		nonzero = self.N != 0.0
		clf()
		for i in unique(self.V.u):
			ortho = (self.J.u % 2 == 1) &  (self.V.u == i) & (self.s2n > 1.0)  #Select only states for ortho-H2, which has the proton spins aligned so J can only be odd (1,3,5...)
			ortho_upperlimit = (self.J.u % 2 == 1) &  (self.V.u == i) & (self.s2n <= 1.0) #Select ortho-H2 lines where there is no detection (e.g. S/N <= 1)
			#stop()
			#plot(self.T[ortho], log(self.N[ortho]), 'o',label=' ',  color=color_list[i*2]) 
			if any(ortho):
				log_N = log(self.N[ortho]) #Log of the column density
				y_error_bars = [abs(log_N - log(self.N[ortho]-self.Nsigma[ortho])), abs(log_N - log(self.N[ortho]+self.Nsigma[ortho]))] #Calculate upper and lower ends on error bars
				errorbar(self.T[ortho], log_N, yerr=y_error_bars, fmt='o',  color=color_list[i*2], label=' ', capthick=3, markersize=8)  #Plot data + error bars
				test = errorbar(self.T[ortho_upperlimit], log(self.Nsigma[ortho_upperlimit]*3.0), yerr=1.0, fmt='o',  color=color_list[i*2], capthick=3, uplims=True, markersize=8, label='_nolegend_') #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
			#else:
				#plot([],[],  'o',  color=color_list[i*2], label=' ') #Make a plot of nothing if there are no datapoints to plot, to get an entry in the legend
		for i in unique(self.V.u):
			para = (self.J.u % 2 == 0) & (self.V.u == i) & (self.s2n > 1.0) #Select only states for para-H2, which has the proton spins anti-aligned so J can only be even (0,2,4,...)
			para_upperlimit =  (self.J.u % 2 == 0) & (self.V.u == i) & (self.s2n <= 1.0) #Select para-H2 lines where there is no detection (e.g. S/N <= 1)
			if any(para):
				log_N = log(self.N[para]) #Log of the column density
				y_error_bars = [abs(log_N - log(self.N[para]-self.Nsigma[para])), abs(log_N - log(self.N[para]+self.Nsigma[para]))] #Calculate upper and lower ends on error bars
				errorbar(self.T[para], log_N, yerr=y_error_bars, fmt='^',  color=color_list[i*2], label='V='+str(i), capthick=3, markersize=8)  #Plot data + error bars
				test = errorbar(self.T[para_upperlimit], log(self.Nsigma[para_upperlimit]*3.0), yerr=1.0, fmt='^',  color=color_list[i*2], capthick=3, uplims=True, markersize=8, label='_nolegend_') #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
				#plot(self.T[para], log(self.N[para]), '^', label='V='+str(i), color=color_list[i*2])
			#else:
				#plot([],[], '^',  color=color_list[i*2], label='V='+str(i)) #Make a plot of nothing if there are no datapoints to plot, to get an entry in the legend
		ylabel("Column Density   log$_e$(N/g) [cm$^{-2}$]", fontsize=18)
		xlabel("Excitation Energy     (E/k)     [K]", fontsize=18)
		xlim([0,1.35*max(self.T)])
		legend(loc=1, ncol=2, fontsize=12, numpoints=1, columnspacing=-0.5, title = 'ortho  para          ')
		if plot_single_temp: #Plot a single temperature line for comparison, if specified
		    plot(self.T, single_temp_y_intercept - (self.T / single_temp), linewidth=2, color='orange')
		    midpoint = size(self.T)/2
		    text(0.7*self.T[midpoint], 0.7*(single_temp_y_intercept - (self.T[midpoint] / single_temp)), "T = "+str(single_temp)+" K", color='orange')
		show()



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
	
