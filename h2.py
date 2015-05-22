from pdb import set_trace as stop #Use stop() for debugging
#from scipy import *
from pylab import *
from matplotlib.backends.backend_pdf import PdfPages  #For outputting a pdf with multiple pages (or one page)


#Global variables, modify
single_temp = 1500.0 #K
single_temp_y_intercept = 22.0
alpha = arange(0.0, -10.0, -0.01) #Save range of power laws to fit extinction curve [A_lambda = A_lambda0 * (lambda/lambda0)^alpha
lambda0 = 2.12 #Wavelength in microns for normalizing the power law exctinoction curve, here it is set to the K-badn at 2.12 um
wave_thresh = 0.5 #Set wavelength threshold (here 0.1 um) for trying to measure extinction, we need the line pairs to be far enough apart we can get a handle on the extinction

#Global variables, do not modify
cloudy_h2_data_dir = 'data/' #Directory where H2 data is stored for cloudy
energy_table = cloudy_h2_data_dir + 'energy_X.dat' #Name of table where Cloudy stores data on H2 electronic ground state rovibrational energies
transition_table = cloudy_h2_data_dir + 'transprob_X.dat' #Name of table where Cloudy stores data on H2 transition probabilities (Einstein A coeffs.)
k = 0.69503476 #Botlzmann constant k in units of cm^-1 K^-1 (from http://physics.nist.gov/cuu/Constants/index.html)
h = 6.6260755e-27 #Plank constant in erg s, used for converting energy in wave numbers to cgs
c = 2.99792458e10 #Speed of light in cm s^-1, used for converting energy in wave numbers to cgs

#Make array of color names
color_list = ['black','gray','yellow','blue','red','green','beige','magenta','darkgoldenrod','purple','bisque','darkolivegreen', 'cyan','darkorange','orange']
symbol_list = ['o','v','8','x','s','*','h','D','^','8','1','o','o','o'] #Symbol list for rotation ladders on black and white Boltzmann plot
#for c in matplotlib.colors.cnames:
    #color_list.append(c)


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
	i = (transitions.N != 0.0) & (transitions.s2n > 10.0) #Find only transitions where a significant measurement of the column density was made (e.g. lines where flux was measured)
	J_upper_found = unique(transitions.J.u[i]) #Find J for all (detected) transition upper states
	V_upper_found = unique(transitions.V.u[i]) #Find V for all (detected) transition upper states
	pairs = [] #Set up array of line pairs for measuring the differential extinction A_lamba1-lambda2
	for V in V_upper_found: #Check each upper V for pairs		
		for J in J_upper_found: #Check each upper J for pairs
			match_upper_states = (transitions.J.u[i] == J) & (transitions.V.u[i] == V) #Find all transitions from the same upper J and V state
			waves = transitions.wave[i][match_upper_states] #Store wavelengths of all found transitions
			#N = transitions.N[i][match_upper_states] #Store all column densities for found transitions
			F = transitions.F[i][match_upper_states] 
			Fsigma = transitions.sigma[i][match_upper_states] 
			intrinsic_constants = 1.0 / (transitions.g[i][match_upper_states] * transitions.E.u[i][match_upper_states] * transitions.A[i][match_upper_states]) #Get constants for calculating the intrinsic ratios
			#Nsigma = transitions.Nsigma[i][match_upper_states] #Grab uncertainity in column densities
			if len(waves) == 2 and abs(waves[0]-waves[1]) > wave_thresh: #If a single pair of lines from the same upper state are found, calculate differential extinction for this single pair
				A_delta_lambda = -2.5*log10((F[0]/F[1]) / (intrinsic_constants[0]/intrinsic_constants[1])) #Calculate differential extinction between two H2 lines
				sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[0]/F[0])**2 + (Fsigma[1]/F[1])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
				pair = differential_extinction([waves[0], waves[1]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
				pairs.append(pair) #Save a single pair
				n_doubles_found = n_doubles_found + 1
			elif len(waves) == 3: #If three liens are found from the same upper state, calculate differential extinction from differences between all three lines
				#Pair 1
				if abs(waves[0] - waves[1]) > wave_thresh: #check if pair of lines are far enough apart
					A_delta_lambda = -2.5*log10((F[0]/F[1]) / (intrinsic_constants[0]/intrinsic_constants[1])) #Calculate differential extinction between two H2 lines
					sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[0]/F[0])**2 + (Fsigma[1]/F[1])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
					pair = differential_extinction([waves[0], waves[1]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
					pairs.append(pair) #Save a single pair
				#Pair 2
				if abs(waves[0] - waves[2]) > wave_thresh: #check if pair of lines are far enoug7h apart
					A_delta_lambda = -2.5*log10((F[0]/F[2]) / (intrinsic_constants[0]/intrinsic_constants[2])) #Calculate differential extinction between two H2 lines
					sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[0]/F[0])**2 + (Fsigma[2]/F[2])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
					pair = differential_extinction([waves[0], waves[2]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
					pairs.append(pair) #Save a single pair
				#Pair 3
				if abs(waves[1] - waves[2]) > wave_thresh: #check if pair of lines are far enough apart
					A_delta_lambda = -2.5*log10((F[1]/F[2]) / (intrinsic_constants[1]/intrinsic_constants[2])) #Calculate differential extinction between two H2 lines
					sigma_A_delta_lambda = (2.5 / log(10.0)) * sqrt( (Fsigma[1]/F[1])**2 + (Fsigma[2]/F[2])**2 ) #Calculate uncertainity in the differential extinction between two H2 lines
					pair = differential_extinction([waves[1], waves[2]], A_delta_lambda, sigma_A_delta_lambda) #Store wavelengths, differential extinction, and uncertainity in a differential_extinction object
					pairs.append(pair) #Save a single pair
				n_trips_found = n_trips_found + 1
	clf() #Clear plot field
	for pair in pairs: #Loop through each pair
		pair.fit_curve()
		plot(alpha, pair.A_K)
	xlabel('Alpha')
	ylabel('$A_K$')
	ylim([-2,20])
	suptitle('V = ' + str(V))
	stop()
	print 'Number of pairs from same upper state = ', n_doubles_found
	print 'Number of tripples from same upper state = ', n_trips_found
	
	
		    
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
		self.path = '' #Store path for saving excitation diagram and other files, read in when reading in region with definition set_Flux
	def calculate_column_density(self): #Calculate the column density and uncertainity for a line's given upper state from the flux and appropriate constants
		self.N = self.F / (self.g * self.E.u * h * c * self.A)
		self.Nsigma = self.sigma /  (self.g * self.E.u * h * c * self.A)
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
	def quick_plot(self): #Create quick boltzmann diagram for previewing and testing purposes
		nonzero = self.N != 0.0
		clf()
		plot(self.T[nonzero], log(self.N[nonzero]), 'o')
		ylabel("Column Density   log$_e$(N/g) [cm$^{-2}$]", fontsize=18)
   		xlabel("Excitation Energy     (E/k)     [K]", fontsize=18)
   		show()
   	def v_plot(self, plot_single_temp = False, show_upper_limits = True, nocolor = False): #Make simple plot first showing all the different rotational ladders for a constant V
		with PdfPages(self.path + '_excitation_diagram.pdf') as pdf: #Make a pdf
			nonzero = self.N != 0.0
			clf()
			symbsize = 9 #Size of symbols on excitation diagram
			labelsize = 24 #Size of text for labels
			orthofill = 'full' #How symbols on excitation diagram are filled, 'full' vs 'none'
			parafill = 'none'
			for i in unique(self.V.u):
				if nocolor: #If user specifies no color,
					current_color = 'gray'
					current_symbol = symbol_list[i-1]
				else: #Or else by default use colors from the color list defined at the top of the code
					current_color = color_list[i]
					current_symbol = 'o'
				ortho = (self.J.u % 2 == 1) &  (self.V.u == i) & (self.s2n > 1.0)  #Select only states for ortho-H2, which has the proton spins aligned so J can only be odd (1,3,5...)
				ortho_upperlimit = (self.J.u % 2 == 1) &  (self.V.u == i) & (self.s2n <= 1.0) #Select ortho-H2 lines where there is no detection (e.g. S/N <= 1)
				#stop()
				#plot(self.T[ortho], log(self.N[ortho]), 'o',label=' ',  color=color_list[i*2]) 
				if any(ortho):
					log_N = log(self.N[ortho]) #Log of the column density
					y_error_bars = [abs(log_N - log(self.N[ortho]-self.Nsigma[ortho])), abs(log_N - log(self.N[ortho]+self.Nsigma[ortho]))] #Calculate upper and lower ends on error bars
					errorbar(self.T[ortho], log_N, yerr=y_error_bars, fmt=current_symbol,  color=current_color, label=' ', capthick=3, markersize=symbsize, fillstyle=orthofill)  #Plot data + error bars
					if show_upper_limits:
						test = errorbar(self.T[ortho_upperlimit], log(self.Nsigma[ortho_upperlimit]*3.0), yerr=1.0, fmt=current_symbol,  color=current_color, capthick=3, uplims=True, markersize=symbsize, fillstyle=orthofill) #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
				#else:
					#errorbar([], [], yerr=1.0, fmt='o',  color=color_list[i], label=' ', capthick=3, markersize=8)  #Plot data + error bars
					#plot([],[],  'o',  color=color_list[i*3], label=' ') #Make a plot of nothing if there are no datapoints to plot, to get an entry in the legend
			for i in unique(self.V.u):
				if nocolor: #If user specifies no color,
					current_color = 'Black'
					current_symbol = symbol_list[i-1]
				else: #Or else by default use colors from the color list defined at the top of the code
					current_color = color_list[i]
					current_symbol = '^'
				para = (self.J.u % 2 == 0) & (self.V.u == i) & (self.s2n > 1.0) #Select only states for para-H2, which has the proton spins anti-aligned so J can only be even (0,2,4,...)
				para_upperlimit =  (self.J.u % 2 == 0) & (self.V.u == i) & (self.s2n <= 1.0) #Select para-H2 lines where there is no detection (e.g. S/N <= 1)
				if any(para):
					log_N = log(self.N[para]) #Log of the column density
					y_error_bars = [abs(log_N - log(self.N[para]-self.Nsigma[para])), abs(log_N - log(self.N[para]+self.Nsigma[para]))] #Calculate upper and lower ends on error bars
					errorbar(self.T[para], log_N, yerr=y_error_bars, fmt=current_symbol,  color=current_color, label='V='+str(i), capthick=3, markersize=symbsize, fillstyle=parafill)  #Plot data + error bars
					if show_upper_limits:
						test = errorbar(self.T[para_upperlimit], log(self.Nsigma[para_upperlimit]*3.0), yerr=1.0, fmt=current_symbol,  color=current_color, capthick=3, uplims=True, markersize=symbsize, fillstyle=parafill) #Plot 1-sigma upper limits on lines with no good detection (ie. S/N < 1.0)
					#plot(self.T[para], log(self.N[para]), '^', label='V='+str(i), color=color_list[i])
				#else:
					#errorbar([], [], yerr=1.0, fmt='^',  color=color_list[i], label='V='+str(i), capthick=3, markersize=8)  #Plot data + error bars
					#plot([],[], '^',  color=color_list[i*3], label='V='+str(i)) #Make a plot of nothing if there are no datapoints to plot, to get an entry in the legend
			tick_params(labelsize=14) #Set tick mark label size
			ylabel("Column Density   log$_e$(N/g) [cm$^{-2}$]", fontsize=labelsize)
			xlabel("Excitation Energy     (E/k)     [K]", fontsize=labelsize)
			xlim([0,1.35*max(self.T)])
			legend(loc=1, ncol=2, fontsize=18, numpoints=1, columnspacing=-0.5, title = 'ortho  para          ')
			if plot_single_temp: #Plot a single temperature line for comparison, if specified
				x = arange(0,20000, 10)
				plot(x, single_temp_y_intercept - (x / single_temp), linewidth=2, color='orange')
				midpoint = size(x)/2
				text(0.7*x[midpoint], 0.7*(single_temp_y_intercept - (x[midpoint] / single_temp)), "T = "+str(single_temp)+" K", color='orange')
			show()
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
	
