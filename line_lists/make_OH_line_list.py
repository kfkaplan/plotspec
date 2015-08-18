#simple script for making a plotspec.py compatible OH line list based on the OH line list from http://www.gemini.edu/sciops/instruments/nir/wavecal/index.html
from pylab import *

line_data = loadtxt('OH_raw_rousselot_2000.dat') #Read in data from original line list
bright_lines = line_data[:,1] > 1e-1 #Find only bright lines we will probably see
line_waves = line_data[bright_lines, 0] / 10000.0 #Convert line wavelengths from  angstroms to microns
output = open('OH_Rousselot_2000.dat', 'w') #Open output line list
for line in line_waves: #Loop through each line
	output.write(str(line) + '\t{OH}\n') #Write line to line list
output.close() #Close line list, you are now done!