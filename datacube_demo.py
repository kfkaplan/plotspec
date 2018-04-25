#Test demo script for make_datacube.py library
from scipy import *
import make_datacube as cubelib #Import library to make datacubes

workdir =  '/Volumes/IGRINS_Data/datacube_demo/' #Set to where you want to save resulting fits files
vrange = [-10.0,10.0] #Velocity range

#Demo of saving files from datacube
cube = cubelib.data() #Create datacube object
cube.fill_gaps() #Fill in nans, this is optional, comments out if you don't want to do this
cube.savecube('1-0 S(1)', workdir+'1-0_S(1)_cube.fits') #Save datacube of an emission line
cube.saveimage('1-0 S(1)', workdir+'1-0_S(1)_img.fits', vrange=vrange) #Save image of an emission line in the velocity range "vrange", here set to +/- 10 km/s
cube.saveratio('2-1 S(1)', '1-0 S(1)', vrange=vrange,  fname=workdir+'ratio_21s1_10s1.fits') #Test save ratio maps