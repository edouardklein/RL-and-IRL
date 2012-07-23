#!/usr/bin/python
from pylab import *
import sys
sys.path+=['..']
from LSPI import *
from phipsi import *
D_FILE_NAME = "RandomSamples.mat"
ACTION_FILE = "actions.mat"

print "Training the expert..."
D = genfromtxt( D_FILE_NAME )
actions = genfromtxt( ACTION_FILE )
omega_expert = LSPI( D, STATE_DIM, 1, phi, PHI_DIM, actions, 0.01, 50 )
savetxt( "omega_expert.mat", omega_expert )

