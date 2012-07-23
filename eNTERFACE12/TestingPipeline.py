#!/usr/bin/python
from pylab import *
from collections import deque
from Pipeline import *
import sys
sys.path+=['..']
from DP import *
from phipsi import *
ACTION_FILE = "actions.mat"
actions = genfromtxt( ACTION_FILE )
omega = genfromtxt( "omega_expert.mat" )
pi = lambda s: greedy_policy( s, omega, phi, actions )

def get_command( a ):
    answer = zeros([4,1]) #Four component, respectively right, up, left, down
    answer[ int(a) ] = 1
    return answer

def natural_command( aCommand ):
    #We select the biggest component
    answer = map( lambda x: [0] if x < max( aCommand ) else [1], aCommand )
    return array( answer )

g_iNaturalScore = 0
g_iIRLScore = 0
g_iNbTrials = 0

g_aGoal = rand(2,1)*g_fXYMax
g_aPosition_t = rand(2,1)*g_fXYMax
while near_enough( g_aGoal, g_aPosition_t ):
    g_aPosition_t = rand(2,1)*g_fXYMax
g_bContinue = True

while g_bContinue:
    g_aCommand = command( g_aGoal, g_aPosition_t )
    g_aNoisyCom = add_noise( g_aCommand )
    g_qPositions.append( g_aPosition_t )
    l_as = get_rl_state( g_qPositions, g_aNoisyCom )
    #l_aa = get_action( g_aCommand )
    l_aa = pi( l_as )
    l_aRLCommand = get_command( l_aa )
    l_aNaturalCommand = natural_command( g_aNoisyCom )
    if( all( g_aCommand == l_aNaturalCommand ) ):
        g_iNaturalScore += 1
    if( all( g_aCommand == l_aRLCommand ) ):
        g_iIRLScore += 1
    # else:
    #     print "True Command : "
    #     print g_aCommand
    #     print "IRL Command : "
    #     print l_aRLCommand
    #     print "EqualityArray : "
    #     print g_aCommand == l_aRLCommand
    g_iNbTrials += 1
    #g_aPosition_t = update_position( g_aPosition_t, g_aCommand ) #Should be changed, but would not garantee the algorithm will stop
    g_aPosition_t = update_position( g_aPosition_t, l_aNaturalCommand )
    if near_enough( g_aGoal, g_aPosition_t ):
        g_bContinue = False
    

print "Natural Argmax action selection performance : %f"% (float(g_iNaturalScore)/float(g_iNbTrials))
print "IRL action selection performance : %f"% (float(g_iIRLScore)/float(g_iNbTrials))


