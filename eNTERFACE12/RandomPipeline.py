#!/usr/bin/python
from pylab import *
from collections import deque
from Pipeline import *
from random import choice
from phipsi import *

g_aPosition_t = map(int,rand(2)*[3,4])
g_iNb_Samples = 0
g_iNb_Episodes = 0
g_iM = 100 #Max Number of episodes
g_iN = 10 #Max Length of an episode

g_mTheta_psi = genfromtxt("theta_scirl.mat")

for g_iNb_Episodes in range(0,g_iM):
    g_aPosition_t = map(int,rand(2)*[3,4])
    l_as = None
    l_aa = None
    l_asdash = None
    for g_iNb_Samples in range( 0, g_iN ):
        command = zeros([4,1])
        command[choice([0,1,2,3])] = 1
        noisy_com = add_noise( command )
        l_asdash = get_rl_state( g_aPosition_t, noisy_com )
        l_eoe = 1 if g_iNb_Samples < g_iN-1 else 0
        if( l_as != None ):
            l_reward = dot( g_mTheta_psi.transpose(), psi( l_as ) )
            for x in concatenate([l_as, l_aa, l_asdash, l_reward,[l_eoe]]):
                print "%d "%x,
            print
        l_as = l_asdash
        l_aa = get_action( command )
        g_aPosition_t = update_position( g_aPosition_t, command )
