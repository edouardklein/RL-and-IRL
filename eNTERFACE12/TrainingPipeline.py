#!/usr/bin/python
from pylab import *
from collections import deque
from Pipeline import *

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
    l_aa = get_action( g_aCommand )
    for l in l_as:
        print l[0]," ",
    print l_aa[0]
    # print "erycdjqsvhbfqsdfs"
    # print g_aGoal
    # print g_aPosition_t
    # print g_aCommand
    g_aPosition_t = update_position( g_aPosition_t, g_aCommand )
    if near_enough( g_aGoal, g_aPosition_t ):
        g_bContinue = False
    


