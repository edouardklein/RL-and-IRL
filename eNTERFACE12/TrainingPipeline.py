#!/usr/bin/python
from pylab import *
from Pipeline import *
from random import choice

g_aGoal = choice([array([0,3]),array([2,3])])
g_aPosition_t = array([1,0])
g_bContinue = 2

while g_bContinue:
    g_aCommand = command( g_aGoal, g_aPosition_t )
    g_aNoisyCom = add_noise( g_aCommand )
    l_as = get_rl_state( g_aPosition_t, g_aNoisyCom )
    l_aa = get_action( g_aCommand )
    for l in l_as:
        print l," ",
    print l_aa[0]
    # print "erycdjqsvhbfqsdfs"
    # print g_aGoal
    # print g_aPosition_t
    # print g_aCommand
    if near_enough( g_aGoal, g_aPosition_t ):
        g_bContinue-=1
    g_aPosition_t = update_position( g_aPosition_t, g_aCommand )
    


