#!/usr/bin/python
from pylab import *
import sys
from phipsi import *

g_iNbEpisodes = int(sys.argv[1])

g_lEpisodes = []
g_msasreoeEps = None

for i in range(1,g_iNbEpisodes+1):
    episode = genfromtxt( "TrainingSA-%d.mat"%i )
    g_lEpisodes.append( episode )

for episode in g_lEpisodes:
    sasreoeEp = zeros( array(episode.shape) + array([-1,STATE_DIM+1+1]) )
    for i in range( 0, len(episode)-1 ):
        sasreoeEp[i,0:STATE_DIM+1] = episode[i] #sa
        sasreoeEp[i,STATE_DIM+1:STATE_DIM+1+STATE_DIM] = episode[i+1,0:STATE_DIM] #s'
        sasreoeEp[i,STATE_DIM+1+STATE_DIM:STATE_DIM+1+STATE_DIM+1] = 0 #r
        sasreoeEp[i,STATE_DIM+1+STATE_DIM+1:STATE_DIM+1+STATE_DIM+1+1] = 1#eoe
    sasreoeEp[-1,STATE_DIM+1+STATE_DIM+1:STATE_DIM+1+STATE_DIM+1+1] = 0#eoe
    if g_msasreoeEps != None:
         concatenate([g_msasreoeEps,sasreoeEp])
    else:
        g_msasreoeEps = sasreoeEp
        

savetxt( "TrainingSASREOE.mat", g_msasreoeEps )
        
