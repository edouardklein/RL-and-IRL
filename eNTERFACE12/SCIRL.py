#!/usr/bin/python

import sys
sys.path+=['..']
from a2str import *
from LAFEM import *
from DP_mu import *
from phipsi import *

class SCIRL( LAFEM ):
    dicPi_E = {}
    A = [0,1,2,3]
    dicMu_E_s_only = {}

    def __init__( self ):
        D_E = genfromtxt(sys.argv[1])
        for trans in D_E:
            self.dicPi_E[l2str(trans[0:STATE_DIM])] = trans[STATE_DIM:STATE_DIM+1][0]
            self.data = self.data +[[ trans[0:STATE_DIM], trans[STATE_DIM:STATE_DIM+1][0]]]

        dicMu_E_data = {}
        #FIXME maybe duplicated feature (cutting into episodes) in ../Cascading.org
        for start_index in range(0,len(D_E)):
            s = D_E[ start_index, 0:STATE_DIM ]
            end_index = (i for i in range(start_index,len(D_E)) if D_E[i,2*STATE_DIM+1+1] == 0).next() #till next eoe
            data_MC = D_E[start_index:end_index+1,0:STATE_DIM]
            try:
                dicMu_E_data[l2str(s)].append(data_MC)
            except KeyError:
                dicMu_E_data[l2str(s)] = [data_MC]
        #Now dicMu_E_data contains the data that allows for the Monte-Carlo computation
        self.dicMu_E_s_only = {}
        for state in dicMu_E_data:
            gamma = 0.9
            lstMu_s = map( lambda traj: monte_carlo_mu( traj, 0.9, psi ), dicMu_E_data[state] )
            mu_s = mean( lstMu_s, 0 )
            self.dicMu_E_s_only[state] = mu_s
        

    def mu_E( self, s, a ):
        gamma = 0.9
        if self.dicPi_E[l2str(s)] == a:
            return self.dicMu_E_s_only[l2str(s)]
        else:
            return gamma*self.dicMu_E_s_only[l2str(s)]

    def l( self, s, a ):
        if self.dicPi_E[l2str(s)] == a:
            return 0
        else:
            return 1

    def alpha( self, t ):
        return 3./(t+1.)
    theta_0 = zeros( (PSI_DIM, 1) )
    Threshold = 0.038
    T = 400
scirl = SCIRL()
theta_scirl = scirl.run()
savetxt( "theta_scirl.mat", theta_scirl )
