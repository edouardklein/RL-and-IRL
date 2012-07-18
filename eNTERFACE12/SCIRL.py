#!/usr/bin/python

import sys
sys.path+=['..']
from a2str import *
from LAFEM import *
from DP_mu import *
STATE_DIM = 8
PSI_DIM = 338
g_fUpdateScale = 0.1
g_lRelativePosCenters = [] #Centers of the gaussians for the first 4 components 
g_fPosSigma = g_fUpdateScale
g_lClassifierCenters = [] #Centers of the gaussians for the following 4 components
g_fClassifierSigma = 0.3
#First four components
relativePosCoords = [ - g_fUpdateScale, 0., g_fUpdateScale]
for x1 in relativePosCoords:
    for y1 in relativePosCoords:
        for x2 in relativePosCoords:
            for y2 in relativePosCoords:
                g_lRelativePosCenters.append(array([x1,y1,x2,y2]))
classifierCoords = [0., 1./3.,2./3., 1.]
for x1 in classifierCoords:
    for y1 in classifierCoords:
        for x2 in classifierCoords:
            for y2 in classifierCoords:
                g_lClassifierCenters.append(array([x1,y1,x2,y2]))
    
def psi( s ): #Gaussian network
    answer = zeros( [PSI_DIM, 1] )
    i = 0
    x = s[0:4]
    for center in g_lRelativePosCenters:
        toSum = map( lambda a : a*a/(2*g_fPosSigma*g_fPosSigma), (x - center) )
        answer[i] = exp( - sum( toSum ) )
        i += 1
    for center in g_lClassifierCenters:
        toSum = map( lambda a : a*a/(2*g_fClassifierSigma*g_fClassifierSigma), (x - center) )
        answer[i] = exp( - sum( toSum ) )
        i += 1
    answer[i] = 1.
    return answer

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
