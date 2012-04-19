
import sys
sys.path+=['..']
import Highway
from DP import *
from DP_mu import *

omega = genfromtxt( sys.argv[1] )
Pi = omega2pi( omega, Highway.phi2, Highway.Sgenerator(), Highway.s_index, [Highway.P( a ) for a in Highway.A ] )
print "Mu computation..."
Mu = DP_mu( Pi, identity( 3*9*9*3 ))
Mu_s_0 = Mu[ Highway.s_index( Highway.S_0() )]

Mu_E = genfromtxt( "Mu_E.mat" )
Mu_E_s_0 = Mu_E[ Highway.s_index( Highway.S_0() )]

perf_agent = dot( Highway.R().transpose() , Mu_s_0 )
perf_expert = dot( Highway.R().transpose() , Mu_E_s_0 )
print "Performance moyenne de l'expert : "
print perf_expert

print "Performance moyenne de l'agent :"
print perf_agent
