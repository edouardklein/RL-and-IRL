# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Training an expert on the highway
from DP import *
P = genfromtxt("Highway_P.mat")
R = genfromtxt("Highway_R.mat")

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

Highway = MDP(P,R)
mPi_E, V_E, pi_E = Highway.optimal_policy()

