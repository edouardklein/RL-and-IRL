# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#All IRL algs on the perturbed Highway
from DP import *
from pylab import *
P = genfromtxt("Highway_P.mat")
perturbed_P = zeros(P.shape)
#Let's add, with probability 0.1 a small value to each element of P
for i in range(0,len(P)):
    for j in range(0,len(P[i])):
        perturbed_P[i,j] = P[i,j]
        if rand()<0.1:
            perturbed_P[i,j] += rand()*0.01
    perturbed_P[i] /= sum(perturbed_P[i])
assert all(f_eq(perturbed_P.sum(axis=1), 1)), "A probabaility matrix's lines should sum to 1"
savetxt("Highway_perturbed_P.mat",perturbed_P)

