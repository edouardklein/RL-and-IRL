from matplotlib import rc
rc('text', usetex=True)
import sys
sys.path+=['..']
from numpy import *
import scipy
import pylab as pylab
from Plot import *

D_Cascading = genfromtxt( "Cascading_Exp5.mat" )
D_C = D_Cascading[:,[0,1]]
D_Classif = D_Cascading[:,[0,2]]

Expert = 7.74390968 #python Expert.py to get this value
Random_mean = -1.5821833963484#See Highway.org about Random.py for information on these values
Random_min = -4.0007295890199996#python Random.py to get this value
Random_max = 2.7064859345599999#python Random.py to get this value
Random_var = 1.4465398450419833#python Random.py to get this value
ANIRL_mean = 7.2183449001

y_min = -5
y_max = 10

#Figure 1 : ANIRL, Random, classif and us
pylab.figure(1)
pylab.clf()
[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_Classif ))
[X,Y_mean,Y_var] = map( array, mean_var( D_Classif ))
x_min = min(X)
x_max = max(X)
pylab.axis([x_min,x_max,y_min,y_max])
pylab.xlabel('Number of samples from the expert')
pylab.ylabel('$\mathbf{E}_{s\sim\mathcal{U}}[V^{\pi}_{\mathcal{R}_E}(s)]$')
pylab.grid(True)

filled_mean_min_max( pylab, X, Random_mean*ones(X.shape), Random_min*ones(X.shape), Random_max*ones(X.shape), 'cyan',0.2,'--',"Agent trained on a random reward",None)
filled_mean_min_max( pylab, X, Random_mean*ones(X.shape), (Random_mean-Random_var)*ones(X.shape), (Random_mean+Random_var)*ones(X.shape), 'cyan',0.4,'-.',None,None)

pylab.plot(X,Expert*ones(X.shape), color='red',label="Expert",lw=2,ls='-')

[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_Classif ))
[X,Y_mean,Y_var] = map( array, mean_var( D_Classif ))
filled_mean_min_max( pylab, X, Y_mean, Y_min, Y_max, 'green', 0.2,'--',None,None)
filled_mean_min_max( pylab, X, Y_mean, Y_mean - Y_var, Y_mean + Y_var, 'green', 0.4,'-.',None,None)

pylab.plot( X, ANIRL_mean*ones(X.shape),color='blue',lw=2)


[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_C ))
[X,Y_mean,Y_var] = map( array, mean_var( D_C ))
filled_mean_min_max( pylab, X, Y_mean, Y_min, Y_max, 'orange', 0.2,'--',None,None)
filled_mean_min_max( pylab, X, Y_mean, Y_mean - Y_var, Y_mean + Y_var, 'orange', 0.4,'-.',None,None)

pylab.savefig("Fig3.pdf",transparent=True)

