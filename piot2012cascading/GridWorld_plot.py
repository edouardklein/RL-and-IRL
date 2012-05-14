from matplotlib import rc
rc('text', usetex=True)
import sys
sys.path+=['..']
from numpy import *
import scipy
import pylab as pylab
from Plot import *

D_Cascading = genfromtxt( "Cascading_Exp3.mat" )
D_C = D_Cascading[:,[0,1]]
D_Classif = D_Cascading[:,[0,2]]
D_ANIRL = genfromtxt( "ANIRL_Exp3.mat" )

Expert = 4.11057591 #python Expert.py to get this value
Random_mean = 0.48848324670295395#See GridWorld.org about Random.py for information on these values
Random_min = 0.069469005947400006#python Random.py to get this value
Random_max = 3.2720195801399998#python Random.py to get this value
Random_var = 0.57868038965027513#python Random.py to get this value

y_min = -1
y_max = 5

#Figure 1 : ANIRL, Random and Classif (Concurrents...)
pylab.figure(1)
pylab.clf()
[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_ANIRL ))
[X,Y_mean,Y_var] = map( array, mean_var( D_ANIRL ))
x_min = min(X)
x_max = max(X)
pylab.axis([x_min,x_max,y_min,y_max])
pylab.xlabel('Number of samples from the expert')
pylab.ylabel('${1\over card(S)}\sum\limits_{s\in S}V(s)$')
pylab.grid(True)
filled_mean_min_max( pylab, X, Y_mean, Y_min, Y_max, 'blue', 0.2,'--',None,None)
filled_mean_min_max( pylab, X, Y_mean, Y_mean - Y_var, Y_mean + Y_var, 'blue', 0.4,'-.',None,None)

filled_mean_min_max( pylab, X, Random_mean*ones(X.shape), Random_min*ones(X.shape), Random_max*ones(X.shape), 'cyan',0.2,'--',"Agent trained on a random reward",None)
filled_mean_min_max( pylab, X, Random_mean*ones(X.shape), (Random_mean-Random_var)*ones(X.shape), (Random_mean+Random_var)*ones(X.shape), 'cyan',0.4,'-.',None,None)

pylab.plot(X,Expert*ones(X.shape), color='cyan',label="Expert",lw=2,ls=':')

[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_Classif ))
[X,Y_mean,Y_var] = map( array, mean_var( D_Classif ))
filled_mean_min_max( pylab, X, Y_mean, Y_min, Y_max, 'yellow', 0.2,'--',None,None)
filled_mean_min_max( pylab, X, Y_mean, Y_mean - Y_var, Y_mean + Y_var, 'yellow', 0.4,'-.',None,None)

pylab.savefig("Fig1.pdf",transparent=True)

#Figure 2 : Cascading et moyenne de ANIRL et Classif pure (comparaison aux concurrents).
pylab.figure(2)
pylab.clf()
[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_Classif ))
[X,Y_mean,Y_var] = map( array, mean_var( D_Classif ))
x_min = min(X)
x_max = max(X)
pylab.axis([x_min,x_max,y_min,y_max])
pylab.xlabel('Number of samples from the expert')
pylab.ylabel('${1\over card(S)}\sum\limits_{s\in S}V(s)$')
pylab.grid(True)

[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_ANIRL ))
[X,Y_mean,Y_var] = map( array, mean_var( D_ANIRL ))
pylab.plot( X, Y_mean,color='blue',lw=2)

[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_Classif ))
[X,Y_mean,Y_var] = map( array, mean_var( D_Classif ))
pylab.plot( X, Y_mean,color='yellow',lw=2)

[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_C ))
[X,Y_mean,Y_var] = map( array, mean_var( D_C ))
filled_mean_min_max( pylab, X, Y_mean, Y_min, Y_max, 'orange', 0.2,'--',None,None)
filled_mean_min_max( pylab, X, Y_mean, Y_mean - Y_var, Y_mean + Y_var, 'orange', 0.4,'-.',None,None)

pylab.savefig("Fig2.pdf",transparent=True)
