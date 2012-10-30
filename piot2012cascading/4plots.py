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
D_SCIRL = genfromtxt("SCIRL_Exp3.mat")


Expert = 7.74390968 #python Expert.py to get this value
Random_mean = -1.5821833963484#See Highway.org about Random.py for information on these values
Random_min = -4.0007295890199996#python Random.py to get this value
Random_max = 2.7064859345599999#python Random.py to get this value
Random_var = 1.4465398450419833#python Random.py to get this value
ANIRL_mean = 7.2183449001

y_min = -5
y_max = 10
#Figure 1 : No X-axis, random, Expert and PIRL
pylab.figure(1)
pylab.clf()
X = array([0,1])
filled_mean_min_max( pylab, X, Random_mean*ones(X.shape), Random_min*ones(X.shape), Random_max*ones(X.shape), 'cyan',0.2,'--',"Agent trained on a random reward",None)
filled_mean_min_max( pylab, X, Random_mean*ones(X.shape), (Random_mean-Random_var)*ones(X.shape), (Random_mean+Random_var)*ones(X.shape), 'cyan',0.4,'-.',None,None)
pylab.axes().get_xaxis().set_visible(False)
pylab.ylabel('$\mathbf{E}_{s\sim\mathcal{U}}[V^{\pi}_{\mathcal{R}_E}(s)]$')
pylab.plot(X,[ANIRL_mean,ANIRL_mean],color='blue',label="Abbeen \& Ng IRL",lw=2,linestyle='-')
pylab.plot(X,[Expert,Expert],color='purple',label="Expert",lw=2,linestyle='-')
pylab.ylim((-5,10))
pylab.savefig("Fig1-NoX.pdf",transparent=True)


#Figure 2 : classif and us
pylab.figure(2)
pylab.clf()
[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_Classif ))
[X,Y_mean,Y_var] = map( array, mean_var( D_Classif ))
x_min = min(X)
x_max = max(X)
pylab.axis([x_min,x_max,y_min,y_max])
pylab.xlabel('Number of samples from the expert')
pylab.ylabel('$\mathbf{E}_{s\sim\mathcal{U}}[V^{\pi}_{\mathcal{R}_E}(s)]$')
pylab.grid(True)

pylab.plot(X,Expert*ones(X.shape), color='purple',label="Expert",lw=2,ls='-')

[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_Classif ))
[X,Y_mean,Y_var] = map( array, mean_var( D_Classif ))
filled_mean_min_max( pylab, X, Y_mean, Y_min, Y_max, 'green', 0.2,'--',None,None)
filled_mean_min_max( pylab, X, Y_mean, Y_mean - Y_var, Y_mean + Y_var, 'green', 0.4,'-.',None,None)

[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_C ))
[X,Y_mean,Y_var] = map( array, mean_var( D_C ))
filled_mean_min_max( pylab, X, Y_mean, Y_min, Y_max, 'red', 0.2,'--',None,None)
filled_mean_min_max( pylab, X, Y_mean, Y_mean - Y_var, Y_mean + Y_var, 'red', 0.4,'-.',None,None)

pylab.savefig("Fig2-ClassifAndUs.pdf",transparent=True)

#Figure 3 : SCIRL and us

pylab.figure(3)
pylab.clf()
[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_SCIRL ))
[X,Y_mean,Y_var] = map( array, mean_var( D_SCIRL ))
x_min = min(X)
x_max = max(X)
pylab.axis([x_min,x_max,y_min,y_max])
pylab.xlabel('Number of samples from the expert')
pylab.ylabel('$\mathbf{E}_{s\sim\mathcal{U}}[V^{\pi}_{\mathcal{R}_E}(s)]$')
pylab.grid(True)

pylab.plot(X,Expert*ones(X.shape), color='purple',label="Expert",lw=2,ls='-')


filled_mean_min_max( pylab, X, Y_mean, Y_min, Y_max, 'orange', 0.2,'--',None,None)
filled_mean_min_max( pylab, X, Y_mean, Y_mean - Y_var, Y_mean + Y_var, 'orange', 0.4,'-.',None,None)

[X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D_C ))
pylab.plot( X, Y_mean,color='red',lw=3)

pylab.savefig("Fig3-SCIRLAndUs.pdf",transparent=True)


#Figure 4 : The legend
fig = pylab.figure(4)
figlegend = pylab.figure(figsize=(4,2.5))
ax = fig.add_subplot(111)
lines = ax.plot([-1,-1],[-1,-2], color='red',label="Cascading IRL",lw=2,linestyle='-')
lines += ax.plot([-1,-1],[-1,-2], color='green',label="Pure classification",lw=2,linestyle='-')
lines += ax.plot([-1,-1],[-1,-2], color='blue',label="Abbeen \& Ng IRL",lw=2,linestyle='-')
lines += ax.plot([-1,-1],[-1,-2], color='cyan',label="Agent trained on a random reward",lw=2,linestyle='-')
lines += ax.plot([-1,-1],[-1,-2], color='orange',label="SCIRL",lw=2,linestyle='-')
lines += ax.plot([-1,-1],[-1,-2], color='purple',label="Expert",lw=2,linestyle='-')
lines += ax.plot([-1,-1],[-1,-2], color='black',label="min, max",lw=1,linestyle='--')
lines += ax.plot([-1,-1],[-1,-2], color='black',label="Standard deviation",lw=1,linestyle='-.')
figlegend.legend(lines,("Cascading IRL","Pure Classification","Abbeel \& Ng IRL","Agent trained on a random reward","SCIRL","Expert","min, max","Standard deviation"),"center")
figlegend.savefig("Fig4-Legend.pdf")

