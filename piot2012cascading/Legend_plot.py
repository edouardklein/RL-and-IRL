from matplotlib import rc
rc('text', usetex=True)
import sys
sys.path+=['..']
from numpy import *
import scipy
import pylab as pylab
from Plot import *

#The legend
fig = pylab.figure(4)
figlegend = pylab.figure(figsize=(4,2.5))
ax = fig.add_subplot(111)
lines = ax.plot([-1,-1],[-1,-2], color='orange',label="Cascading IRL",lw=2,linestyle='-')
lines += ax.plot([-1,-1],[-1,-2], color='green',label="Pure classification",lw=2,linestyle='-')
lines += ax.plot([-1,-1],[-1,-2], color='blue',label="Abbeen \& Ng IRL",lw=2,linestyle='-')
lines += ax.plot([-1,-1],[-1,-2], color='cyan',label="Agent trained on a random reward",lw=2,linestyle='-')
lines += ax.plot([-1,-1],[-1,-2], color='black',label="min, max",lw=1,linestyle='--')
lines += ax.plot([-1,-1],[-1,-2], color='black',label="Standard deviation",lw=1,linestyle='-.')
lines += ax.plot([-1,-1],[-1,-2], color='cyan',label="Expert",lw=2,linestyle=':')
figlegend.legend(lines,("Cascading IRL","Pure Classification","Abbeel \& Ng IRL","Agent trained on a random reward","min, max","Standard deviation","Expert"),"center")
figlegend.savefig("Legend.pdf")
