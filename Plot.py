from numpy import *
import scipy
import pylab as pylab


def mean_min_max( D ):
    "Returns the abscissa, mean, min and max value for each abscissa in matrix D. D follows format : [[x,y],...,[x,y]]"
    X = unique(D[:,0])
    #Yes, it is unreadable.
    Y_min = map(min,map( lambda d_x: map( lambda d_xi:d_xi[1],d_x),[filter(lambda a:a[0]==x,D) for x in X]))
    Y_max = map(max,map( lambda d_x: map( lambda d_xi:d_xi[1],d_x),[filter(lambda a:a[0]==x,D) for x in X]))
    Y_mean = map(mean,map( lambda d_x: map( lambda d_xi:d_xi[1],d_x),[filter(lambda a:a[0]==x,D) for x in X]))
    return [X,Y_mean,Y_min,Y_max]

def naive_variance(data):
    n = 0
    Sum = 0
    Sum_sqr = 0 
    for x in data:
        n = n + 1
        Sum = Sum + x
        Sum_sqr = Sum_sqr + x*x
    mean = Sum/n
    variance = (Sum_sqr - Sum*mean)/(n - 1)
    return variance


def mean_var( D ):
    "Returns the abscissa, mean, and variance values for each abscissa in matrix D. D follows format : [[x,y],...,[x,y]]"
    X = unique(D[:,0])
    #Yes, it is unreadable
    Y_mean = map(mean,map( lambda d_x: map( lambda d_xi:d_xi[1],d_x),[filter(lambda a:a[0]==x,D) for x in X]))
    Y_var = map(lambda a: sqrt(naive_variance(a)),map( lambda d_x: map( lambda d_xi:d_xi[1],d_x),[filter(lambda a:a[0]==x,D) for x in X]))
    return [X,Y_mean,Y_var]

def filled_mean_min_max( p, X, Y_mean, Y_min, Y_max, color, _alpha, style, lblmain,lblminmax ):
    "Plot data, with bold mean line, and a light color fill betwee the min and max"
    if lblmain == None:
        p.plot( X, Y_mean,color=color,lw=2)
    else:
        p.plot( X, Y_mean,color=color,lw=2, label=lblmain)
    if lblminmax == None:
        p.plot( X, Y_min, color=color,lw=1,linestyle=style)
    else:
        p.plot( X, Y_min, color=color,lw=1,linestyle=style, label=lblminmax)
    p.plot( X, Y_max, color=color,lw=1,linestyle=style)
    p.fill_between(X,Y_min,Y_max,facecolor=color,alpha=_alpha)

class Plot:
    Random_mean = None
    Random_min = None
    Random_var = None
    Random_max = None
    Expert = None
    ymin = None
    ymax = None

    def __init__( self ):
        pass
    def plot( self, D, color, filename ):
        [X,Y_mean,Y_min,Y_max] = map( array, mean_min_max( D ))
        [X,Y_mean,Y_var] = map( array, mean_var( D ))
        pylab.figure(1)
        pylab.clf()
        x_min = min(X)
        x_max = max(X)
        y_max = self.ymax if self.ymax else mint(max( self.Expert, max( Y_max + Y_var) ))+1
        y_min = self.ymin if self.ymin else int(min( self.Random_min,min(Y_min - Y_var) ) - 1)
        pylab.axis([x_min,x_max,y_min,y_max])
        pylab.xlabel('Number of samples from the expert')
        pylab.ylabel('$\mathbf{E}_{s\sim\mathcal{U}}[V^{\pi}_{\mathcal{R}_E}(s)]$')
        pylab.grid(True)
        filled_mean_min_max( pylab, X, Y_mean, Y_min, Y_max, color, 0.2,'--',None,None)
        filled_mean_min_max( pylab, X, Y_mean, Y_mean - Y_var, Y_mean + Y_var, color, 0.4,'-.',None,None)
        filled_mean_min_max( pylab, X, self.Random_mean*ones(X.shape), self.Random_min*ones(X.shape), self.Random_max*ones(X.shape), 'cyan',0.2,'--',"Agent trained on a random reward",None)
        filled_mean_min_max( pylab, X, self.Random_mean*ones(X.shape), (self.Random_mean-self.Random_var)*ones(X.shape), (self.Random_mean+self.Random_var)*ones(X.shape), 'cyan',0.4,'-.',None,None)
        pylab.plot(X,self.Expert*ones(X.shape), color='cyan',label="Expert",lw=2,ls=':')
        pylab.savefig(filename,transparent=True)

