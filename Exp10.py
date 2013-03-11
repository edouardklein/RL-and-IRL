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

#Relative Entropy
class GradientDescent(object):
   def alpha( self, t ):
      raise NotImplementedError, "Cannot call abstract method"

   theta_0=None
   Threshold=None
   T = -1
   sign = None
        
   def run( self, f_grad, f_proj=None, b_norm=False ): #grad is a function of theta
      theta = self.theta_0.copy()
      best_theta = theta.copy()
      best_norm = float("inf")
      best_iter = 0
      t=0
      while True:#Do...while loop
         t+=1
         DeltaTheta = f_grad( theta )
         current_norm = norm( DeltaTheta )
         if b_norm and  current_norm > 0.:
             DeltaTheta /= norm( DeltaTheta )
         theta = theta + self.sign * self.alpha( t )*DeltaTheta
         if f_proj:
             theta = f_proj( theta )
         print "Norme du gradient : "+str(current_norm)+", pas : "+str(self.alpha(t))+", iteration : "+str(t)

         if current_norm < best_norm:
             best_norm = current_norm
             best_theta = theta.copy()
             best_iter = t
         if current_norm < self.Threshold or (self.T != -1 and t >= self.T):
             break

      print "Gradient de norme : "+str(best_norm)+", a l'iteration : "+str(best_iter)
      return best_theta
                            
class RelativeEntropy(GradientDescent):
    sign=-1.
    Threshold=0.1 #Sensible default
    T=100 #Sensible default
    Epsilon = 0.01 #RelEnt parameter, sensible default
    
    def alpha(self, t):
        return 3./(t+1)#Sensible default
    
    def __init__(self, mu_E, mus):
        self.Mu_E = mu_E
        self.Mus = mus
    
    def gradient(self, theta):
        numerator = 0
        denominator = 0
        for mu in self.Mus:
            c = exp(dot(theta,mu))
            numerator += c*mu
            denominator += c
        assert denominator != 0,"A sum of exp(...) is null, some black magic happened here."
        return self.Mu_E - numerator/denominator - sign(theta)*self.Epsilon
    
    def run(self):
        f_grad = lambda theta: self.gradient(theta)
        theta = super(StructuredClassifier,self).run( f_grad, b_norm=True)
        return theta

# <codecell>

Highway = MDP(P,R)
mPi_E, V_E, Pi_E = Highway.optimal_policy()

# <codecell>

rho = lambda : int(rand()*729) #uniform distribtion over S
D_E = Highway.D_func(Pi_E, 50, 100,  rho)

# <codecell>

Mu_E = zeros((3645,1))
for sasr in D_E:
    s = sasr[0]
    Mu_E[s] += 1.
Mu_E /= float(len(D_E))

# <codecell>

random_policy = lambda s: int(rand()*5)
random_trajs = []
for i in range(0,50):
    D = Highway.D_func(random_policy, 1, 100, rho)
    random_trajs.append(D)
Mus = []
for D in random_trajs:
    mu = zeros((3645,1))
    for sasr in D:
        s = sasr[0]
        mu[s] += 1.
    mu /= float(len(D))
    Mus.append(mu)

