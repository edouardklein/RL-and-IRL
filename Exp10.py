# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Relative Entropy on the Highway
from DP import *
P = genfromtxt("Highway_P.mat")
R = genfromtxt("Highway_R.mat")
Gamma = 0.99

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

sqrt(-log2(1-0.0001)/(2*100))*(pow(0.99,(100+1))-1)/(0.99-1)#Epsilon

# <codecell>

#Relative Entropy
class GradientDescent(object):
   def alpha( self, t ):
      raise NotImplementedError, "Cannot call abstract method"

   theta_0=None
   Threshold=None
   T = -1
   sign = None
        
   def run( self, f_grad, f_proj=None, b_norm=False, b_best=True ): #grad is a function of theta
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

         if current_norm < best_norm or not b_best:
             best_norm = current_norm
             best_theta = theta.copy()
             best_iter = t
         if current_norm < self.Threshold or (self.T != -1 and t >= self.T):
             break

      print "Gradient de norme : "+str(best_norm)+", a l'iteration : "+str(best_iter)
      return best_theta
                            
class RelativeEntropy(GradientDescent):
    sign=+1.
    Threshold=0.01 #Sensible default
    T=50 #Sensible default
    Epsilon = 0.05 #RelEnt parameter, sensible default

    def alpha(self, t):
        return 1./(t+1)#Sensible default
    
    def __init__(self, mu_E, mus):
        self.theta_0 = zeros(mu_E.shape)
        self.Mu_E = mu_E
        self.Mus = mus
    
    def gradient(self, theta):
        numerator = 0
        denominator = 0
        for mu in self.Mus:
            c = exp(dot(theta.transpose(),mu))
            numerator += c*mu
            denominator += c
        assert denominator != 0,"A sum of exp(...) is null, some black magic happened here."
        return self.Mu_E - numerator/denominator - sign(theta)*self.Epsilon
    
    def run(self):
        f_grad = lambda theta: self.gradient(theta)
        theta = super(RelativeEntropy,self).run( f_grad, b_norm=True, b_best=False)
        return theta

# <codecell>

Highway = MDP(P,R)
mPi_E, V_E, Pi_E = Highway.optimal_policy()

# <codecell>

rho = lambda : int(rand()*729) #uniform distribtion over S
D_E = Highway.D_func(Pi_E, 1, 100,  rho)

# <codecell>

Mu_E = zeros((3645,1))
for i in range(0,100):
    s = D_E[i][0]
    a = D_E[i][1]
    Mu_E[s + a*729] += pow(Gamma,i)
Mu_E /= 100.
Mu_E.shape

# <codecell>

random_policy = lambda s: int(rand()*5)
Mus = []
for i in range(0,100):
    D = Highway.D_func(random_policy, 1, 10, rho)
    mu = zeros((3645,1))
    for i in range(0,10):
        s = D[i][0]
        a = D[i][1]
        mu[s + a*729] += pow(Gamma,i)
    mu /= 10.
    Mus.append(mu)
Mus.append(Mu_E)


# <codecell>

RE = RelativeEntropy(Mu_E, Mus)
reward_RE = RE.run()

# <codecell>

reward_RE.shape
Highway2 = MDP(P,reward_RE)
mPi_A, V_A, Pi_A = Highway2.optimal_policy()
true_V_A = linalg.solve( identity( 729 ) - 0.9*dot(mPi_A,P), dot( mPi_A, R) )
mean(true_V_A),mean(V_E)

# <codecell>

Highway3 = MDP(P,rand(3645,1))
mPi_R, V_R, Pi_R = Highway3.optimal_policy()
true_V_R = linalg.solve( identity( 729 ) - 0.9*dot(mPi_R,P), dot( mPi_R, R) )
mean(true_V_A),mean(true_V_R),mean(V_E)

