# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Mountain Car
from stuff import *
from pylab import *
from random import *
import numpy
from rl import *

ACTION_SPACE=[-1,0,1] #Departing from the convention A = [0..card(A)] may be a bad idea. FIXME change this when I get around to it.

def mountain_car_next_state(state,action):
    position,speed=state
    next_speed = squeeze(speed+action*0.001+cos(3*position)*(-0.0025))
    next_position = squeeze(position+next_speed)
    if not -0.07 <= next_speed <= 0.07:
        next_speed = sign(next_speed)*0.07
    if not -1.2 <= next_position <= 0.6:
        next_speed=0.
        next_position = -1.2 if next_position < -1.2 else 0.6
    return array([next_position,next_speed])

def mountain_car_uniform_state():
    return array([numpy.random.uniform(low=-1.2,high=0.6),numpy.random.uniform(low=-0.07,high=0.07)])
mountain_car_mu_position, mountain_car_mu_speed = meshgrid(linspace(-1.2,0.6,7),linspace(-0.07,0.07,7))

mountain_car_sigma_position = 2*pow((0.6+1.2)/10.,2)
mountain_car_sigma_speed = 2*pow((0.07+0.07)/10.,2)

def mountain_car_single_psi(state):
    position,speed=state
    psi=[]
    for mu in zip_stack(mountain_car_mu_position, mountain_car_mu_speed).reshape(7*7,2):
        psi.append(exp( -pow(position-mu[0],2)/mountain_car_sigma_position 
                        -pow(speed-mu[1],2)/mountain_car_sigma_speed))
    psi.append(1.)
    return array(psi).reshape((7*7+1,1))

mountain_car_psi= non_scalar_vectorize(mountain_car_single_psi,(2,),(50,1))

def mountain_car_single_phi(sa):
    state=sa[:2]
    index_action = int(sa[-1])+1
    answer=zeros(((7*7+1)*3,1))
    answer[index_action*(7*7+1):index_action*(7*7+1)+7*7+1] = mountain_car_single_psi(state)
    return answer

mountain_car_phi= non_scalar_vectorize(mountain_car_single_phi,(3,),(150,1))

def mountain_car_reward(sas):
    position=sas[0]
    return 1 if position > 0.5 else 0

def mountain_car_episode_length(initial_position,initial_speed,policy):
    answer = 0
    reward = 0.
    state = array([initial_position,initial_speed])
    while answer < 300 and reward == 0. :
        action = policy(state)
        next_state = mountain_car_next_state(state,action)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        state=next_state
        answer+=1
    return answer

def mountain_car_episode_vlength(policy):
    return vectorize(lambda p,s:mountain_car_episode_length(p,s,policy))


def mountain_car_training_data(freward=mountain_car_reward,traj_length=5,nb_traj=1000):
    traj = []
    random_policy = lambda s:choice(ACTION_SPACE)
    for i in range(0,nb_traj):
        state = mountain_car_uniform_state()
        reward=0
        t=0
        while t < traj_length and reward == 0:
            t+=1
            action = random_policy(state)
            next_state = mountain_car_next_state(state, action)
            reward = freward(hstack([state, action, next_state]))
            traj.append(hstack([state, action, next_state, reward]))
            state=next_state
    return array(traj)
def mountain_car_plot( f, draw_contour=True, contour_levels=50, draw_surface=False ):
    '''Display a surface plot of function f over the state space'''
    pos = linspace(-1.2,0.6,30)
    speed = linspace(-0.07,0.07,30)
    pos,speed = meshgrid(pos,speed)
    Z = f(pos,speed)
    #fig = figure()
    if draw_surface:
        ax=Axes3D(fig)
        ax.plot_surface(pos,speed,Z)
    if draw_contour:
        contourf(pos,speed,Z,levels=linspace(min(Z.reshape(-1)),max(Z.reshape(-1)),contour_levels+1))
        colorbar()
def mountain_car_plot_policy( policy ):
    two_args_pol = lambda p,s:squeeze(policy(zip_stack(p,s)))
    mountain_car_plot(two_args_pol,contour_levels=3)

def mountain_car_V(omega):
    policy = greedy_policy( omega, mountain_car_phi, ACTION_SPACE )
    def V(pos,speed):
        actions = policy(zip_stack(pos,speed))
        Phi=mountain_car_phi(zip_stack(pos,speed,actions))
        return squeeze(dot(omega.transpose(),Phi))
    return V

def mountain_car_interesting_state():
    position = numpy.random.uniform(low=-1.2,high=-0.9)
    speed = numpy.random.uniform(low=-0.07,high=0)
    return array([position,speed])

def mountain_car_IRL_traj():
    traj = []
    state = mountain_car_interesting_state()
    reward = 0
    while reward == 0:
        action = mountain_car_manual_policy(state)
        next_state = mountain_car_next_state(state, action)
        next_action = mountain_car_manual_policy(next_state)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        traj.append(hstack([state, action, next_state, next_action, reward]))
        state=next_state
    return array(traj)

def mountain_car_IRL_data(nbsamples):
    data = mountain_car_IRL_traj()
    while len(data) < nbsamples:
        data = vstack([data,mountain_car_IRL_traj()])
    return data[:nbsamples]

def mountain_car_manual_policy(state):
    position,speed = state
    return -1. if speed <=0 else 1.

TRAJS = mountain_car_IRL_data(1000)
scatter(TRAJS[:,0],TRAJS[:,1],c=TRAJS[:,2])
axis([-1.2,0.6,-0.07,0.07])
phi=mountain_car_phi

# <codecell>

def mountain_car_RE_traj():
    traj = []
    state = mountain_car_interesting_state()
    reward = 0
    t=0
    while reward == 0 and t<60:
        t+=1
        action = choice(ACTION_SPACE)
        next_state = mountain_car_next_state(state, action)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        traj.append(hstack([state, action, next_state, reward]))
        state=next_state
    return array(traj)

data_r = vstack([mountain_car_RE_traj() for i in range(0,400)])
savetxt("mountain_car_RE_trajs.mat",data_r)

# <codecell>

sqrt(-log2(1-0.0001)/(2*70))*(pow(0.99,(70+1))-1)/(0.99-1)#Epsilon

# <codecell>

GAMMA = 0.99

def end_of_episode(data,i):
    try:
        if all(data[i,3:5] == data[i+1,:2]):
            return False
        else:
            return True
    except:
        return True

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

    
data_r_LSPI = genfromtxt("mountain_car_batch_data.mat") #See Exp4 : rho uniform, M=1000, L=5
data_r_RE = genfromtxt("mountain_car_RE_trajs.mat") #rho  interesting, M=400, L<=60
data_r_Other = mountain_car_training_data( nb_traj=50, traj_length=100) #rho uniform, M=50,L=100

#Computing the feature expectations
t=0.
Mu_E = zeros(((7*7+1)*3,1))
for i in range(0,len(TRAJS)):
    Mu_E += pow(GAMMA,t)*mountain_car_phi(TRAJS[i,:3])
    if end_of_episode(TRAJS,i):
        t=0.
    else:
        t+=1.
Mu_E /= float(len(TRAJS))

Mus_LSPI=[]
mu = zeros(((7*7+1)*3,1))
t=0.
for i in range(0,len(data_r_LSPI)):
    mu += pow(GAMMA,t)*mountain_car_phi(data_r_LSPI[i,:3])
    if end_of_episode(data_r_LSPI,i):
        mu /= t+1.
        Mus_LSPI.append(mu)
        t=0.
        mu = zeros(((7*7+1)*3,1))
    else:
        t += 1. 
Mus_LSPI.append(Mu_E)

Mus_RE=[]
mu = zeros(((7*7+1)*3,1))
t=0.
for i in range(0,len(data_r_RE)):
    mu += pow(GAMMA,t)*mountain_car_phi(data_r_RE[i,:3])
    if end_of_episode(data_r_RE,i):
        mu /= t+1.
        Mus_RE.append(mu)
        t=0.
        mu = zeros(((7*7+1)*3,1))
    else:
        t += 1. 
Mus_RE.append(Mu_E)

Mus_Other=[]
mu = zeros(((7*7+1)*3,1))
t=0.
for i in range(0,len(data_r_Other)):
    mu += pow(GAMMA,t)*mountain_car_phi(data_r_Other[i,:3])
    if end_of_episode(data_r_Other,i):
        mu /= t+1.
        Mus_Other.append(mu)
        t=0.
        mu = zeros(((7*7+1)*3,1))
    else:
        t += 1. 
Mus_Other.append(Mu_E)

RE_LSPI = RelativeEntropy(Mu_E, Mus_LSPI)
theta_RE_LSPI = RE_LSPI.run()
     
RE_RE = RelativeEntropy(Mu_E, Mus_RE)
theta_RE_RE = RE_RE.run()

RE_Other = RelativeEntropy(Mu_E, Mus_Other)
theta_RE_Other = RE_Other.run()

# <codecell>

def RE_reward_LSPI(sas):
    sa = sas[:3]
    return squeeze(dot(theta_RE_LSPI.transpose(),mountain_car_phi(sa)))
vRE_reward = non_scalar_vectorize( RE_reward_LSPI, (5,),(1,1) )
data = genfromtxt("mountain_car_batch_data.mat")
data[:,5] = squeeze(vRE_reward(data[:,:5]))
policy_RE_LSPI,omega_RE = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=150, iterations_max=20 )

def RE_reward_RE(sas):
    sa = sas[:3]
    return squeeze(dot(theta_RE_RE.transpose(),mountain_car_phi(sa)))
vRE_reward = non_scalar_vectorize( RE_reward_RE, (5,),(1,1) )
data = genfromtxt("mountain_car_batch_data.mat")
data[:,5] = squeeze(vRE_reward(data[:,:5]))
policy_RE_RE,omega_RE = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=150, iterations_max=20 )

def RE_reward_Other(sas):
    sa = sas[:3]
    return squeeze(dot(theta_RE_Other.transpose(),mountain_car_phi(sa)))
vRE_reward = non_scalar_vectorize( RE_reward_Other, (5,),(1,1) )
data = genfromtxt("mountain_car_batch_data.mat")
data[:,5] = squeeze(vRE_reward(data[:,:5]))
policy_RE_Other,omega_RE = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=150, iterations_max=20 )

mountain_car_plot_policy(policy_RE_LSPI)
figure()
mountain_car_plot_policy(policy_RE_RE)
figure()
mountain_car_plot_policy(policy_RE_Other)

# <codecell>

plottable_episode_length = mountain_car_episode_vlength(policy_RE)
X = linspace(-1.2,0.6,20)
Y = linspace(-0.07,0.07,20)
X,Y = meshgrid(X,Y)
Z2 = plottable_episode_length(X,Y)
figure()
mountain_car_plot_policy(policy_RE)
figure()
contourf(X,Y,Z2,50)
colorbar()

# <codecell>

def mountain_car_testing_state():
    position = numpy.random.uniform(low=-1.2,high=0.5)
    speed = numpy.random.uniform(low=-0.07,high=0.07)
    return array([position,speed])

def mountain_car_mean_performance(policy):
    return mean([mountain_car_episode_length(state[0],state[1],policy) for state in [mountain_car_testing_state() for i in range(0,100)]])

mountain_car_plot_policy(policy_RE)
print mountain_car_mean_performance(policy_RE)

