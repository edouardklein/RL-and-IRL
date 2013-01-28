# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Misc.

# <headingcell level=3>

# Pythonesque stuff

# <codecell>

#!/usr/bin/env python
from random import choice
from pylab import *
from mpl_toolkits.mplot3d import axes3d, Axes3D

def non_scalar_vectorize(func, input_shape, output_shape):
    """Return a featurized version of func, where func takes a potentially matricial argument and returns a potentially matricial answer.

    These functions can not be naively vectorized by numpy's vectorize.
 
    With vfunc = non_scalar_vectorize( func, (2,), (10,1) ),
    
    func([p,s]) will return a 2D matrix of shape (10,1).

    func([[p1,s1],...,[pn,sn]]) will return a 3D matrix of shape (n,10,1).

    And so on.
    """
    def vectorized_func(arg):
        #print 'Vectorized : arg = '+str(arg)
        nbinputs = prod(arg.shape)/prod(input_shape)
        if nbinputs == 1:
            return func(arg)
        outer_shape = arg.shape[:len(arg.shape)-len(input_shape)]
        outer_shape = outer_shape if outer_shape else (1,)
        arg = arg.reshape((nbinputs,)+input_shape)
        answers=[]
        for input_matrix in arg:
            answers.append(func(input_matrix))
        return array(answers).reshape(outer_shape+output_shape)
    return vectorized_func

def zip_stack(*args):
    """Given matrices of same shape, return a matrix whose elements are tuples from the arguments (i.e. with one more dimension).

    zip_stacking three matrices of shape (n,p) will yeld a matrix of shape (n,p,3)
    """
    shape = args[0].shape
    nargs = len(args)
    args = [m.reshape(-1) for m in args]
    return array(zip(*args)).reshape(shape+(nargs,))
#zip_stack(array([[1,2,3],[4,5,6]]),rand(2,3))

# <headingcell level=3>

# General Mathematics code

# <codecell>

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
         if norm < self.Threshold or (self.T != -1 and t >= self.T):
             break

      print "Gradient de norme : "+str(best_norm)+", a l'iteration : "+str(best_iter)
      return best_theta

# <codecell>

class StructuredClassifier(GradientDescent):
    sign=-1.
    Threshold=0.1 #Sensible default
    T=40 #Sensible default
    phi=None
    phi_xy=None
    inputs=None
    labels=None
    label_set=None
    dic_data={}
    x_dim=None
    
    def alpha(self, t):
        return 3./(t+1)#Sensible default
    
    def __init__(self, data, x_dim, phi, phi_dim, Y):
        self.x_dim=x_dim
        self.inputs = data[:,:-1]
        shape = list(data.shape)
        shape[-1] = 1
        self.labels = data[:,-1].reshape(shape)
        self.phi=phi
        self.label_set = Y
        self.theta_0 = zeros((phi_dim,1))
        self.phi_xy = self.phi(data)
        for x,y in zip(self.inputs,self.labels):
            self.dic_data[str(x)] = y
        print self.inputs.shape
    
    def structure(self, xy):
        return 0. if xy[-1] == self.dic_data[str(xy[:-1])] else 1.
        
    def structured_decision(self, theta):
        def decision( x ):
            score = lambda xy: dot(theta.transpose(),self.phi(xy)) + self.structure(xy)
            input_label_couples = [hstack([x,y]) for y in self.label_set]
            best_label = argmax(input_label_couples, score)[-1]
            return best_label
        vdecision = non_scalar_vectorize(decision, (self.x_dim,), (1,1))
        return lambda x: vdecision(x).reshape(x.shape[:-1]+(1,))
    
    def gradient(self, theta):
        classif_rule = self.structured_decision(theta)
        y_star = classif_rule(self.inputs)
        #print "Gradient : "+str(y_star)
        #print str(self.labels)
        phi_star = self.phi(hstack([self.inputs,y_star]))
        return mean(phi_star-self.phi_xy,axis=0)
    
    def run(self):
        f_grad = lambda theta: self.gradient(theta)
        theta = super(StructuredClassifier,self).run( f_grad, b_norm=True)
        classif_rule = greedy_policy(theta,self.phi,self.label_set)
        return classif_rule,theta
        

# <codecell>

def least_squares_regressor(data, psi, x_dim, _lambda=0.1):
    """Return a function that given a x, will output a y based on a least square regression of the data provided in argument"""
    X = data[:,0:x_dim]
    Y = data[:,x_dim:]
    psi_X = squeeze(psi(X))
    #print psi_X
    omega = dot(dot(inv(dot(psi_X.transpose(),psi_X)+_lambda*identity(psi_X.shape[1])),psi_X.transpose()),Y)
    #omega = dot(inv(dot(psi_X.transpose(),psi_X)+_lambda*identity(psi_X.shape[1])),psi_X.transpose())
    #omega = inv(dot(psi_X.transpose(),psi_X)+_lambda*identity(psi_X.shape[1]))
    #omega = dot(psi_X.transpose(),psi_X)+_lambda*identity(psi_X.shape[1])
    #print psi_X.shape
    #omega = dot(psi_X.transpose(),psi_X)
    regression = lambda x:dot(omega.transpose(),psi(x))
    return regression
                

# <headingcell level=2>

# Inverted Pendulum-specific code

# <codecell>

RANDOM_RUN_LENGTH=5000
EXPERT_RUN_LENGTH=3000
TRANS_WIDTH=6
ACTION_SPACE=[0,1,2]
GAMMA=0.9 #Discout factor
LAMBDA=0.1 #Regularization coeff for LSTDQ

# <codecell>

def inverted_pendulum_single_psi( state ):
    position,speed=state
    answer = zeros((10,1))
    index = 0
    answer[index] = 1.
    index+=1
    for i in linspace(-pi/4,pi/4,3):
        for j in linspace(-1,1,3):
            answer[index] = exp(-(pow(position-i,2) +
                                  pow(speed-j,2))/2.)
            index+=1
    #print "psi stops ar index "+str(index)
    return answer

inverted_pendulum_psi = non_scalar_vectorize( inverted_pendulum_single_psi,(2,), (10,1) )

#[inverted_pendulum_psi(rand(2)).shape,
# inverted_pendulum_psi(rand(2,2)).shape,
# inverted_pendulum_psi(rand(3,5,2)).shape]

# <codecell>

def inverted_pendulum_single_phi(state_action):
    position, speed, action = state_action
    answer = zeros((30,1))
    index = action*10
    answer[ index:index+10 ] = inverted_pendulum_single_psi( [position, speed] )
    return answer

inverted_pendulum_phi = non_scalar_vectorize(inverted_pendulum_single_phi, (3,), (30,1))

#[inverted_pendulum_phi(rand(3)).shape,
# inverted_pendulum_phi(rand(3,3)).shape,
# inverted_pendulum_phi(rand(4,5,3)).shape]

# <codecell>

def inverted_pendulum_V(omega):
    policy = greedy_policy( omega, inverted_pendulum_phi, ACTION_SPACE )
    def V(pos,speed):
        actions = policy(zip_stack(pos,speed))
        Phi=inverted_pendulum_phi(zip_stack(pos,speed,actions))
        return squeeze(dot(omega.transpose(),Phi))
    return V

# <codecell>

def inverted_pendulum_next_state(state, action):
    position,speed = state
    noise = rand()*20.-10.
    control = None
    if action == 0:
        control = -50 + noise;
    elif action == 1:
        control = 0 + noise;
    else: #action==2
        control = 50 + noise;
    g = 9.8;
    m = 2.0;
    M = 8.0;
    l = 0.50;
    alpha = 1./(m+M);
    step = 0.1;
    acceleration = (g*sin(position) - 
                    alpha*m*l*pow(speed,2)*sin(2*position)/2. - 
                    alpha*cos(position)*control) / (4.*l/3. - alpha*m*l*pow(cos(position),2))
    next_position = position +speed*step;
    next_speed = speed + acceleration*step;
    return array([next_position,next_speed])

def inverted_pendulum_reward( sas ):
    position,speed = sas[-2:]
    #print "position is "+str(position)
    if abs(position)>pi/2.:
    #    print "-1"
        return -1.
    #print "0"
    return 0.

def inverted_pendulum_uniform_initial_state():
    return (rand(2)*2.-1.)*pi/2.

def inverted_pendulum_optimal_initial_state():
    return rand(2)*0.2-0.1

def inverted_pendulum_trace( policy,run_length=RANDOM_RUN_LENGTH,
                             initial_state=inverted_pendulum_optimal_initial_state,
                             reward = inverted_pendulum_reward):
    data = zeros((run_length, TRANS_WIDTH))
    state = initial_state()
    for i,void in enumerate( data ):
        action = policy( state )
        new_state = inverted_pendulum_next_state( state, action )
        r = reward( hstack([state,action,new_state]) )
        data[i,:] = hstack([state,action,new_state,[r]])
        if r == 0.:
            state = new_state
        else: #Pendulum has fallen
            state = initial_state()
    return data

def inverted_pendulum_random_trace(reward=inverted_pendulum_reward):
    pi = lambda s: choice(ACTION_SPACE)
    return inverted_pendulum_trace( pi,reward=reward )

def inverted_pendulum_expert_trace( reward ):
    data = inverted_pendulum_random_trace(reward=reward)
    policy,omega = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=inverted_pendulum_phi, phi_dim=30 )
    inverted_pendulum_plot(inverted_pendulum_V(omega))
    two_args_pol = lambda p,s:squeeze(policy(zip_stack(p,s)))
    inverted_pendulum_plot(two_args_pol,contour_levels=3)
    return inverted_pendulum_trace( policy,run_length=EXPERT_RUN_LENGTH ),policy,omega

# <codecell>

def inverted_pendulum_plot( f, draw_contour=True, contour_levels=50, draw_surface=False ):
    '''Display a surface plot of function f over the state space'''
    pos = linspace(-pi,pi,30)
    speed = linspace(-pi,pi,30)
    pos,speed = meshgrid(pos,speed)
    Z = f(pos,speed)
    fig = figure()
    if draw_surface:
        ax=Axes3D(fig)
        ax.plot_surface(pos,speed,Z)
    if draw_contour:
        contourf(pos,speed,Z,levels=linspace(min(Z.reshape(-1)),max(Z.reshape(-1)),contour_levels+1))
        colorbar()
    #show()
def inverted_pendulum_plot_policy( policy ):
    two_args_pol = lambda p,s:squeeze(policy(zip_stack(p,s)))
    inverted_pendulum_plot(two_args_pol,contour_levels=3)
def inverted_pendulum_plot_SAReward(reward,policy):
    X = linspace(-pi,pi,30)
    Y = X
    X,Y = meshgrid(X,Y)
    XY = zip_stack(X,Y)
    XYA = zip_stack(X,Y,squeeze(policy(XY)))
    Z = squeeze(reward(XYA))
    contourf(X,Y,Z,levels=linspace(min(Z.reshape(-1)),max(Z.reshape(-1)),51))
    colorbar()
                               
#test_omega=zeros((30,1)
#test_omega[5]=1.
#inverted_pendulum_plot(inverted_pendulum_V(test_omega))
#pol = greedy_policy(test_omega,inverted_pendulum_phi,ACTION_SPACE)
#inverted_pendulum_plot_policy(policy_good)

# <headingcell level=2>

# Reinforcement Learning Code

# <codecell>

def argmax( set, func ):
     return max( zip( set, map(func,set) ), key=lambda x:x[1] )[0]

def greedy_policy( omega, phi, A ): 
    def policy( *args ):
        state_actions = [hstack(args+(a,)) for a in A]
        q_value = lambda sa: float(dot(omega.transpose(),phi(sa)))
        best_action = argmax( state_actions, q_value )[-1] #FIXME6: does not work for multi dimensional actions
        return best_action
    vpolicy = non_scalar_vectorize( policy, (2,), (1,1) )
    return lambda state: vpolicy(state).reshape(state.shape[:-1]+(1,))

#test_omega=zeros((30,1))
#test_omega[1]=1.
#pol = greedy_policy( test_omega, inverted_pendulum_phi, ACTION_SPACE )
#[pol(rand(2)),pol(rand(3,2)).shape,pol(rand(3,3,2)).shape]

# <codecell>

def lstdq(phi_sa, phi_sa_dash, rewards, phi_dim=1):
    #print "shapes of phi de sa, phi de sprim a prim, rewards"+str(phi_sa.shape)+str(phi_sa_dash.shape)+str(rewards.shape)
    A = zeros((phi_dim,phi_dim))
    b = zeros((phi_dim,1))
    for phi_t,phi_t_dash,reward in zip(phi_sa,phi_sa_dash,rewards):
        A = A + dot( phi_t,
                     (phi_t - GAMMA*phi_t_dash).transpose())
        b = b + phi_t*reward
    return dot(inv(A + LAMBDA*identity( phi_dim )),b)

def lspi( data, s_dim=1, a_dim=1, A = [0], phi=None, phi_dim=1, epsilon=0.01, iterations_max=30,
          plot_func=None):
    nb_iterations=0
    sa = data[:,0:s_dim+a_dim]
    phi_sa = phi(sa)
    s_dash = data[:,s_dim+a_dim:s_dim+a_dim+s_dim]
    rewards = data[:,s_dim+a_dim+s_dim]
    omega = zeros(( phi_dim, 1 ))
    #omega = genfromtxt("../Code/InvertedPendulum/omega_E.mat").reshape(30,1)
    diff = float("inf")
    cont = True
    policy = greedy_policy( omega, phi, A )
    while cont:
        if plot_func:
            plot_func(omega)
        sa_dash = hstack([s_dash,policy(s_dash)])
        phi_sa_dash = phi(sa_dash)
        omega_dash = lstdq(phi_sa, phi_sa_dash, rewards, phi_dim=phi_dim)
        diff = norm( omega_dash-omega )
        omega = omega_dash
        policy = greedy_policy( omega, phi, A )
        nb_iterations+=1
        print "LSPI, iter :"+str(nb_iterations)+", diff : "+str(diff)
        if nb_iterations > iterations_max or diff < epsilon:
            cont = False
    return policy,omega

# <headingcell level=2>

# Inverse Reinforcement Learning code

# <codecell>

def CSI(data, classifier, regressor, A, s_dim=1, gamma=0.9):#FIXME7: does not work for vectorial actions
    column_shape = (len(data),1)
    s = data[:,0:s_dim]
    a = data[:,s_dim:s_dim+1].reshape(column_shape)
    sa = data[:,0:s_dim+1]
    s_dash = data[:,s_dim+1:s_dim+1+s_dim]
    pi_c,q = classifier(hstack([s,a]))
    a_dash = pi_c(s_dash).reshape(column_shape)
    sa_dash = hstack([s_dash,a_dash])
    hat_r = (q(sa)-gamma*q(sa_dash)).reshape(column_shape)
    print "Shapes : s, a, s_dash, a_dash, sa, sa_dash, hat_r"+str([x.shape for x in [s,a,s_dash,a_dash,sa, sa_dash,hat_r]])
    r_min = min(hat_r)-ones(column_shape)
    regression_input_matrices = [hstack([s,action*ones(column_shape)]) for action in A] 
    def add_output_column( reg_mat ):
        actions = reg_mat[:,-1].reshape(column_shape)
        hat_r_bool_table = array(actions==a)
        r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
        output_column = hat_r_bool_table*hat_r+r_min_bool_table*r_min
        return hstack([reg_mat,output_column])
    regression_matrix = vstack(map(add_output_column,regression_input_matrices))
    return regressor( regression_matrix )
    

# <codecell>

def get_structured_classifier_for_SCI(s_dim, phi, phi_dim, A):
    """Return a classifier function as expected by the CSI algorithm using the StructuredClassifier class"""
    def classifier(data):
        structured_classifier = StructuredClassifier(data, s_dim, phi, phi_dim, A)
        classif_rule,omega_classif = structured_classifier.run()
        score = lambda sa: squeeze(dot(omega_classif.transpose(),phi(sa)))
        return classif_rule,score
    return classifier

def get_least_squares_regressor_for_SCI(phi, sa_dim):
    #ax=b,find x
    def regressor(data):
        a=squeeze(phi(data[:,0:sa_dim]))
        b=data[:,sa_dim:]
        print "Regressor : data, a and b shapes and s_dim: "+str(data.shape)+str(a.shape)+str(b.shape)+str(sa_dim)
        x=lstsq(a,b)[0]
        print "Regressor : x.shape : "+str(x.shape)
        return lambda sa:dot(x.transpose(),phi(sa))
    return lambda data: regressor(data)

# <headingcell level=2>

# Experiment 1 Code

# <codecell>

            
data_random = inverted_pendulum_random_trace()
data_expert,policy,omega = inverted_pendulum_expert_trace(inverted_pendulum_reward)
random_falls_rate = - mean( data_random[:,5] )
expert_falls_rate = - mean( data_expert[:,5] )
print "Rate of falls for random controller "+str(random_falls_rate)
print "Rate of falls for expert controller "+str(expert_falls_rate)
classifier = get_structured_classifier_for_SCI(2, inverted_pendulum_phi, 30, ACTION_SPACE)#
regressor = get_least_squares_regressor_for_SCI(inverted_pendulum_phi, 3)
reward = CSI( data_expert, classifier, regressor, ACTION_SPACE, s_dim=2)
CSI_reward = lambda sas: squeeze(reward(sas[:3]))
CSI_reward(rand(5))
data_CSI,policy_CSI,omega_CSI = inverted_pendulum_expert_trace( reward=CSI_reward)
inverted_pendulum_plot_SAReward( reward, policy)
def mean_reward(s,p):
    actions = [a*ones(s.shape) for a in ACTION_SPACE]
    matrices = [zip_stack(s,p,a) for a in actions]
    return mean(array([squeeze(reward(m)) for m in matrices]), axis=0)
inverted_pendulum_plot( mean_reward)
inverted_pendulum_plot_policy( policy_CSI)

# <codecell>

states = rand(1000,2)*6-3
actions = policy(states)
data = hstack([states,actions])
classifier = get_structured_classifier_for_SCI(2, inverted_pendulum_phi, 30, ACTION_SPACE)
classif_rule,score = classifier(data)
data_0 = array([l for l in data if l[2]==0.])
data_1 = array([l for l in data if l[2]==1.])
data_2 = array([l for l in data if l[2]==2.])
inverted_pendulum_plot_policy(policy)
plot(data_0[:,0],data_0[:,1],ls='',marker='o')
plot(data_1[:,0],data_1[:,1],ls='',marker='o')
plot(data_2[:,0],data_2[:,1],ls='',marker='o')
inverted_pendulum_plot_policy( classif_rule )
two_args_score = lambda p,s: score(zip_stack(p,s,squeeze(classif_rule(zip_stack(p,s)))))
inverted_pendulum_plot(two_args_score)

# <codecell>

reg_S = rand(1000,2)*2-1
reg_A = policy(reg_S)
reg_X = hstack([reg_S,reg_A])
test_omega = rand(30,1)-0.5
reg_Y = dot(test_omega.transpose(),inverted_pendulum_phi(reg_X)).reshape(1000,1)

regressor = get_least_squares_regressor_for_SCI(inverted_pendulum_phi, 3)
reg = regressor(hstack([reg_X,reg_Y]))

inverted_pendulum_plot_SAReward( reg, policy)

X = linspace(-pi,pi,30)
Y = X
X,Y = meshgrid(X,Y)
XY=zip_stack(X,Y)
A = squeeze(policy(XY))
B = zip_stack(X,Y,A)
Z = squeeze(reg(B))

from matplotlib import cm
fig = figure()
ax=Axes3D(fig)
ax.plot_surface(X,Y,Z, cmap=cm.jet,cstride=1,rstride=1)
ax.scatter(reg_X[:,0],reg_X[:,1],reg_Y,c=reg_Y)
show()

# <codecell>

#Déroulage "à la main" de l'algo CSI, pour comprendre ce qui cloche
#def CSI(data, classifier, regressor, A, s_dim=1, gamma=0.9):#FIXME7: does not work for vectorial actions
#data = data_expert[:50,:]

data = rand(2000,2)*2*pi - pi
data_classif = hstack([data,policy(data)])
data_classif = hstack([data_classif,zeros((2000,2))])
for index, line in enumerate(data_classif):
    data_classif[index,-2:] = inverted_pendulum_next_state(data_classif[index,:2],data_classif[index,2])
random_policy= lambda s:choice(ACTION_SPACE)
vrandom_policy = non_scalar_vectorize( random_policy, (2,), (1,1) )
pi_r = lambda state: vrandom_policy(state).reshape(state.shape[:-1]+(1,))
data_regress = hstack([data,pi_r(data)])
data_regress = hstack([data_regress,zeros((2000,2))])
for index, line in enumerate(data_regress):
    data_regress[index,-2:] = inverted_pendulum_next_state(data_regress[index,:2],data_regress[index,2])

gamma=0.9
#classifier, regressor, deja definis
A = ACTION_SPACE
s_dim=2
phi=inverted_pendulum_phi
psi=inverted_pendulum_psi

# <codecell>

#Algo
column_shape = (len(data_classif),1)
s = data_classif[:,0:s_dim]
a = data_classif[:,s_dim:s_dim+1].reshape(column_shape)
sa = data_classif[:,0:s_dim+1]
s_dash = data_classif[:,s_dim+1:s_dim+1+s_dim]
#from sklearn import svm
#clf = svm.SVC(C=1000., probability=True)
#clf.fit(squeeze(psi(s)), a)
#clf_predict= lambda state : clf.predict(squeeze(psi(state)))
#vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
#pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
#clf_score = lambda sa : squeeze(clf.predict_proba(squeeze(psi(sa[:2]))))[sa[-1]]
#vscore = non_scalar_vectorize( clf_score,(3,),(1,1) )
#q = lambda sa: vscore(sa).reshape(sa.shape[:-1])
pi_c,q=policy,lambda sa: squeeze(dot(omega.transpose(),phi(sa)))
#pi_c,q = classifier(hstack([s,a]))
#Plots de la politique de l'expert, des données fournies par l'expert, de la politique du classifieur
inverted_pendulum_plot_policy(policy)
scatter(data_classif[:,0],data_classif[:,1],c=data_classif[:,2])
figure()
scatter(data_classif[:,0],data_classif[:,1],c=data_classif[:,2])
inverted_pendulum_plot_policy(pi_c)
##Plots de Q et de la fonction de score du classifieur et évaluation de la politique du classifieur
#phi=inverted_pendulum_phi
Q = lambda sa: squeeze(dot(omega.transpose(),phi(sa)))
Q_0 = lambda p,s:Q(zip_stack(p,s,0*ones(p.shape)))
Q_1 = lambda p,s:Q(zip_stack(p,s,1*ones(p.shape)))
Q_2 = lambda p,s:Q(zip_stack(p,s,2*ones(p.shape)))
q_0 = lambda p,s:q(zip_stack(p,s,0*ones(p.shape)))
q_1 = lambda p,s:q(zip_stack(p,s,1*ones(p.shape)))
q_2 = lambda p,s:q(zip_stack(p,s,2*ones(p.shape)))
inverted_pendulum_plot(Q_0)
inverted_pendulum_plot(Q_1)
inverted_pendulum_plot(Q_2)
inverted_pendulum_plot(q_0)
inverted_pendulum_plot(q_1)
inverted_pendulum_plot(q_2)
##FIXME: combien de fois le classifieur tombe-t-il ?

# <codecell>

#On  continue CSI
s = data_regress[:,0:s_dim]
a = data_regress[:,s_dim:s_dim+1].reshape(column_shape)
sa = data_regress[:,0:s_dim+1]
s_dash = data_regress[:,s_dim+1:s_dim+1+s_dim]
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-gamma*q(sa_dash)).reshape(column_shape)
r_min = min(hat_r)-1.*ones(column_shape)
#Plot des samples hat_r Pour chacune des 3 actions
sar = hstack([sa,hat_r])
for action in ACTION_SPACE:
    sr = array([l for l in sar if l[2]==action])
    axis([-pi,pi,-pi,pi])
    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', cmap = cm.jet );
    colorbar()
    figure()
#On continue SCI

##Avec l'heuristique : 
#regression_input_matrices = [hstack([s,action*ones(column_shape)]) for action in A] 
#def add_output_column( reg_mat ):
#    actions = reg_mat[:,-1].reshape(column_shape)
#    hat_r_bool_table = array(actions==a)
#    r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
#    output_column = hat_r_bool_table*hat_r+r_min_bool_table*r_min
#    return hstack([reg_mat,output_column])
#regression_matrix = vstack(map(add_output_column,regression_input_matrices))
#On plotte les mêmes données que juste précedemment, mais avec l'heuristique en prime
#for action in ACTION_SPACE:
#    sr = array([l for l in regression_matrix if l[2]==action])
#    axis([-pi,pi,-pi,pi])
#    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', cmap = cm.jet );
#    colorbar()
#    figure()

##Sans heuristique :
regression_matrix=sar

# <codecell>

#On continue CSI
def regressor(data):
    a=squeeze(phi(data[:,0:3]))
    #a=data[:,0:3]
    b=data[:,3:]
#    print x.shape
    #return lambda sa:dot(x,phi(sa))
    ##Trichons
    def triche(sa):
        s_dash = inverted_pendulum_next_state(sa[:2],sa[-1])
        a_dash = pi_c(s_dash)
        sa_dash=hstack([s_dash,a_dash])
        return q(sa)-gamma*q(sa_dash)
    vtriche=non_scalar_vectorize(triche, (3,), (1,1))
    return lambda sa:vtriche(sa)
reg = regressor( regression_matrix )
CSI_reward = lambda sas: squeeze(reg(sas[:3]))
#On plotte les rewards en fonction de l'action
for action in ACTION_SPACE:
    sr = array([l for l in regression_matrix if l[2]==action])
    R = lambda p,s: squeeze( reg(zip_stack(p,s,action*ones(p.shape))))
    pos = linspace(-pi,pi,30)
    speed = linspace(-pi,pi,30)
    pos,speed = meshgrid(pos,speed)
    Z = R(pos,speed)
    figure()
    contourf(pos,speed,Z,50)
    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', )#cmap = cm.jet );
    clim(vmin=min(Z.reshape(-1)),vmax=max(Z.reshape(-1)))
    colorbar()
def mean_reward(s,p):
    actions = [a*ones(s.shape) for a in ACTION_SPACE]
    matrices = [zip_stack(s,p,a) for a in actions]
    return mean(array([squeeze(reg(m)) for m in matrices]), axis=0)
inverted_pendulum_plot(mean_reward)

# <codecell>

def inverted_pendulum_expert_trace( reward ):
    data = inverted_pendulum_random_trace(reward=reward)
    policy,omega = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=inverted_pendulum_phi, phi_dim=30, iterations_max=10 )
    inverted_pendulum_plot(inverted_pendulum_V(omega))
    two_args_pol = lambda p,s:squeeze(policy(zip_stack(p,s)))
    inverted_pendulum_plot(two_args_pol,contour_levels=3)
    return inverted_pendulum_trace( policy,run_length=EXPERT_RUN_LENGTH ),policy,omega
#data_expert,policy,omega = inverted_pendulum_expert_trace(inverted_pendulum_reward)
data_CSI,policy_CSI,omega_CSI = inverted_pendulum_expert_trace(CSI_reward)
CSI_falls_rate = - mean( data_CSI[:,5] )
print "Rate of falls for IRL controller "+str(CSI_falls_rate)

# <codecell>

a

