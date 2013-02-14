# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Mountain Car
from stuff import *
from pylab import *
from random import *
import numpy
from rl import *

ACTION_SPACE=[-1,0,1]

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

def mountain_car_boubou_traj():
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

data_r = vstack([mountain_car_boubou_traj() for i in range(0,400)])
savetxt("mountain_car_boubou_trajs.mat",data_r)

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

#function reward=relative_entropy_boularias(phi,n_s,gamma,data_c,data_r,L_c,H_c,L_r,H_r,delta,epsilon_re,N_final)
def relative_entropy(data_c,data_r,delta):
    #size_phi=size(phi);
    
    #feature_c=zeros(L_c,size_phi(2));
    #feature_r=zeros(L_r,size_phi(2));
    feature_c=[]
    feature_r=[]
    
    #theta=zeros(1,size_phi(2));
    theta=zeros((1,150))
    #for i=1:L_c
        #for j=1:H_c
    eoe_indices = [i for i in range(0,len(data_c)) if end_of_episode(data_c,i)]
    for start_index,end_index in zip( [0] + map(lambda x:x+1,eoe_indices[:-1]),eoe_indices):
        #feature_c(i,:)=feature_c(i,:)+gamma^j*phi(data_c(j+H_c*(i-1),1)+n_s*(data_c(j+H_c*(i-1),2)-1),:);
        data_MC=data_c[start_index:end_index+1,:3]
        GAMMAS = range(0,len(data_MC))
        GAMMAS = array(map( lambda x: pow(GAMMA,x), GAMMAS))
        state_action = data_MC[0,:3]
        mu = None
        if len(data_MC) > 1:
            mu = dot( GAMMAS,squeeze(phi(data_MC[:,:3])))
        else:
            mu = squeeze(phi(squeeze(data_MC[:,:3])))
        feature_c.append(mu)   
    feature_c=array(feature_c)

    #feature_c_mean=mean(feature_c,1);
    feature_c_mean=mean(feature_c,0);
    
    #epsilon=sqrt(-log2(1-delta)/(2*H_c))*(gamma^(H_c+1)-1)/(gamma-1);
    #epsilon=sqrt(-log2(1-delta)/(2*300))*(pow(GAMMA,(300+1))-1)/(GAMMA-1);
    epsilon = 0.01
    print "epsilon "+str(epsilon)
    
    #for i=1:L_r
    #    for j=1:H_r
    eoe_indices = [i for i in range(0,len(data_r)) if end_of_episode(data_r,i)]
    for start_index,end_index in zip( [0] + map(lambda x:x+1,eoe_indices[:-1]),eoe_indices):
        #feature_r(i,:)=feature_r(i,:)+gamma^j*phi(data_r(j+H_r*(i-1),1)+n_s*(data_r(j+H_r*(i-1),2)-1),:)
        data_MC=data_r[start_index:end_index+1,:3]
        GAMMAS = range(0,len(data_MC))
        GAMMAS = array(map( lambda x: pow(GAMMA,x), GAMMAS))
        state_action = data_MC[0,:3]
        mu = None
        if len(data_MC) > 1:
            mu = dot( GAMMAS,squeeze(phi(data_MC[:,:3])))
        else:
            mu = squeeze(phi(squeeze(data_MC[:,:3])))
        feature_r.append(mu)
        #    end   
        #end
    feature_r=array(feature_r)

    #Criterion=epsilon_re+1;
    criterion=numpy.inf
    #counter=1;
    counter=1

    #while Criterion>epsilon_re&&counter<N_final
    while criterion > 0.01 and counter < 1000:
        #buffer_derivative_t=zeros(1,size_phi(2));
        derivative = zeros((1,150))
        #buffer_t=0;
        t = 0
        #buffer_theta=theta;
        #for i=1:L_r
        for i in range(0,len(feature_r)):
            #buffer_derivative_t=buffer_derivative_t+exp(theta*feature_r(i,:)')*feature_r(i,:);
            derivative += exp(dot(theta,feature_r[i,:].transpose()))*feature_r[i,:]
            #buffer_t=buffer_t+exp(theta*feature_r(i,:)');
            t +=  exp(dot(theta,feature_r[i,:].transpose()))
            #end    

        #buffer_derivative=feature_c_mean-buffer_derivative_t/(buffer_t)-sign(theta)*epsilon;
        derivative=feature_c_mean-derivative/(t)-sign(theta)*epsilon

        print "Boubou run \t"+str(counter)+" criterion is \t"+str(criterion)+" ||derivative|| is \t"+str(norm(derivative))
        
        #if norm(buffer_derivative)==0
        if norm(derivative)==0:
            #theta=buffer_theta;
            break
        else:
            #theta=buffer_theta+(1/counter)*buffer_derivative/norm(buffer_derivative,2);
            delta_theta = (100./float(counter))*derivative/norm(derivative,2)
            #end
        #Criterion=norm(buffer_theta-theta,2);
        criterion=norm(delta_theta,2)
        theta+=delta_theta
        #counter=counter+1;    
        counter += 1
        #end    
    print "Stop boubou @ run "+str(counter-1)+" criterion is "+str(criterion)
    #reward=phi*theta';
    return lambda sas: dot(theta,phi(sas[:3]))[0]
    

#data_r = genfromtxt("mountain_car_batch_data.mat")
data_r = genfromtxt("mountain_car_boubou_trajs.mat")

RE_reward = relative_entropy(TRAJS, data_r, 0.99)
vRE_reward = non_scalar_vectorize( RE_reward, (5,),(1,1) )
data = genfromtxt("mountain_car_batch_data.mat")
data[:,5] = squeeze(vRE_reward(data[:,:5]))
policy_RE,omega_RE = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=150, iterations_max=20 )#None,zeros((75,1))#

# <codecell>

plottable_episode_length = mountain_car_episode_vlength(policy_RE)
X = linspace(-1.2,0.6,20)
Y = linspace(-0.07,0.07,20)
X,Y = meshgrid(X,Y)
#Z2 = plottable_episode_length(X,Y)
figure()
mountain_car_plot_policy(policy_RE)
figure()
#contourf(X,Y,Z2,50)
#colorbar()

# <codecell>

def mountain_car_testing_state():
    position = numpy.random.uniform(low=-1.2,high=0.5)
    speed = numpy.random.uniform(low=-0.07,high=0.07)
    return array([position,speed])

def mountain_car_mean_performance(policy):
    return mean([mountain_car_episode_length(state[0],state[1],policy) for state in [mountain_car_testing_state() for i in range(0,100)]])

mountain_car_plot_policy(policy_RE)
print mountain_car_mean_performance(policy_RE)

