#Common code for all pipelines
from pylab import *
from collections import deque

def action2command( a ):
    if a == "Left":
        return array([[0],[0],[1],[0]])
    elif a == "Right":
        return array([[1],[0],[0],[0]])
    elif a == "Up":
        return array([[0],[1],[0],[0]])
    elif a == "Down":
        return array([[0],[0],[0],[1]])
    else:
        return None

#Returns the appropriate command to reach the goal from the current position
def command( goal, pos ):
    action = None
    if goal[0] < pos[0]:#Align the x first
        action = "Left"
    elif goal[0] > pos[0]:
        action = "Right"
    else: #Align Y if x are aligned
        action = "Up"
    return action2command( action )

#Add a certain level and kind of noise to the given command
g_fCommandNoiseLevel = 10
def add_noise( command ):
    answer = command + random_sample( command.shape )*g_fCommandNoiseLevel
    answer /= norm( answer, 1 )
    assert( abs( sum( answer ) - 1.) < 0.000001 )
    return answer    

#Return the state vector (in the RL sense) from the current state (in the "state machine" sense)
def get_rl_state( position, aNoisyCom ):
    com = map( int, aNoisyCom/(1./3.) ) #discretization of the prob space
    return concatenate([position,com])
    
#Return the action chosen by the expert in integer format
def get_action( aCommand ):
    return abs(dot( [0,1,2,3],aCommand ))

#Move the arm
g_aUpdate = [[1,0],[0,1],[-1,0],[0,-1]] #Encodes which commands corresponds to which direction
def update_position( aPosition, aCommand ):
    command_index = filter( lambda i: aCommand[i]==1,range(0,len(aCommand)))
    assert( len( command_index ) == 1 )
    command_index = command_index[0]
    update = array( g_aUpdate[ command_index ] )
    answer = aPosition + update
    #staying in bounds
    if answer[0] < 0:
        answer[0] = 0
    elif answer[0] > 2:
        answer[0] = 2
    if answer[1] < 0:
        answer[1] = 0
    elif answer[1] > 3:
        answer[1] = 3
    return answer
    
    

#Check if we attained the goal
def near_enough( aGoal, aPosition ):
    return all( aGoal == aPosition )

