#!/usr/bin/python
from pylab import *
from collections import deque

#Returns the appropriate command to reach the goal from the current position
def command( goal, pos ):
    answer = zeros([4,1]) #Four component, respectively right, up, left, down
    diff = goal - pos
    axis = (max(abs(diff))==abs(diff))*1. #We select the axis with the biggest diff
    if sign(dot(axis.transpose(),diff)) == 1 : #positive diff means right or up
        answer[0:2] = axis 
    else:#negative diff means left or down
        answer[2:4] = axis
    return answer

#Add a certain level and kind of noise to the given command
g_fCommandNoiseLevel = 0.1
def add_noise( command ):
    answer = command + random_sample( command.shape )*g_fCommandNoiseLevel
    answer /= norm( answer, 1 )
    assert( abs( sum( answer ) - 1.) < 0.000001 )
    return answer    

#Return the state vector (in the RL sense) from the current state (in the "state machine" sense)
def get_state( qPositions, aNoisyCom ):
    com = aNoisyCom
    relative_positions = zeros([(g_iNLastPos-1)*2,1])
    if( len( qPositions ) >= 2 ):
        off = qPositions[-1] #Positions are centered
        for i in range(0,len( qPositions )-1 ):
            relative_positions[2*i:2*i+2] = qPositions[i] - off
    return concatenate([relative_positions,com])
    
#Return the action chosen by the expert in intger format
def get_action( aCommand ):
    return abs(dot( [0,1,2,3],aCommand ))

#Move the arm
g_aUpdate = [[[1],[0]],[[0],[1]],[[-1],[0]],[[0],[-1]]] #Encodes which commands corresponds to which direction
g_fUpdateScale = 0.1
g_fUpdateNoiseLevel = 0.05
def update_position( aPosition, aCommand ):
    command_index = filter( lambda i: aCommand[i],range(0,len(aCommand)))
    assert( len( command_index ) == 1 )
    command_index = command_index[0]
    update = array( map( lambda l: [float(l[0])], g_aUpdate[ command_index ] ) )
    update *= g_fUpdateScale + random_sample( update.shape )*g_fUpdateNoiseLevel
    # print "update"
    # print update
    return aPosition + update

#Check if we attained the goal
def near_enough( aGoal, aPosition ):
    return norm( aGoal - aPosition,1 ) < 1.5*g_fUpdateScale


g_fXYMax = 5 #and Min is 0
g_aGoal = rand(2,1)*g_fXYMax
g_aPosition_t = rand(2,1)*g_fXYMax
while near_enough( g_aGoal, g_aPosition_t ):
    g_aPosition_t = rand(2,1)*g_fXYMax
g_iNLastPos = 3
g_qPositions = deque([],maxlen=g_iNLastPos)
g_bContinue = True

while g_bContinue:
    g_aCommand = command( g_aGoal, g_aPosition_t )
    g_aNoisyCom = add_noise( g_aCommand )
    g_qPositions.append( g_aPosition_t )
    l_as = get_state( g_qPositions, g_aNoisyCom )
    l_aa = get_action( g_aCommand )
    for l in l_as:
        print l[0]," ",
    print l_aa[0]
    # print "erycdjqsvhbfqsdfs"
    # print g_aGoal
    # print g_aPosition_t
    # print g_aCommand
    g_aPosition_t = update_position( g_aPosition_t, g_aCommand )
    if near_enough( g_aGoal, g_aPosition_t ):
        g_bContinue = False
    


