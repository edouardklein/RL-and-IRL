from pylab import *

STATE_DIM = 8
ACTION_CARD = 4
PSI_DIM = 338
PHI_DIM = PSI_DIM*ACTION_CARD
g_fUpdateScale = 0.1
g_lRelativePosCenters = [] #Centers of the gaussians for the first 4 components 
g_fPosSigma = g_fUpdateScale
g_lClassifierCenters = [] #Centers of the gaussians for the following 4 components
g_fClassifierSigma = 0.3
#First four components
relativePosCoords = [ - g_fUpdateScale, 0., g_fUpdateScale]
for x1 in relativePosCoords:
    for y1 in relativePosCoords:
        for x2 in relativePosCoords:
            for y2 in relativePosCoords:
                g_lRelativePosCenters.append(array([x1,y1,x2,y2]))
classifierCoords = [0., 1./3.,2./3., 1.]
for x1 in classifierCoords:
    for y1 in classifierCoords:
        for x2 in classifierCoords:
            for y2 in classifierCoords:
                g_lClassifierCenters.append(array([x1,y1,x2,y2]))
    
def psi( s ): #Gaussian network
    answer = zeros( [PSI_DIM, 1] )
    i = 0
    x = s[0:4]
    for center in g_lRelativePosCenters:
        toSum = map( lambda a : a*a/(2*g_fPosSigma*g_fPosSigma), (x - center) )
        answer[i] = exp( - sum( toSum ) )
        i += 1
    for center in g_lClassifierCenters:
        toSum = map( lambda a : a*a/(2*g_fClassifierSigma*g_fClassifierSigma), (x - center) )
        answer[i] = exp( - sum( toSum ) )
        i += 1
    answer[i] = 1.
    return answer

def phi( s, a ):
    answer = zeros([PHI_DIM, 1])
    index = int(a)*PSI_DIM
    answer[index:index+PSI_DIM] = psi( s )
    return answer
