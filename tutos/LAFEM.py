from numpy import *
import scipy.linalg
import os
import sys
class LAFEM:
   def __init__( self ):
      if self.__class__ == LAFEM:
         raise NotImplementedError, "Cannot create object of class LAFEM"

   def l( self, s, a ):
      raise NotImplementedError, "Cannot call abstract method"

   def mu_E( self, s, a ):
      raise NotImplementedError, "Cannot call abstract method"

   def alpha( self, t ):
      raise NotImplementedError, "Cannot call abstract method"

   data=[]

   theta_0=array([])

   Threshold = 'a'

   T = -1

   A=[]

   def run( self ):
      theta = self.theta_0.copy()
      best_theta = theta.copy()
      best_norm = 1000000.
      best_iter = 0

      #for t in range(0,self.T):
      t=-1
      while True:#Do...while loop
         t += 1

         DeltaTheta = zeros(( self.theta_0.size, 1 ))

         for sa in self.data:

            val = -Inf
            a_star = array([])
            for a in self.A:
               newval = dot( theta.transpose(), self.mu_E( sa[0], a ) ) + self.l( sa[0], a )
               assert(newval.size == 1)
               if newval[0] > val:
                  val = newval
                  a_star = a

            DeltaTheta = DeltaTheta + self.mu_E( sa[0], a_star ) - self.mu_E( sa[0], sa[1] )

         DeltaTheta = DeltaTheta / len(self.data) #1/N
         norm = scipy.linalg.norm(DeltaTheta)
         if norm > 0.:
             theta = theta - self.alpha( t ) * DeltaTheta / norm
         sys.stderr.write("Gradient norm "+str(norm)+", step : "+str(self.alpha(t))+", iteration : "+str(t)+"\n")

         if norm < best_norm:
             best_norm = norm
             best_theta = theta.copy()
             best_iter = t
         if norm < self.Threshold or (self.T != -1 and t >= self.T):
             break

      sys.stderr.write("Gradient of norm : "+str(best_norm)+", at iteration : "+str(best_iter)+"\n")
      return best_theta
