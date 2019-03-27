"""
utility function
"""

import sys
import os
import numpy as np
from mpi4py import MPI
import time
from scipy.interpolate import interp1d
import cmath

def distance(X1,Y1, X2,Y2):
    return np.sqrt( (X2-X1)**2. + (Y2-Y1)**2. )
    

class InterfaceCurvature(object):

    def __init__(self, X, Y, N_Pt_vor):
        '''
        Initialisation
        '''
        self.X = X
        self.Y = Y
        self.N_Pt_vor = N_Pt_vor
    
    # calculating the arc-length by the Euclidean distance    
    def ArcLength(self):
        delta = np.zeros(self.N_Pt_vor)
        delta[0] = distance(self.X[0],self.Y[0],self.X[1],self.Y[1])
        delta[self.N_Pt_vor-1] =\
        distance(self.X[self.N_Pt_vor-2],self.Y[self.N_Pt_vor-2],self.X[self.N_Pt_vor-1],self.Y[self.N_Pt_vor-1])

        for ArrySze in range(1,self.N_Pt_vor-1):
          delta[ArrySze] = 0.5*distance(self.X[ArrySze+1],self.Y[ArrySze+1],self.X[ArrySze-1],self.Y[ArrySze-1])            
        
        return delta
    
    
    def UnitTangentVector(self):
        deltas = self.ArcLength()
          
        Tvec_X = np.zeros(self.N_Pt_vor)
        Tvec_Y = np.zeros(self.N_Pt_vor)
        T_X = np.zeros(self.N_Pt_vor)
        T_Y = np.zeros(self.N_Pt_vor) 
        
        Tvec_X[0] = (self.X[1]-self.X[0])/deltas[0]
        Tvec_Y[0] = (self.Y[1]-self.Y[0])/deltas[0]

        Tvec_X[self.N_Pt_vor-1] = (self.X[self.N_Pt_vor-1]-self.X[self.N_Pt_vor-2])/deltas[self.N_Pt_vor-1]
        Tvec_Y[self.N_Pt_vor-1] = (self.Y[self.N_Pt_vor-1]-self.Y[self.N_Pt_vor-2])/deltas[self.N_Pt_vor-1]

        for ArrySze in range(1,self.N_Pt_vor-1):
          Tvec_X[ArrySze] = 0.5*( self.X[ArrySze+1]-self.X[ArrySze-1] )/deltas[ArrySze]
          Tvec_Y[ArrySze] = 0.5*( self.Y[ArrySze+1]-self.Y[ArrySze-1] )/deltas[ArrySze]

        for ArrySze in range(self.N_Pt_vor):
          T_X[ArrySze] = Tvec_X[ArrySze]/np.sqrt( Tvec_X[ArrySze]**2. + Tvec_Y[ArrySze]**2. )
          T_Y[ArrySze] = Tvec_Y[ArrySze]/np.sqrt( Tvec_X[ArrySze]**2. + Tvec_Y[ArrySze]**2. )

        return Tvec_X, Tvec_Y, T_X, T_Y 

class InitCirculation(object):

    def __init__(self, X, Y, At, N_Pt_vor, Ampl):
        '''
        Initialisation
        '''
        self.X = X
        self.Y = Y
        self.At = At
        self.N_Pt_vor = N_Pt_vor
        self.Ampl = Ampl

    def Kappa_0(self):     
        Kappa = np.zeros(self.N_Pt_vor)
        r = (1.-self.At)/(1.+self.At) 
        for ArrySize in range(self.N_Pt_vor):
          Kappa[ArrySize] =\
          ( 1. + 2.*np.pi*self.Ampl*( self.At*np.sin(2*np.pi*self.X[ArrySize]) - 2.*np.sqrt(r)/(1.+r)*np.cos(2*np.pi*self.X[ArrySize]) ) )
          Kappa[ArrySize] = Kappa[ArrySize]/self.N_Pt_vor       
        return Kappa
 

class InitVortexStrength(object):

    def __init__(self, X, Y, At, N_Pt_vor, Ampl, Kappa):
        '''
        Initialisation
        '''
        self.X = X
        self.Y = Y
        self.At = At
        self.N_Pt_vor = N_Pt_vor
        self.Ampl = Ampl
        self.Kappa = Kappa

    def VortexStrength_0(self):
        # calculating arc length
        Obj_0 = InterfaceCurvature(self.X, self.Y, self.N_Pt_vor)
        deltas = Obj_0.ArcLength()
                
        gamma = np.zeros(self.N_Pt_vor)
        for ArrySize in range(self.N_Pt_vor):
          gamma[ArrySize] = self.Kappa[ArrySize]/deltas[ArrySize]  
        #print(np.shape(gamma)) 
        return gamma                  
           
           
class GeneralFunction(object):

    def __init__(self, X, Y, N_Pt_vor, delta, Kappa):
        '''
        Initialisation
        '''
        self.X = X
        self.Y = Y
        self.N_Pt_vor = N_Pt_vor
        self.delta = delta 
        self.Kappa = Kappa

    def InterfaceVelocity(self):
        uiv = np.zeros( (self.N_Pt_vor,2) )

        cnst = 2.*np.pi
        for iArrySze in range(self.N_Pt_vor-1):
          sum_1 = 0.
          sum_2 = 0.   
          for jArrySze in range(self.N_Pt_vor-1):
            if jArrySze != iArrySze:
              denom = np.cosh(cnst*(self.Y[iArrySze]-self.Y[jArrySze])) - np.cos(cnst*(self.X[iArrySze]-self.X[jArrySze])) + self.delta**2.

              sum_1 += + 0.5*self.Kappa[jArrySze]*np.sinh(cnst*(self.Y[iArrySze]-self.Y[jArrySze]))/denom
              sum_2 += - 0.5*self.Kappa[jArrySze]*np.sin (cnst*(self.X[iArrySze]-self.X[jArrySze]))/denom 
          
          uiv[iArrySze, 0] = sum_1
          uiv[iArrySze, 1] = sum_2
          
        # periodic domain condition
        uiv[self.N_Pt_vor-1,0] = uiv[0,0]
        uiv[self.N_Pt_vor-1,1] = uiv[0,1]  
        
        return uiv

#    def Above_Below_InterfaceVelocity(self):
#        # calculating Lagrangian velocities at the interface - below and above
#        U = np.zeros( (self.N_Pt_vor,2) )      # 1 -> below and 2 -> above 
#        V = np.zeros( (self.N_Pt_vor,2) )  
    
#        Obj_0 = InitVortexStrength(self.X, self.Y, self.At, self.N_Pt_vor, self.Ampl)
#        gamma = Obj_0.VortexStrength_0()
        
#        Obj_1 = InterfaceCurvature(self.X, self.Y, self.N_Pt_vor)
#        Tvec_X, Tvec_Y, T_X, T_Y = Obj_1.UnitTangentVector()
        
#        uiv = self.InterfaceVelocity()
        
#        for ArrySze in range(self.N_Pt_vor):
#          U[ArrySze,0] = uiv[ArrySze,0] - 0.5*gamma[ArrySze]*(self.alpha-1.)*T_X[ArrySze] 
#          V[ArrySze,0] = uiv[ArrySze,1] - 0.5*gamma[ArrySze]*(self.alpha-1.)*T_Y[ArrySze]

#          U[ArrySze,1] = uiv[ArrySze,0] - 0.5*gamma[ArrySze]*(self.alpha+1.)*T_X[ArrySze] 
#          V[ArrySze,1] = uiv[ArrySze,1] - 0.5*gamma[ArrySze]*(self.alpha+1.)*T_Y[ArrySze]        
          
#        return U, V  
          

class Clc_Circulation(object):

    def __init__(self, X, Y, N_Pt_vor, delta, Kappa):
        '''
        Initialisation
        '''
        self.X = X
        self.Y = Y
        self.N_Pt_vor = N_Pt_vor
        self.delta = delta 
        self.Kappa = Kappa                     

    def C_InterfaceVelocity(self):
        uiv = np.zeros( (self.N_Pt_vor,2) )
        
        cnst = 2.*np.pi        
        for iArrySze in range(self.N_Pt_vor-1):
          sum_1 = 0.
          sum_2 = 0.   
          for jArrySze in range(self.N_Pt_vor-1):
            if jArrySze != iArrySze:
              denom = np.cosh(cnst*(self.Y[iArrySze]-self.Y[jArrySze])) - np.cos(cnst*(self.X[iArrySze]-self.X[jArrySze])) + self.delta**2.

              sum_1 += + 0.5*self.Kappa[jArrySze]*np.sinh(cnst*(self.Y[iArrySze]-self.Y[jArrySze]))/denom
              sum_2 += - 0.5*self.Kappa[jArrySze]*np.sin (cnst*(self.X[iArrySze]-self.X[jArrySze]))/denom 
          
          uiv[iArrySze, 0] = sum_1
          uiv[iArrySze, 1] = sum_2
          
        # periodic domain condition
        uiv[self.N_Pt_vor-1,0] = uiv[0,0]
        uiv[self.N_Pt_vor-1,1] = uiv[0,1]   
        
        return uiv          



#class TimeStep(object):

#    def __init__(self, X, Y, N_Pt_vor, delta, Kappa):
#        '''
#        Initialisation
#        '''
#        self.X = X
#        self.Y = Y
#        self.N_Pt_vor = N_Pt_vor
#        self.delta = delta 
#        self.Kappa = Kappa                     
        
#    def RungeKutta4(self):
            
        
        
