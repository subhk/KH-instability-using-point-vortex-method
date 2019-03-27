"""
function to handle point vortices insertion
and delete
"""

import sys
import os
import numpy as np
from mpi4py import MPI
#import time
#from scipy.interpolate import interp1d
#import cmath


def threshold_v(val, delta):
    '''
    threshold distance between two point vortices
    '''
    return val*delta    

def ArcLength(X, Y):
    '''
    arc length 
    '''
    X_diff = np.diff(X)
    Y_diff = np.diff(Y)
    return np.sqrt(X_diff**2. + Y_diff**2.)    
                
                 
class PointVortexInsert(object):

    def __init__(self, X_loc, Y_loc, N_vor, deltax, threshold, GAMMA):
        '''
        Initialisation
        '''
        self.X_loc = X_loc
        self.Y_loc = Y_loc
        self.N_vor = N_vor
        self.deltax = deltax 
        self.threshold = threshold 
        self.GAMMA = GAMMA 
    
    def InsertVortex(self): 
        
        Nw_X_Loc = np.zeros( int(4*self.N_vor) )   # Xp_new
        Nw_Y_Loc = np.zeros( int(4*self.N_vor) )   # Yp_new        
        Nw_GAMMA = np.zeros( int(4*self.N_vor) )    # Nw_GAMMA
        Nw_Loc   = np.zeros( int(4*self.N_vor) )    # Nw_Loc        
                     
        # Arc Length
        #arc_len = ArcLength(self.X_loc, self.Y_loc)
        
        Nw_Pts = 0        
        for ArrySize in range(self.N_vor):
          p1 = self.X_loc[ArrySize+1] - self.X_loc[ArrySize] 
          p2 = self.Y_loc[ArrySize+1] - self.Y_loc[ArrySize]
          rel_dist = np.sqrt( p1**2. + p2**2. )
          
          # insert criteria -> arc_len > threshols_value
          if rel_dist > threshold_v(self.threshold, self.deltax): # and N_vor_add>0:
            #print('I am here too', ArrySize, arc_len[ArrySize], threshold_v(self.threshold, self.deltax) )
            #print(np.shape(Nw_X_loc))
            #print('going through here')
            Nw_X_Loc[ArrySize+Nw_Pts+1] = self.X_loc[ArrySize] + 0.5*p1
            Nw_Y_Loc[ArrySize+Nw_Pts+1] = \
            np.interp(Nw_X_Loc[ArrySize+Nw_Pts+1], self.X_loc[ArrySize:ArrySize+1], self.Y_loc[ArrySize:ArrySize+1]) 
            
        # interpolate the circulation to the new grid points
            Nw_GAMMA[ArrySize+Nw_Pts+1] =\
            np.interp(Nw_X_Loc[ArrySize+Nw_Pts+1], self.X_loc[ArrySize:ArrySize+1], self.GAMMA[ArrySize:ArrySize+1] )
            Nw_Pts = Nw_Pts + 1
            Nw_Loc[Nw_Pts] = ArrySize + Nw_Pts            
            #print('ArrySize + Nw_Pts= ', ArrySize + Nw_Pts)
                      

        N_vor_update = self.N_vor + Nw_Pts + 1
       
        tmpA = np.zeros( int(4*self.N_vor) )  # X_loc_update
        tmpB = np.zeros( int(4*self.N_vor) )  # Y_loc_update
        tmpC = np.zeros( int(4*self.N_vor) )  # GAMMA_update
        
        #Pt_Vor_Loc = np.zeros(Nw_Pts)                
        #print(Nw_Pts)
        
        if Nw_Pts != 0:
          #print(Nw_Loc[1])
          #Pt_Vor_Loc[0:Nw_Pts-1] = Nw_Loc[0:Nw_Pts-1]
          Cntr = 0
          for ArrySize in range(N_vor_update):           
            if ArrySize == Nw_Loc[Cntr] and Nw_Pts >= Cntr:
              tmpA[ArrySize] = Nw_X_Loc[ArrySize]   
              tmpB[ArrySize] = Nw_Y_Loc[ArrySize] 
              tmpC[ArrySize] = Nw_GAMMA[ArrySize]
              Cntr = Cntr + 1
            else:
              tmpA[ArrySize] = self.X_loc[ArrySize-Cntr+1]   
              tmpB[ArrySize] = self.Y_loc[ArrySize-Cntr+1] 
              tmpC[ArrySize] = self.GAMMA[ArrySize-Cntr+1]
              
          print('No of Point Vortices have been added', Nw_Pts)
          
          X_loc_update = np.zeros(N_vor_update)
          Y_loc_update = np.zeros(N_vor_update)
          GAMMA_update = np.zeros(N_vor_update)
          
          for ArrySize in range(N_vor_update):
            X_loc_update[ArrySize] = tmpA[ArrySize]
            Y_loc_update[ArrySize] = tmpB[ArrySize]
            GAMMA_update[ArrySize] = tmpC[ArrySize]
              
        else:
          N_vor_update = self.N_vor + 1        
           
          X_loc_update = np.copy(self.X_loc)
          Y_loc_update = np.copy(self.Y_loc)
          GAMMA_update = np.copy(self.GAMMA)
                         
        return N_vor_update, Nw_Pts, X_loc_update, Y_loc_update, GAMMA_update  
        #return results  

         
class PointVortexDelete(object):

    def __init__(self, X_loc, Y_loc, N_vor, deltax, threshold, GAMMA):
        '''
        Initialisation
        '''
        self.X_loc = X_loc
        self.Y_loc = Y_loc
        self.N_vor = N_vor
        self.deltax = deltax 
        self.threshold = threshold       
        self.GAMMA = GAMMA  
         
    def DeleteVortices(self): 
        
        tmpA = np.zeros( int(4*self.N_vor) )  # X_loc_update
        tmpB = np.zeros( int(4*self.N_vor) )  # Y_loc_update
        tmpC = np.zeros( int(4*self.N_vor) )  # GAMMA_update
          
        #arc_len = ArcLength(self.X_loc, self.Y_loc)
           
        ArrySize = 0
        Cntr = 0
        while ArrySize <= self.N_vor-2:
          p1 = self.X_loc[ArrySize+1] - self.X_loc[ArrySize] 
          p2 = self.Y_loc[ArrySize+1] - self.Y_loc[ArrySize]
          rel_dist = np.sqrt( p1**2. + p2**2. )
          if rel_dist < threshold_v(self.threshold, self.deltax):
            Cntr += 1
            ArrySize += 2
          else:
            ArrySize += 1
              
        Pts_to_Delete = Cntr        
        Pts_Delete_Cnt = 0
        #print('Pts_to_Delete= ', Pts_to_Delete)
        
        if Pts_to_Delete != 0:
          ArrySize = 0
          while ArrySize <= self.N_vor-2:
            p1 = self.X_loc[ArrySize+1] - self.X_loc[ArrySize] 
            p2 = self.Y_loc[ArrySize+1] - self.Y_loc[ArrySize]
            rel_dist = np.sqrt( p1**2. + p2**2. )
            if rel_dist < threshold_v(self.threshold, self.deltax) and Pts_to_Delete > Pts_Delete_Cnt:
              #print('I am here and localtion is = ', ArrySize)
              #print('Array Size=', ArrySize)
              tmpA[ArrySize-Pts_Delete_Cnt] = self.X_loc[ArrySize]
              tmpB[ArrySize-Pts_Delete_Cnt] = self.Y_loc[ArrySize]             
              tmpC[ArrySize-Pts_Delete_Cnt] = self.GAMMA[ArrySize]
              
              tmpA[ArrySize-Pts_Delete_Cnt+1] = self.X_loc[ArrySize+2]
              tmpB[ArrySize-Pts_Delete_Cnt+1] = self.Y_loc[ArrySize+2]
              tmpC[ArrySize-Pts_Delete_Cnt+1] = self.GAMMA[ArrySize+2]
              
              Pts_Delete_Cnt += 1
              ArrySize += 2
               
            else:
              #print('I am inside else and Pts_Delete_Cnt = ', Pts_Delete_Cnt)
              #print('Size is= ', ArrySize-Pts_Delete_Cnt)
              tmpA[ArrySize-Pts_Delete_Cnt] = self.X_loc[ArrySize]
              tmpB[ArrySize-Pts_Delete_Cnt] = self.Y_loc[ArrySize]             
              tmpC[ArrySize-Pts_Delete_Cnt] = self.GAMMA[ArrySize]              
              
              ArrySize += 1
              
          #Sze_Pts_to_Delete = len(X_loc_update)    
          Pts_after_Delete = ArrySize+1
          #print('ArrySize= ', ArrySize)
          #print('N_vor= ', self.N_vor)
                    
          tmpA[ArrySize] = self.X_loc[self.N_vor-1]
          tmpB[ArrySize] = self.Y_loc[self.N_vor-1]    
          tmpC[ArrySize] = self.GAMMA[self.N_vor-1]
          
          #print('ArrySize+1= ', ArrySize+1)
          #print('self.N_vor+1= ', self.N_vor+1)
          #print('Size= ', np.shape(self.X_loc))
          #print('Value= ', self.X_loc[self.N_vor+1])
          
          tmpA[ArrySize+1] = self.X_loc[self.N_vor]
          tmpB[ArrySize+1] = self.Y_loc[self.N_vor]    
          tmpC[ArrySize+1] = self.GAMMA[self.N_vor]
          
          #if Pts_to_Delete > 0:
          #  print('No of Point Vortices have been deleted ', Pts_to_Delete)
          #  print('Vorticies after deletion= ', Pts_after_Delete)
          
          X_loc_update = np.zeros(Pts_after_Delete+1)
          Y_loc_update = np.zeros(Pts_after_Delete+1)
          GAMMA_update = np.zeros(Pts_after_Delete+1)
          
          for ArrySize in range(Pts_after_Delete+1):
            X_loc_update[ArrySize] = tmpA[ArrySize]
            Y_loc_update[ArrySize] = tmpB[ArrySize]
            GAMMA_update[ArrySize] = tmpC[ArrySize]
          
        else:
          #print('No point vortices deteleted')
          Pts_after_Delete = self.N_vor+1
            
          X_loc_update = np.copy(self.X_loc)
          Y_loc_update = np.copy(self.Y_loc)
          GAMMA_update = np.copy(self.GAMMA)  
                       
        return Pts_after_Delete, X_loc_update, Y_loc_update, GAMMA_update       
        #return results      
                           
