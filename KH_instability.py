"""
This script simulates KH instability using point vortex method

"""

import sys
import os
import numpy as np
from mpi4py import MPI
import scipy.io as sio
import time
from operators import PointVortexInsert, PointVortexDelete
from utility   import InterfaceCurvature, InitCirculation, InitVortexStrength, GeneralFunction, Clc_Circulation

def distance(X1,Y1, X2,Y2):
    return np.sqrt( (X2-X1)**2. + (Y2-Y1)**2. )
    

#def ArcLength(X, Y):
#    '''
#    arc length 
#    '''
#    X_diff = np.diff(X)
#    Y_diff = np.diff(Y)
#    return np.sqrt(X_diff**2. + Y_diff**2.)       
    
   
def Initialisation(N_vor, Ld, Hd,  U1, U2, At):    
    #N_vor = 300            # max no of point vortices    
    #Ld = 2.*np.pi          # length of the domain
    #Hd = 1.               # thickness of shear layer
    # shear layer profile
    #U1 = 0.2               # upper layer
    #U2 = 0.                # lower layer
    #Du = (U1-U2)/(2.*Hd)  # Vortex trength
    # density layer profile        
    #At = 0.05              # Atwood number 
    #r = (1.-At)/(1.+At)      # density ratio
    gra = 1.               # gravity    
    delta = 0.1
    alpha = 0.     
    # grids
    dx = Ld/N_vor
    threshold = 1.5
    Ampl = 1.e-2
    Xv = np.zeros(N_vor)
    Yv = np.zeros(N_vor)    
    Xv = np.linspace(0, Ld, N_vor)
    Yv = Ampl*np.sin(2.*np.pi*Xv)  
    
    return gra, delta, alpha, dx, threshold, Ampl, Xv, Yv
    
    
def simulation(N_vor, Ld, Hd, U1, U2, At, Iter):  
    dt = 0.002     # time step
    #Iter = 1.e5    # no of time steps
    
#    if Iter == 1:
    # Initialising the variables
    gra, delta, alpha, dx, threshold, Ampl, Xv, Yv = Initialisation(N_vor, Ld, Hd,  U1, U2, At)
        
    # creating array to handle point vortices adding or deletion
    GAMMA = np.zeros( (int(4*N_vor), Iter) )
    uIter = np.zeros( (int(4*N_vor), Iter) )
    vIter = np.zeros( (int(4*N_vor), Iter) ) 
    XIter = np.zeros( (int(4*N_vor), Iter) )
    YIter = np.zeros( (int(4*N_vor), Iter) )
    Kappa = np.zeros(  int(4*N_vor) )
    No_pt_vor = np.zeros( Iter )
    
    # calculating local circulation circulation
    Obj_0 = InitCirculation(Xv, Yv, At, N_vor, Ampl)  
    kappa = Obj_0.Kappa_0()
         
    #calculating vortex strength from local circulation 
    Obj_1 = InitVortexStrength(Xv, Yv, At, N_vor, Ampl, kappa)
    gamma = Obj_1.VortexStrength_0()   
   
    # Average velocities -> (u and v)
    Obj_2 = GeneralFunction(Xv, Yv, N_vor, delta, kappa)
    uiv = Obj_2.InterfaceVelocity()

    #sio.savemat('u0.mat', {'u0':uiv[:,0]}) 
    #sio.savemat('v0.mat', {'v0':uiv[:,1]})  
      
    Obj_3 = InterfaceCurvature(Xv, Yv, N_vor)
    Tvec_X, Tvec_Y, T_X, T_Y = Obj_3.UnitTangentVector()
    deltas = Obj_3.ArcLength()
     
    for ArrySize in range(N_vor):
      GAMMA[ArrySize,0] = gamma[ArrySize]    
      uIter[ArrySize,0] = uiv[ArrySize,0]
      vIter[ArrySize,0] = uiv[ArrySize,1]
      XIter[ArrySize,0] = Xv[ArrySize]
      YIter[ArrySize,0] = Yv[ArrySize] 
      No_pt_vor[0] =  N_vor
            
    time = 0    
    # !!!!! start time-stepping
    for Iter in range(1,Iter):
      #print (Iter)
      
      time += dt
      print('Iter= ', Iter,  'time= ', time) 
      #print('Iter= ', Iter) 
      
      GAMMA[0,Iter] = GAMMA[0,Iter-1] + 2.*At*gra*Tvec_Y[0]*dt
      tmpA = T_X[0]*( uIter[1,Iter-1] - uIter[N_vor-2,Iter-1] )/( deltas[1]+deltas[N_vor-1] )
      tmpB = T_Y[0]*( vIter[1,Iter-1] - vIter[N_vor-2,Iter-1] )/( deltas[1]+deltas[N_vor-1] )
      GAMMA[0,Iter] += - dt*GAMMA[0,Iter-1]*( tmpA + tmpB) 
      
      for ArrySze in range(1, N_vor-1):
        GAMMA[ArrySze,Iter] = GAMMA[ArrySze,Iter-1] + 2.*At*gra*Tvec_Y[ArrySze]*dt
        tmpA = T_X[ArrySze]*( uIter[ArrySze+1,Iter-1] - uIter[ArrySze-1,Iter-1] )/( 2.*deltas[ArrySze] )
        tmpB = T_Y[ArrySze]*( vIter[ArrySze+1,Iter-1] - vIter[ArrySze-1,Iter-1] )/( 2.*deltas[ArrySze] )
        GAMMA[ArrySze,Iter] += - dt*GAMMA[ArrySze,Iter-1]*( tmpA + tmpB )   
      
      GAMMA[N_vor-1,Iter] = GAMMA[0,Iter]
      
      Kappa[:] = 0.
      
      # calculating the circulation
      for ArrySze in range(N_vor):
        Kappa[ArrySze] = GAMMA[ArrySze,Iter]*deltas[ArrySze]
        
      #Kappa[N_vor-1] = Kappa[0]     
      
      # calculating ther interface velocity
      Obj_1 = Clc_Circulation(XIter[0:N_vor,Iter], YIter[0:N_vor,Iter], N_vor, delta, Kappa[0:N_vor])
      uiv = Obj_1.C_InterfaceVelocity()
      
      # updating the interface coordinates using Trapezoidal rule
      for ArrySze in range(N_vor-1):
        XIter[ArrySze,Iter] = XIter[ArrySze,Iter-1] + 0.5*dt*( uIter[ArrySze,Iter-1] + uiv[ArrySze,0] ) #+ 0.5*GAMMA[ArrySze,Iter-1]*alpha*T_X[ArrySze] ) 
        YIter[ArrySze,Iter] = YIter[ArrySze,Iter-1] + 0.5*dt*( vIter[ArrySze,Iter-1] + uiv[ArrySze,1] ) #+ 0.5*GAMMA[ArrySze,Iter-1]*alpha*T_Y[ArrySze] )
      
      XIter[N_vor-1,Iter] = XIter[0,Iter] + Ld
      YIter[N_vor-1,Iter] = YIter[0,Iter]
      
      #print('uIter[mid] = ',  uIter[int(0.4*N_vor),Iter-1])
           
      if Iter%10 == 0:
        
        # inserting point vortices 
        Obj_0 = PointVortexInsert(XIter[0:N_vor,Iter], YIter[0:N_vor,Iter], N_vor-1, dx, 1.2, GAMMA[0:N_vor,Iter])
        Total_V_Pts, Nw_V_Pts_Add, X_loc_update, Y_loc_update, GAMMA_update = Obj_0.InsertVortex()                
        
        # total no of point vortices updated
        N_vor = int(Total_V_Pts)
        #print(N_vor)        
        
        print('No of Point Vortices after adding = ', N_vor)
        
        #print('Iter = ', Iter)  
        #if Iter == 60:
          #sio.savemat('X_chk.mat', {'Xupdate':X_loc_update} )
          #sio.savemat('Y_chk.mat', {'Yupdate':Y_loc_update} )  
        
        # deleting point vortices        
        #Obj_1 = PointVortexDelete(X_loc_update, Y_loc_update, N_vor-1, dx, 0.6, GAMMA_update)
        #Total_V_Pts, X_loc_update1, Y_loc_update1, GAMMA_update1 = Obj_1.DeleteVortices()        
        
        # total no of point vortices updated
        #N_vor = int(Total_V_Pts)
        
        #print('No of Point Vortices after deleting = ', N_vor)
        
        #XIter[:,Iter] = 0.
        #YIter[:,Iter] = 0.
        #GAMMA[:,Iter] = 0.
        
        for ArrySize in range(N_vor):
          XIter[ArrySize,Iter] = X_loc_update[ArrySize]
          YIter[ArrySize,Iter] = Y_loc_update[ArrySize]
          GAMMA[ArrySize,Iter] = GAMMA_update[ArrySize] 
            
      else:
        N_vor = int(N_vor)      

      Obj_2 = InterfaceCurvature(XIter[0:N_vor,Iter], YIter[0:N_vor,Iter], N_vor )
      Tvec_X, Tvec_Y, T_X, T_Y = Obj_2.UnitTangentVector() 
      deltas = Obj_2.ArcLength()
     
      Kappa[:] = 0.
      # calculating new circulation
      for ArrySze in range(N_vor):
        Kappa[ArrySze] = GAMMA[ArrySze,Iter]*deltas[ArrySze]
        
      #if Iter==3:
        #sio.savemat('kappa_t.mat', {'Kappa':Kappa})  
      
      #Kappa[N_vor-1] = Kappa[0]     
      
      # calculating ther interface velocity
      Obj_1 = Clc_Circulation(XIter[0:N_vor,Iter], YIter[0:N_vor,Iter], N_vor, delta, Kappa[0:N_vor])
      uiv = Obj_1.C_InterfaceVelocity()
      
      #uI, vI = GeneralFunction.InterfaceVelocity(XIter[:,Iter], YIter[:,Iter], At, N_vor, delta, Ampl, alpha)
      for ArrySize in range(N_vor):
        uIter[ArrySize,Iter] = uiv[ArrySize,0]
        vIter[ArrySize,Iter] = uiv[ArrySize,1]
      
      #print('uIter[mid] = ',  uiv[int(0.4*N_vor),0])
      
      #sio.savemat('ut.mat', {'ut':uIter}) 
      #sio.savemat('vt.mat', {'vt':vIter})  
                            
      # calculating velocities above and below interfaces
      #Obj_4 = GeneralFunction(XIter[0:N_vor,Iter], YIter[0:N_vor,Iter], At, N_vor, delta, Ampl, alpha) 
      #U, V = Obj_4.Above_Below_InterfaceVelocity()
      
      No_pt_vor[Iter] =  N_vor
       
      sio.savemat('X_pos.mat', {'XIter':XIter}) #, {'YIter':YIter} )
      sio.savemat('Y_pos.mat', {'YIter':YIter}) #, {'VIter':vIter} ) 
      sio.savemat('N_vor.mat', {'N_vor':No_pt_vor})
 
if __name__ == '__main__':
    N_vor = 401            # max no of point vortices    
    Ld = 2.         # length of the domain
    Hd = 0.               # thickness of shear layer
    # shear layer profile
    U1 = 0.               # upper layer
    U2 = 0.                # lower layer
    #Du = (U1-U2)/(2.*Hd)  # Vortex trength
    # density layer profile        
    At = 0.05              # Atwood number    
    # time iteration
    Iter = 700   
    simulation(N_vor, Ld, Hd, U1, U2, At, Iter)
    
     

