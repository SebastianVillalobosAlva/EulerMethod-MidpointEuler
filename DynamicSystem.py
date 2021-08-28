import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from time import time


if __name__ == '__main__':
    print('Dynamic System Module')
    
    

def euler_method(t, f_y_t, y0, vin):
    """ Euler Method

    Time evolves a dynamic system defined by its prior function, f(y,t), f_y_t, initial conditions y0 and excitation
    or control function vin.
    
    Parameters:
    -----------
    
        t (narray):
            time vector for the function to be evaluated
            
        f_y_t (callable):
            System dynamics function of the form func(tn, yn, dt)
            
                tn (float):
                    point in time
                yn (float):
                    previous state
                dt (float):
                    timestep
        y0 (tuple):
            Initial conditions
        vin (callable):
            Input of controlled function or independent variable function.
            It should be of the form func(t, *kargs)
            
        
    Returns:
    --------
        ndarray:
            System evolution in time of size (len(y), len(t))
        

"""
    
    y = np.zeros((len(y0), len(t)+1))
    dt = t[1]-t[0]
    print(y.shape)
    y[:,0] = y0
    

    
    for index, tn in enumerate(t):
        
        y[:,index+1] = dt * (f_y_t(tn, y[:,index], dt)) + y[:,index]
        
    return y[:,:len(t)]
    
    
    
def midpoint_euler_method(t, f_y_t, y0, vin):
    """ midpoint_euler_method

    Time evolves a dynamic system defined by its prior function, f(y,t), f_y_t, initial conditions y0 and excitation
    or control function vin.
    
    Parameters:
    -----------
    
        t (narray):
            time vector for the function to be evaluated
            
        f_y_t (callable):
            System dynamics function of the form func(tn, yn, dt)
            
                tn (float):
                    point in time
                yn (float):
                    previous state
                dt (float):
                    timestep
        y0 (tuple):
            Initial conditions
        vin (callable):
            Input of controlled function or independent variable function.
            It should be of the form func(t, *kargs)
            
        
    Returns:
    --------
        ndarray:
            System evolution in time of size (len(y), len(t))
        

"""
    
    y = np.zeros((len(y0), len(t)+1))
    dt = t[1]-t[0]
    print(y.shape)
    y[:,0] = y0
    
    dth = dt/2
    
    for index, tn in enumerate(t):
        th = tn + dth
        yh = y[:,index] + dth * f_y_t(tn, y[:,index], dt)
        
        y[:,index+1] = dt * (f_y_t(th, yh, dt)) + y[:,index]
    return y[:,:len(t)]
    
    
def signal_plot(t, y, **kwargs):
    """signal plot method

       Formats and plots the signal y under a dependent
       varianle t using the params in kwargs dictionary
       
       Parameters:
       -----------
       
           t (narray):
               time vector for the function to be evaluated
               
           y (ndarray):
               System dynamics signals
               
           kwargs (dict):
                {'figsize': [xsize, ysize],
                 'yc' : 'colour',
                 'tc' : 'colour',
                 'y_legends' : [labels],
                 'x_label' : [],
                 'vin' : callable fun(x)
                 }
        
           
    """


    fun = kwargs['vin']

    plt.figure(figsize=kwargs['figsize'])
    (plt.plot(t, fun(t), 'r', linewidth = 2, label = 'Input'),
    plt.plot(t, y[1].T, 'b', linewidth = 2, label = "Out "),
    plt.plot(t, y[0].T*0.2, 'orange', linewidth = 2, label = 'Change in S (Scaled 1 to 0.2)'),
    plt.xlabel('Time [s]'), plt.ylabel('Out [Adm]'),
    plt.title('Dynamic System Evolution'),
    plt.grid(), plt.legend(), plt.axis([0,np.max(t)*1.10, np.min(y*0.2)*1.1, np.max(y*0.2)*1.1]),
    plt.show())






def RK4_step(tm, um, dt, fun, coeff, s_vars):
 

 
     dt_2 = dt/2.0 # Calculating half of the step size
     

     # Calculating midpoint prediction
         
     th = tm + dt_2
     uK1 = fun(tm, um, coeff, s_vars)
     uh1 = um + dt_2 * uK1 # Half step point
     
     uK2 = fun(th, uh1, coeff, s_vars)
     uh2 = um + dt_2 * uK2 # Second quarter step
     
     uK3 = fun(th, uh2, coeff, s_vars)
     uh3 = um + dt * uK3 #
     
     uK4 = fun(tm + dt, uh3, coeff, s_vars)
     uh4 = um + dt * uK4
     
     
         
     # Calculating next step via the halfway prediction
         
     t = tm + dt
     u = um + dt*(uK1/6 + uK2/3 + uK3/3 + uK4/6)
         
     # Storing previous results for next time calculation

     return u



@contextmanager
def time_complexity(context_name, n):
    start = time()
    yield start
    end = time() - start
    print("Process name : <"+context_name+"> Execution Time: " + str(end) + "time units")
