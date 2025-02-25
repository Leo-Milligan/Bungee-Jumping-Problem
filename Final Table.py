import numpy as np
from scipy import integrate
import pandas as pd


#Main Parameters to change when making a new table
L_max = 60
M_list = np.linspace(50,90,5) #masses of bunjee jumpers
l_list = np.linspace(L_max/3,L_max/3 + 10,11) #lists of the lengths of rope



#
#
#



#damped functions to solve
def solver_fall_d(Y,t,m,M,l,g,k,A,rho,c,f):
    return [Y[1], g + ( (1/2*m - l*rho*c*A) * Y[1]**2 ) / ( 2*l*M + m*(l-Y[0]) )]

def solver_elastic_d(Y,t,m,M,l,g,k,A,rho,c,f):
    return [Y[1], (6*M*g + 3*m*g - 6*k*(Y[0] - l ) - 3*rho*c*A*Y[1]**2 - 6*f*Y[1]) / (6*M + 2*m)]


#running bungee jump sim
def run_sim(m,M,l,g,k,A,rho,c,f):

    # Time Step Parameters
    t_step = 0.01 #time interval between solving differential equations
    maxtime = 10 #how many seconds the animation runs for
    time = np.arange(0,maxtime,t_step) #time intervals to simulate

    #keeps track of postion and velocity at each time step
    properties = np.zeros((len(time), 2))
    vel = [0]

    #loops through time steps and solves diff equations
    for t in range(len(time)-1):
        #Falling regime
        if properties[t,0] <= l:
            sol = integrate.odeint(solver_fall_d, properties[t,:] , (t,t+t_step),(m,M,l,g,k,A,rho,c,f)) 
            properties[t+1] = sol[1]

        #Elastic regime
        if properties[t,0] > l:
            sol = integrate.odeint(solver_elastic_d, properties[t,:] , (t,t+t_step),(m,M,l,g,k,A,rho,c,f)) 
            properties[t+1] = sol[1]

        vel.append(sol[1,1])

    #adds time to first column of properties
    stack = np.c_[time, properties[:,0], properties[:, 1]]
    accel = np.gradient(vel,t_step)

    #unit change for acceleration to be in terms of number if Gs
    for i,a in enumerate(accel):
        accel[i] = a/g

    return stack,accel


def main(L_max, l_list, M_list): 
    
    #fine detain parameters
    k_list = np.linspace(1,1000,2000) #spring constant of rope
    g = 9.81 #gravity
    m = 0.059*3**2 #mass of rope in kg per meter
    A = 0.28 #CSA of person
    rho = 1.205 #density of air
    c =   0  #1.2 #drag coeff
    f =   0  #9 #drag coeff caused by heating in the rope
    
    saftey_margins = np.zeros([len(l_list), len(M_list)])
    

    for i,l in enumerate(l_list):
        print("Computing Values for L = " + str(l) + 'm')
        for j,M in enumerate(M_list):
            for k in k_list:
                #solves diff eqautions
                stack, accel = run_sim(m*l,M,l,g,k,A,rho,c,f)

                max_accel = max(abs(accel))
                max_position = max(stack[:,1])

                #checks that no saftey limits are breached and that the jump just reaches L_max
                if max_position <= L_max:
                    if max_accel <= 3:
                        saftey_margins[i,j] = round(k*2)/2 #rounds number to nearest 0.5
                        break
                    else:
                        saftey_margins[i,j] = None
                        break
                
                if k == k_list[-1]:
                    saftey_margins[i,j] = None
                    break

    #creates table
    df = pd.DataFrame(saftey_margins,index=l_list ,columns=M_list)
    df.to_csv('mass_length_table.csv', index=True)
    print('table Created!')
    return df


df = main(L_max, l_list, M_list)



