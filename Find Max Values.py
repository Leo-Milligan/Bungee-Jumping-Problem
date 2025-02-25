import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt



# Time Step Parameters
t_step = 0.001 #time interval between solving differential equations
maxtime = 10 #how many seconds the animation runs for
time = np.arange(0,maxtime,t_step) #time intervals to simulate


#damped functions to solve
def solver_fall_d(Y,t,m,M,l,g,k,A,rho,c,f):
    return [Y[1], g + ( (1/2*m - l*rho*c*A) * Y[1]**2 ) / ( 2*l*M + m*(l-Y[0]) )]

def solver_elastic_d(Y,t,m,M,l,g,k,A,rho,c,f):
    return [Y[1], (6*M*g + 3*m*g - 6*k*(Y[0] - l ) - 3*rho*c*A*Y[1]**2 - 6*f*Y[1]) / (6*M + 2*m)]


#running bungee jump sim
def run_sim(m,M,l,g,k,A,rho,c,f):
    #keeps track of postion and velocity at each time step
    properties = np.zeros((len(time), 2))
    vel = [0]


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

    for i,a in enumerate(accel):
        accel[i] = a/g

    return stack,accel


def main():
    m = 0.059*3**2 #mass of rope in kg per meter
    M = 75 #mass of bunjee jumper
    l = 30 #length of rope
    g = 9.81 #gravity
    k_list = np.linspace(5,800,150) #spring constant of rope
    A = 0.28 #CSA of person
    rho = 1.205 #density of air
    c =   0  #1.2 #drag coeff
    f =   0  #9 #drag coeff caused by heating in the rope
    

    accel_list = np.zeros(len(k_list))
    pos_list = np.zeros(len(k_list))

    for i,k in enumerate(k_list):
        stack, accel = run_sim(m*l,M,l,g,k,A,rho,c,f)

        max_accel = max(abs(accel))
        max_position = max(stack[:,1])

        accel_list[i] = max_accel
        pos_list[i] = max_position/l
        


    uncomf_g = np.array([100,100])
    for i,a in enumerate(accel_list):
        if abs(a-3) < abs(uncomf_g[0]- 3):
            uncomf_g = [a,i]


    uncomf_stretch = np.array([100,100])
    for i,s in enumerate(pos_list):
        if abs(s-3) < abs(uncomf_stretch[0]- 3):
            uncomf_stretch = [s,i]



    accel_graph = True
    stretch_graph = False
    

    if accel_graph:    
        plt.plot(k_list,accel_list, color = 'black')
        plt.axvline(x=k_list[uncomf_g[1]],linestyle = 'dashed',color = 'blue', label = 'Force Limit (3G)')
        plt.axvline(x=k_list[uncomf_stretch[1]],linestyle = 'dashed', color = 'red', label = 'Stretch Limit (3x)')
        plt.ylabel('Maximum Acceleration (g)')
        plt.xlabel('k (N/m)')
        plt.minorticks_on()
        plt.legend()
        plt.savefig('Force_Graph.png')
        plt.show()
        

    if stretch_graph:    
        plt.plot(k_list,pos_list, color = 'black')
        plt.axvline(x=k_list[uncomf_g[1]],linestyle = 'dashed', color = 'blue', label = 'Force Limit (3G)')
        plt.axvline(x=k_list[uncomf_stretch[1]],linestyle = 'dashed', color = 'red', label = 'Stretch Limit (3x)')
        plt.ylabel('Maximum stretch (L)')
        plt.xlabel('k (N/m)')
        plt.minorticks_on()
        plt.legend()
        plt.savefig('Stretch_Graph.png')
        plt.show()
        
    


main()

