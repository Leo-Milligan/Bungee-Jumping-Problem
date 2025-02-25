import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib

#sets whether there should be drag in the system 
drag = False
#



# Time Step Parameters
t_step = 0.01 #time interval between solving differential equations
maxtime = 100 #how many seconds the animation runs for
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


def main(drag):
    m = 0.059*3**2 #mass of rope in kg per meter
    M = 75 #mass of bunjee jumper
    l = 9 #length of rope
    g = 9.81 #gravity
    k = 100 #spring constant of rope
    A = 0.28 #CSA of person
    rho = 1.205 #density of air

    if drag:
        c =  1.2 #Air drag coeff
        f =  9 #drag coeff caused by heating in the rope
    else:
        c= 0
        f= 0

    
    stack, accel = run_sim(m*l,M,l,g,k,A,rho,c,f) 

    return stack, accel



stack, accel = main(drag)




#Plots Animation

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,12))
line1, = ax1.plot([], [], lw=4,c="blue",ls="-",ms=15,marker="s",mfc="gray",fillstyle="none",mec="black",markevery=2)
line2, = ax2.plot([], [], lw=1, color='r')
line3, = ax3.plot([], [], lw=1, color='g')
time_template = '\nTime = %.1fs'
time_text = ax1.text(0.1, 0.9, '', transform=ax1.transAxes)


ax1.set_ylim(1.2*min(stack[:,1]), 1.2*max(stack[:,1]))
ax1.invert_yaxis()
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')

ax2.set_ylim(-2, 1.2*max(stack[:,1]))
ax2.invert_yaxis()
ax2.set_xlim(0, maxtime)
ax2.set_xlabel('t (s)')
ax2.set_ylabel('y (m)')

ax3.set_xlim(0, maxtime)
ax3.set_ylim(1.2*min(accel[:]), 1.2*max(accel[:]))
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Acceleration (g)')

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    time_text.set_text('')
    return line1, line2, line3, time_text

def animate(i):
    line1.set_data([0,0], [stack[i,1],0])
    line2.set_data(time[:i], stack[:i,1])
    #line3.set_data(stack[:i,1], stack[:i,2])
    line3.set_data(time[:i], accel[:i]) #shows acceleration as a function of time
    time_text.set_text(time_template % (i*t_step))
    return line1, line2, line3, time_text  

ani = animation.FuncAnimation(fig, animate, frames = len(time),
                              interval=1, blit=True, init_func=init,repeat=False)


matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Users\\Leo Milligan\\Downloads\\ffmpeg-7.1-full_build\\bin\\ffmpeg.exe"
writer = animation.FFMpegFileWriter(fps = 15)

plt.show()

#ani.save('animation.mp4')

