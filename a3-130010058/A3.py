import numpy as np
import matplotlib.pyplot as plt

def euler_explicit(charge, mass, e_field, b_field, v0, tstep, ntime): #only for uniform electric and magnetic fields
    v = np.zeros((ntime+1, 3))
    x = np.zeros((ntime + 1, 3))
    x[0] = np.array([0, 0, 0])
    v[0] = v0
    for i in range(1, ntime + 1):
        v[i] = v[i-1] + charge*(e_field + np.cross(v[i-1], b_field))/mass*tstep
        x[i] = x[i-1] + v[i-1]*tstep
    en = 0.5*mass*(v[:, 0]**2 + v[:,1]**2 + v[:,2]**2)
    return x, en

def euler_semi(charge, mass, e_field, b_field, v0, tstep, ntime): #only for uniform electric and magnetic fields
    v = np.zeros((ntime+1, 3))
    x = np.zeros((ntime + 1, 3))
    x[0] = [0, 0, 0]
    v[0] = v0
    for i in range(1, ntime + 1):
        v[i] = v[i-1] + charge*(e_field + np.cross(v[i-1], b_field))/mass*tstep
        x[i] = x[i-1] + v[i]*tstep
    en = 0.5*mass*(v[:, 0]**2 + v[:,1]**2 + v[:,2]**2)
    return x, en

def rk_2(charge, mass, e_field, b_field, v0, tstep, ntime): #only for uniform electric and magnetic fields
    v = np.zeros((ntime+1, 3))
    x = np.zeros((ntime + 1, 3))
    x[0] = [0, 0, 0]
    v[0] = v0
    for i in range(1, ntime + 1):
        kv1 = charge*(e_field + np.cross(v[i-1], b_field))/mass*tstep
        kx1 = v[i-1]*tstep
        kv2 = charge*(e_field + np.cross(v[i-1]+kv1/2, b_field))/mass*tstep
        kx2 = (v[i-1] + kv1)*tstep
        v[i] = v[i-1] + kv2
        x[i] = x[i-1] + kx2
    en = 0.5*mass*(v[:, 0]**2 + v[:,1]**2 + v[:,2]**2)
    return x, en

def boris_push(charge, mass, e_field, b0, gradB, v0, tstep, ntime):
    v = np.zeros((ntime+1, 3))
    x = np.zeros((ntime + 1, 3))
    x[0] = [0, 0, 0]
    v[0] = v0
    for i in range(1, ntime+1):
        vp = np.zeros(3)
        b = float (b0 + x[i-1, 0]*gradB)
        print b0, b, x[i-1], gradB
        alp = float (charge*b/(2*mass)*tstep)
        vm = v[i-1] + alp*e_field
        vp[0] = vm[0]*(1-alp**2)/(1+alp**2) + 2*alp/(1+alp**2)*vm[1]
        vp[1] = vm[1]*(1-alp**2)/(1+alp**2) - 2*alp/(1+alp**2)*vm[0]
        vp[2] = 0
        v[i] = vp + alp*e_field
        x[i] = x[i-1] + v[i]*tstep
    en = 0.5*mass*(v[:, 0]**2 + v[:,1]**2 + v[:,2]**2)
    return x, en


def expec(charge, mass, e_field, b0, v0, tstep, ntime):
    radius = mass*np.sum(v0**2)**0.5/(charge*b0[2])
    omega = charge*b0[2]/mass
    x = np.zeros((ntime + 1, 3))
    x[0] = [0, 0, 0]
    vd = np.cross(e_field, b0)/(np.sum(b0**2))
    for i in range(1, ntime + 1):
        x[i, 0] = radius*np.sin(omega*i*tstep) + vd[0]
        x[i, 1] = radius*np.cos(omega*i*tstep) - radius + vd[1]
    return x



def plotfig(name, title, xlabel, ylabel, xdata, ydata, legend, scale = "lin"):
    plt.figure()
    plt.title(title)
    if scale == "log":
        plt.xscale('log')
        plt.yscale('log')
    for i in range(len(xdata)):
        plt.plot(xdata[i], ydata[i], label = legend[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc = 2)
    plt.savefig(name)
    plt.close()

def plotq1(euler_en, euler_pos, euler_en2, euler_pos2, rk2en, rk2pos, bor_en, bor_pos, ntime, i):
    plotfig("Euler_exp_%g.png" %i, "Euler Explicit method", "x", "y", [euler_pos[:, 0]], [euler_pos[:,1]], ["Euler Explicit"])
    plotfig("Euler_exp_en_%g.png" %i, "Euler Explicit method", "time", "energy", [ntime], [euler_en], ["Euler Explicit"])
    plotfig("Euler_sem_%g.png" %i, "Euler Semi-implicit method", "x", "y", [euler_pos[:, 0]], [euler_pos[:,1]], ["Euler Semi-implicit"])
    plotfig("Euler_sem_en_%g.png" %i, "Euler Semi-implicit method", "time", "energy", [ntime], [euler_en2], ["Energy"])
    plotfig("RK2_pos_%g.png" %i, "Runge Kutta Scheme", "x", "y", [rk2pos[:, 0]], [rk2pos[:, 1]], ["position"])
    plotfig("RK2_en_%g.png" %i, "Runge Kutta method", "time", "energy", [ntime], [rk2en], ["Energy"])
    plotfig("Boris_pos_%g.png" %i, "Boris pusher pos", "x", "y", [bor_pos[:, 0]], [bor_pos[:, 1]], ["position"])
    plotfig("Boris_energy_%g.png" %i, "Boris pusher energy", "time", "energy", [ntime], [bor_en], ["energy"])

def ques1(tstep, tf, plot = 0, e_field = np.zeros(3)):
    charge = 5
    mass = 10
    b_field = np.array([0, 0, 1.0])
    v0 = np.array([10**4, 0, 0])
    time = int (tf/tstep)
    ntime = np.arange(0,time+1)
    exact_pos = expec(charge, mass, e_field, b_field, v0, tstep, time)

    euler_exp = euler_explicit(charge, mass, e_field, b_field, v0, tstep, time)
    euler_pos = euler_exp[0]
    euler_en = euler_exp[1]
    euler_error = np.sum((euler_pos[time] - exact_pos[time])**2)**0.5
    eu_en_err = euler_en[time] - 0.5*mass*10**8

    euler_sem = euler_semi(charge, mass, e_field, b_field, v0, tstep, time)
    euler_pos2 = euler_sem[0]
    euler_en2 = euler_sem[1]
    euler_error2 = np.sum((euler_pos2[time] - exact_pos[time])**2)**0.5
    eu_en_err2 = euler_en2[time] - 0.5*mass*10**8
    
    rk2 = rk_2(charge, mass, e_field, b_field, v0, tstep, time)
    rk2pos = rk2[0]
    rk2en = rk2[1]
    rk2_error = np.sum((rk2pos[time] - exact_pos[time])**2)**0.5
    rk2en_err = rk2en[time] - 0.5*mass*10**8
    
    boris = boris_push(charge, mass, e_field, b_field[2], 0, v0, tstep, time)
    bor_pos = boris[0]
    bor_en = boris[1]
    bor_error = np.sum((bor_pos[time] - exact_pos[time])**2)**0.5
    boren_err = bor_en[time] - 0.5*mass*10**8

    if plot == 1:
        plotq1(euler_en, euler_pos, euler_en2, euler_pos2, rk2en, rk2pos, bor_en, bor_pos, ntime, 1)

    if plot == 2:
        plotq1(euler_en, euler_pos, euler_en2, euler_pos2, rk2en, rk2pos, bor_en, bor_pos, ntime, 2)
    
    return [euler_error, eu_en_err, euler_error2, eu_en_err2, rk2_error, rk2en_err, bor_error, boren_err]

def ques2():
    tstep = [0.1, 0.05, 0.01, 0.005, 0.001]
    error = np.zeros((len(tstep), 8))
    for i in range(len(tstep)):
        error[i, :] = ques1(tstep[i], 200)
    plotfig("Pos_err.png", "Error in position v/s time step", "tstep", "error", [tstep, tstep, tstep, tstep], [error[:, 0], error[:, 2], error[:, 4], error[:, 6]], ["Explicit Euler", "Semi-implicit Euler", "Runge-Kutta", "Boris Pusher"], "log")

def ques3():
    ques1(0.01, 200, 2, np.array([10**5, 0, 0]))

def ques4():
    charge = 5
    mass = 10
    e_field = np.array([0, 0, 0])
    b_field = np.array([0, 0, 1.0])
    v0 = np.array([10**4, 0, 0])
    time = 20000
    tstep = 0.01
    ntime = np.arange(0,time+1)
    gradB = 0.0001
    boris = boris_push(charge, mass, e_field, b_field[2], gradB, v0, tstep, time)
    bor_pos = boris[0]
    plotfig("GradB_boris.png", "Grad B shift", "x", "y", [bor_pos[:, 0]], [bor_pos[:, 1]], ["position"])
    

#ques1(0.01, 200, 1)
#ques2()
#ques3()
ques4()

