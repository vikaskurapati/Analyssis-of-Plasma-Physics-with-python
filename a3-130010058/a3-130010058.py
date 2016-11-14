import numpy as np
from matplotlib import pyplot as plt
# In the whole assigment gradB is considered only in x direction for
# convenience. If you need it in any other direction, just rotate the axis


def euler(dt, tf, q=5.0, m=10.0, E=np.array([0.0, 0.0, 0.0]), Bi=np.array([0.0, 0.0, 1.0]), gradB=0.0, ui=np.array([10**4, 0.0, 0.0]), x0=np.array([0.0, 0.0, 0.0])):
    ntime = int(tf/dt)
    v_euler = np.zeros((ntime+1, 3))
    x_euler = np.zeros((ntime + 1, 3))
    x_euler[0] = x0
    v_euler[0] = ui
    B = Bi.copy()
    T = np.linspace(0.0, tf, ntime+1)
    for i in range(1, ntime + 1):
        B[2] = Bi[2] + (x_euler[i-1, 0] - x0[0])*gradB
        v_euler[i] = v_euler[i-1] + q*(E + np.cross(v_euler[i-1], B))/m*dt
        x_euler[i] = x_euler[i-1] + v_euler[i-1]*dt
    e_euler = 0.5*m*(v_euler[:, 0]**2 + v_euler[:, 1]**2 + v_euler[:, 2]**2)
    return x_euler, e_euler, T


def euler2(dt, tf, q=5.0, m=10.0, E=np.array([0.0, 0.0, 0.0]), Bi=np.array([0.0, 0.0, 1.0]), gradB=0.0, ui=np.array([10**4, 0.0, 0.0]), x0=np.array([0.0, 0.0, 0.0])):
    ntime = int(tf/dt)
    v_euler2 = np.zeros((ntime+1, 3))
    x_euler2 = np.zeros((ntime + 1, 3))
    x_euler2[0] = x0
    v_euler2[0] = ui
    T = np.linspace(0.0, tf, ntime+1)
    B = Bi.copy()
    for i in range(1, ntime + 1):
        B[2] = Bi[2] + (x_euler2[i-1, 0] - x0[0])*gradB
        v_euler2[i] = v_euler2[i-1]+q*(E + np.cross(v_euler2[i-1], B))/m*dt
        x_euler2[i] = x_euler2[i-1] + v_euler2[i]*dt
    e_euler2 = 0.5*m*(v_euler2[:, 0]**2+v_euler2[:, 1]**2+v_euler2[:, 2]**2)
    return x_euler2, e_euler2, T


def RK2(dt, tf, q=5.0, m=10.0, E=np.array([0.0, 0.0, 0.0]), Bi=np.array([0.0, 0.0, 1.0]), gradB=0.0, ui=np.array([10**4, 0.0, 0.0]), x0=np.array([0.0, 0.0, 0.0])):
    ntime = int(tf/dt)
    v_RK2 = np.zeros((ntime+1, 3))
    x_RK2 = np.zeros((ntime + 1, 3))
    x_RK2[0] = x0
    v_RK2[0] = ui
    T = np.linspace(0.0, tf, ntime+1)
    B = Bi.copy()
    for i in range(1, ntime + 1):
        B[2] = Bi[2] + (x_RK2[i-1, 0] - x0[0])*gradB
        kv1 = q*(E + np.cross(v_RK2[i-1], B))/m*dt
        kx1 = v_RK2[i-1]*dt
        kv2 = q*(E + np.cross(v_RK2[i-1]+kv1/2.0, B))/m*dt
        kx2 = (v_RK2[i-1] + kv1)*dt
        v_RK2[i] = v_RK2[i-1] + kv2
        x_RK2[i] = x_RK2[i-1] + kx2
    e_RK2 = 0.5*m*(v_RK2[:, 0]**2 + v_RK2[:, 1]**2 + v_RK2[:, 2]**2)
    return x_RK2, e_RK2, T


def boris(dt, tf, q=5.0, m=10.0, E=np.array([0.0, 0.0, 0.0]), Bi=np.array([0.0, 0.0, 1.0]), gradB=0.0, ui=np.array([10**4, 0.0, 0.0]), x0=np.array([0.0, 0.0, 0.0])):
    ntime = int(tf/dt)
    v_boris = np.zeros((ntime+1, 3))
    x_boris = np.zeros((ntime + 1, 3))
    x_boris[0] = x0
    v_boris[0] = ui
    T = np.linspace(0.0, tf, ntime+1)
    B = Bi.copy()
    for i in range(1, ntime+1):
        vplus = np.zeros(3)
        B[2] = Bi[2] + (x_boris[i-1, 0] - x0[0])*gradB
        alpha = 0.5*q*B[2]*dt/m
        vminus = v_boris[i-1] + 0.5*q*E/m
        vplus[0] = vminus[0]*(1-alpha**2)/(1+alpha**2) + 2*alpha/(1+alpha**2)*vminus[1]
        vplus[1] = vminus[1]*(1-alpha**2)/(1+alpha**2) - 2*alpha/(1+alpha**2)*vminus[0]
        v_boris[i] = vplus + 0.5*q*E/m
        x_boris[i] = x_boris[i-1] + v_boris[i]*dt
    e_boris = 0.5*m*(v_boris[:, 0]**2 + v_boris[:, 1]**2 + v_boris[:, 2]**2)
    return x_boris, e_boris, T


def expec(dt, tf, q=5.0, m=10.0, E=np.array([0.0, 0.0, 0.0]), B0=np.array([0.0, 0.0, 1.0]), gradB=np.array([0.0,0.0,0.0]), ui=np.array([10**4, 0.0, 0.0]), x0=np.array([0.0, 0.0, 0.0])):
    ntime = int(tf/dt)
    r = m*np.sqrt(np.sum(ui*ui))/(q*B0[2])
    w = q*B0[2]/m
    x = np.zeros((ntime + 1, 3))
    x[0] = x0
    vd = np.cross(E, B0)/(np.sum(B0*B0)) + 0.5*m*np.sum(ui**2)/(q*np.sum(B0**2)**1.5)*np.cross(B0, gradB)
    for i in range(1, ntime + 1):
        x[i, 0] = r*np.sin(w*i*dt) + vd[0] + x[0, 0]
        x[i, 1] = r*np.cos(w*i*dt) - r + vd[1] + x[0, 1]
    return x


def q1(dt, plot=True):
    x_euler, e_euler, T = euler(dt, 200)
    x_euler2, e_euler2, T = euler2(dt, 200)
    x_RK2, e_RK2, T = RK2(dt, 200)
    x_boris, e_boris, T = boris(dt, 200)
    x_expec = expec(dt, 200)
    m = 10.0
    u = 10**4
    e_exac = 0.5*m*u*u
    if plot is True:
        plt.plot(x_euler[:, 0], x_euler[:, 1])
        plt.title('Path of particle using Euler Scheme')
        plt.savefig('q1_path_euler.png')
        plt.close()
        plt.plot(T, e_euler)
        plt.title('Energy of particle using Euler Scheme')
        plt.savefig('q1_energy_euler.png')
        plt.close()
        plt.plot(T, e_euler-e_exac)
        plt.title('Dissipation of particle using Euler Scheme')
        plt.savefig('q1_dissipation_euler.png')
        plt.close()
        plt.plot(T, np.sqrt(((x_euler-x_expec)[:, 0]**2+(x_euler-x_expec)[:, 1]**2)))
        plt.title('Error in Euler Scheme')
        plt.savefig('q1_error_euler.png')
        plt.close()
        plt.plot(x_euler2[:, 0], x_euler2[:, 1])
        plt.title('Path of particle using Semi Implicit Euler Scheme')
        plt.savefig('q1_path_euler2.png')
        plt.close()
        plt.plot(T, e_euler2)
        plt.title('Energy of particle using Semi Implicit Euler Scheme')
        plt.savefig('q1_energy_euler2.png')
        plt.close()
        plt.plot(T, e_euler2-e_exac)
        plt.title('Dissipation of particle using Semi Implicit Euler Scheme')
        plt.savefig('q1_dissipation_euler2.png')
        plt.close()
        plt.plot(T, np.sqrt((x_euler2-x_expec)[:, 0]**2+(x_euler2-x_expec)[:, 1]**2))
        plt.title('Error in Semi Implicit Euler Scheme')
        plt.savefig('q1_error_euler2.png')
        plt.close()
        plt.plot(x_RK2[:, 0], x_RK2[:, 1])
        plt.title('Path of particle using RK2 Scheme')
        plt.savefig('q1_path_RK2.png')
        plt.close()
        plt.plot(T, e_RK2)
        plt.title('Energy of particle using RK2 Scheme')
        plt.savefig('q1_energy_RK2.png')
        plt.close()
        plt.plot(T, e_RK2-e_exac)
        plt.title('Dissipation of particle using RK2 Scheme')
        plt.savefig('q1_dissipation_RK2.png')
        plt.close()
        plt.plot(T, np.sqrt((x_RK2-x_expec)[:, 0]**2+(x_RK2-x_expec)[:, 1]**2))
        plt.title('Error in RK2 Scheme')
        plt.savefig('q1_error_RK2.png')
        plt.close()
        plt.plot(x_boris[:, 0], x_boris[:, 1])
        plt.title('Path of particle using Boris Scheme')
        plt.savefig('q1_path_boris.png')
        plt.close()
        plt.plot(T, e_boris)
        plt.title('Energy of particle using Boris Scheme')
        plt.savefig('q1_energy_boris.png')
        plt.close()
        plt.plot(T, e_boris-e_exac)
        plt.title('Dissipation of particle using Boris Scheme')
        plt.savefig('q1_dissipation_boris.png')
        plt.close()
        plt.plot(T, np.sqrt((x_boris-x_expec)[:, 0]**2+(x_boris-x_expec)[:, 1]**2))
        plt.title('Error in Boris Scheme')
        plt.savefig('q1_error_boris.png')
        plt.close()
        pass
    else:
        euler_error = np.sum((x_euler[-1] - x_expec[-1])**2)**0.5/len(x_expec)
        euler2_error = np.sum((x_euler2[-1] - x_expec[-1])**2)**0.5/len(x_expec)
        RK2_error = np.sum((x_RK2[-1] - x_expec[-1])**2)**0.5/len(x_expec)
        boris_error = np.sum((x_boris[-1] - x_expec[-1])**2)**0.5/len(x_expec)
        return euler_error, euler2_error, RK2_error, boris_error


def q2():
    DT = [0.1, 0.05, 0.01, 0.005, 0.001]
    euler = np.zeros(len(DT))
    euler2 = np.zeros(len(DT))
    RK2 = np.zeros(len(DT))
    boris = np.zeros(len(DT))
    for i, dt in enumerate(DT):
        euler[i], euler2[i], RK2[i], boris[i] = q1(dt, plot=False)
    plt.loglog(DT, euler, label='Euler')
    plt.loglog(DT, euler2, label="Semi-Implicit Euler")
    plt.loglog(DT, RK2, label='RK2')
    plt.loglog(DT, boris, label='Boris')
    plt.legend()
    plt.title('Error with time step')
    plt.savefig('q2_Error.png')
    plt.close()


def q3(dt, plot=True):
    x_euler, e_euler, T = euler(dt, 200, E=np.array([10**5, 0.0, 0.0]))
    x_euler2, e_euler2, T = euler2(dt, 200, E=np.array([10**5, 0.0, 0.0]))
    x_RK2, e_RK2, T = RK2(dt, 200, E=np.array([10**5, 0.0, 0.0]))
    x_boris, e_boris, T = boris(dt, 200, E=np.array([10**5, 0.0, 0.0]))
    x_expec = expec(dt, 200, E=np.array([10**5, 0.0, 0.0]))
    plt.plot(x_euler[:, 0], x_euler[:, 1])
    plt.title('Path of particle using Euler Scheme')
    plt.savefig('q3_euler.png')
    plt.close()
    plt.plot(x_euler2[:, 0], x_euler2[:, 1])
    plt.title('Path of particle using Semi-Implicit Euler Scheme')
    plt.savefig('q3_euler2.png')
    plt.close()
    plt.plot(x_RK2[:, 0], x_RK2[:, 1])
    plt.title('Path of particle using RK2 Scheme')
    plt.savefig('q3_RK2.png')
    plt.close()
    plt.plot(x_boris[:, 0], x_boris[:, 1])
    plt.title('Path of particle using boris Scheme')
    plt.savefig('q3_boris.png')
    plt.close()
    plt.plot(T, np.sqrt((x_euler-x_expec)[:, 0]**2+(x_euler-x_expec)[:, 1]**2))
    plt.title('Error in Euler Scheme')
    plt.savefig('q3_error_euler.png')
    plt.close()
    plt.plot(T, np.sqrt((x_euler2-x_expec)[:, 0]**2+(x_euler2-x_expec)[:, 1]**2))
    plt.title('Error in Semi-Implicit Euler Scheme')
    plt.savefig('q3_error_euler2.png')
    plt.close()
    plt.plot(T, np.sqrt((x_RK2-x_expec)[:, 0]**2+(x_RK2-x_expec)[:, 1]**2))
    plt.title('Error in RK2 Scheme')
    plt.savefig('q3_error_RK2.png')
    plt.close()
    plt.plot(T, np.sqrt((x_boris-x_expec)[:, 0]**2+(x_boris-x_expec)[:, 1]**2))
    plt.title('Error in Boris Scheme')
    plt.savefig('q3_error_boris.png')
    plt.close()


def q4(dt, plot=True):
    x_euler, e_euler, T = euler(dt, 200, gradB=0.0001)
    x_euler2, e_euler2, T = euler2(dt, 200, gradB=0.0001)
    x_RK2, e_RK2, T = RK2(dt, 200, gradB=0.0001)
    x_boris, e_boris, T = boris(dt, 200, gradB=0.0001)
    x_expec = expec(dt, 200, gradB=np.array([0.0001, 0.0, 0.0]))
    plt.plot(x_euler[:, 0], x_euler[:, 1])
    plt.title('Path of particle using Euler Scheme')
    plt.savefig('q4_euler.png')
    plt.close()
    plt.plot(x_euler2[:, 0], x_euler2[:, 1])
    plt.title('Path of particle using Semi-Implicit Euler Scheme')
    plt.savefig('q4_euler2.png')
    plt.close()
    plt.plot(x_RK2[:, 0], x_RK2[:, 1])
    plt.title('Path of particle using RK2 Scheme')
    plt.savefig('q4_RK2.png')
    plt.close()
    plt.plot(x_boris[:, 0], x_boris[:, 1])
    plt.title('Path of particle using boris Scheme')
    plt.savefig('q4_boris.png')
    plt.close()
    plt.plot(T, np.sqrt((x_euler-x_expec)[:, 0]**2+(x_euler-x_expec)[:, 1]**2))
    plt.title('Error in Euler Scheme')
    plt.savefig('q4_error_Euler.png')
    plt.close()
    plt.plot(T, np.sqrt((x_euler2-x_expec)[:, 0]**2+(x_euler2-x_expec)[:, 1]**2))
    plt.title('Error in Semi-Implicit Euler Scheme')
    plt.savefig('q4_error_euler2.png')
    plt.close()
    plt.plot(T, np.sqrt((x_RK2-x_expec)[:, 0]**2+(x_RK2-x_expec)[:, 1]**2))
    plt.title('Error in RK2 Scheme')
    plt.savefig('q4_error_RK2.png')
    plt.close()
    plt.plot(T, np.sqrt((x_boris-x_expec)[:, 0]**2+(x_boris-x_expec)[:, 1]**2))
    plt.title('Error in Boris Scheme')
    plt.savefig('q4_error_boris.png')
    plt.close()

if __name__ == '__main__':
    q1(0.01)
    q2()
    q3(0.01)
    q4(0.01)
