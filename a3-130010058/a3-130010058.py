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

if __name__ == '__main__':
    x, e, T = boris(0.01, 200)
    plt.plot(x[:, 0], x[:, 1])
    plt.show()
    plt.plot(T, e)
    plt.show()
