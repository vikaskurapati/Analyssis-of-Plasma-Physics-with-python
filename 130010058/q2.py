import numpy as np
from matplotlib import pyplot as plt
k = 1.38064852e-23


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
    nu = 0.5*m*(v_boris[:, 0]**2 + v_boris[:, 1]**2)/(Bi[2])
    return x_boris, e_boris, nu, T


def borisE(dt, tf, q=5.0, m=10.0, Bi=np.array([0.0, 0.0, 1.0]), gradB=0.0, ui=np.array([10**4, 0.0, 0.0]), x0=np.array([0.0, 0.0, 0.0])):
    ntime = int(tf/dt)
    E = np.zeros((ntime+1, 3))
    v_boris = np.zeros((ntime+1, 3))
    x_boris = np.zeros((ntime + 1, 3))
    x_boris[0] = x0
    v_boris[0] = ui
    T = np.linspace(0.0, tf, ntime+1)
    B = Bi.copy()
    for i in range(1, ntime+1):
        E[i, 1] = 10**5*np.sin(np.pi*i/10.0)
        vplus = np.zeros(3)
        B[2] = Bi[2] + (x_boris[i-1, 0] - x0[0])*gradB
        alpha = 0.5*q*B[2]*dt/m
        vminus = v_boris[i-1] + 0.5*q*E[i]/m
        vplus[0] = vminus[0]*(1-alpha**2)/(1+alpha**2) + 2*alpha/(1+alpha**2)*vminus[1]
        vplus[1] = vminus[1]*(1-alpha**2)/(1+alpha**2) - 2*alpha/(1+alpha**2)*vminus[0]
        v_boris[i] = vplus + 0.5*q*E[i]/m
        x_boris[i] = x_boris[i-1] + v_boris[i]*dt
    e_boris = 0.5*m*(v_boris[:, 0]**2 + v_boris[:, 1]**2 + v_boris[:, 2]**2)
    nu = 0.5*m*(v_boris[:, 0]**2 + v_boris[:, 1]**2)/(Bi[2])
    return x_boris, e_boris, nu, v_boris, T


def expecE(dt, tf, q=5.0, m=10.0, B0=np.array([0.0, 0.0, 1.0]), ui=np.array([10**4, 0.0, 0.0]), x0=np.array([0.0, 0.0, 0.0])):
    ntime = int(tf/dt)
    E = np.zeros((ntime+1, 3))
    r = m*np.sqrt(np.sum(ui*ui))/(q*B0[2])
    w = q*B0[2]/m
    x = np.zeros((ntime + 1, 3))
    x[0] = x0
    for i in range(1, ntime + 1):
        E[i, 1] = 10**5*np.sin(np.pi*i/10.0)
        vd = np.cross(E[i], B0)/(np.sum(B0*B0))
        x[i, 0] = r*np.sin(w*i*dt) + vd[0]*i*dt + x[0, 0]
        x[i, 1] = r*np.cos(w*i*dt) - r + vd[1]*i*dt + x[0, 1]
    return x


def euler(dt, tf, q=5.0, m=10.0, E=np.array([0.0, 10**5, 0.0]), Bi=np.array([0.0, 0.0, 0.0]), gradB=0.0, ui=np.array([10**4, 0.0, 0.0]), x0=np.array([0.0, 0.0, 0.0])):
    ntime = int(tf/dt)
    v_euler = np.zeros((ntime+1, 3))
    x_euler = np.zeros((ntime + 1, 3))
    x_euler[0] = x0
    v_euler[0] = ui
    B = Bi.copy()
    T = np.linspace(0.0, tf, ntime+1)
    for i in range(1, ntime + 1):
        B[2] = Bi[2] + (x_euler[i-1, 0] - x0[0])*gradB
        v_euler[i] = v_euler[i-1] + (1.0/m)*dt*(q*E - v_euler[i-1]*4.8*10**(-14)*15*(m*np.sum(v_euler[i-1]**2)*10**20*8.61793324e-5/(3*k))**(-1.5))
        x_euler[i] = x_euler[i-1] + v_euler[i-1]*dt
    e_euler = 0.5*m*(v_euler[:, 0]**2 + v_euler[:, 1]**2 + v_euler[:, 2]**2)
    return x_euler, e_euler, T
q2a1_x, q2a1_e, q2a1_nu, T = boris(0.01, 100.0)
plt.plot(T, q2a1_nu)
plt.title('Variation of magnetic moment with time')
plt.savefig('q2a1.png')
plt.close()
q2a2_x, q2a2_e, q2a2_nu, T = boris(0.01, 100.0, E=np.array([0.0, 10**5,  0.0]))
plt.plot(T, q2a1_nu)
plt.title('Variation of magnetic moment with time')
plt.savefig('q2a2.png')
plt.close()
q2b_x, q2b_e, q2b_nu, q2b_v, T = borisE(0.1, 100.0)
q2b_exact = expecE(0.1, 100.0)
plt.plot(q2b_x[:, 0], q2b_x[:, 1], label='Boris')
plt.title('Path of the particle using Boris')
plt.savefig('q2b_path.png')
plt.close()
plt.plot(q2b_exact[:, 0], q2b_exact[:, 1], label='Exact')
plt.title('path of the particle using drift velocity')
plt.savefig('q2b_exact.png')
plt.close()
plt.plot(T, np.sqrt((q2b_v[:, 0]**2 + q2b_v[:, 1]**2 + q2b_v[:, 2]**2)))
plt.title('Velocity with time')
plt.savefig('q2b_vel.png')
plt.close()
x_euler, e_euler, T = euler(0.01, 100.0)
plt.plot(x_euler[:, 0], x_euler[:, 1])
plt.savefig('q2c_path.png')
plt.close()
plt.plot(T, np.sqrt(2*e_euler/10.0))
plt.savefig('q2c_vel.png')
plt.close()
