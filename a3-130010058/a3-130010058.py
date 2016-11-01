import numpy as np
from matplotlib import pyplot as plt


def q1():
    '''taking a unit mass with unit charge with initial velocity of 0i+1j
    under the influence of 1magnetic field starting at (1,0)'''
    m = 1.0
    B = 1.0
    q = 1.0
    T = 2*np.pi  # Time period for the following specifications is 2pi seconds.
    tf = 10*t
    dt = 0.05*T
    t = 0.0
    u = 0.0
    v = 1.0
    x = 1.0
    y = 0.0
    T = [t]
    X_Euler = [x]
    Y_Euler = [x]
    U_Euler = [u]
    V_Euler = [v]
    E_Euler = [0.5*m*(u**2 + v**2)]
    X_Euler_Symplectic = [x]
    Y_Euler_Symplectic = [x]
    U_Euler_Symplectic = [u]
    V_Euler_Symplectic = [v]
    E_Euler_Symplectic = [0.5*m*(u**2 + v**2)]
    X_RK2 = [x]
    Y_RK2 = [x]
    U_RK2 = [u]
    V_RK2 = [v]
    E_RK2 = [0.5*m*(u**2 + v**2)]
    X_Boris = [x]
    Y_Boris = [x]
    U_Boris = [u]
    V_Boris = [v]
    E_Boris = [0.5*m*(u**2 + v**2)]
    while t < tf:
        U_Euler.append(U_Euler[-1] + V_Euler[-1]*B)
        V_Euler.append(V_Euler[-1] - U_Euler[-1]*B)
        X_Euler.append(X_Euler[-1] + U_Euler[-2]*dt)
        Y_Euler.append(Y_Euler[-1] + V_Euler[-2]*dt)
        E_Euler.append(0.5*m*(U_Euler[-1]**2 + V_Euler[-1]**2))
        # Euler Scheme ends here
        U_Euler_Symplectic.append(U_Euler_Symplectic[-1] +
                                  V_Euler_Symplectic[-1]*B)
        V_Euler_Symplectic.append(V_Euler_Symplectic[-1] -
                                  U_Euler_Symplectic[-1]*B)
        X_Euler_Symplectic.append(X_Euler_Symplectic[-1] +
                                  U_Euler_Symplectic[-1]*dt)
        Y_Euler_Symplectic.append(Y_Euler_Symplectic[-1] +
                                  V_Euler_Symplectic[-1]*dt)
        E_Euler_Symplectic.append(0.5*m*(U_Euler_Symplectic[-1]**2 +
                                         V_Euler_Symplectic[-1]**2))
        # Euler Symplectic scheme ends here
        ku1 = V_RK2[-1]*B*dt
        kv1 = -U_RK2[-1]*B*dt
        kx1 = U_RK2[-1]*dt
        ky1 = V_RK2[-1]*dt
        ku2 = (V_RK2[-1] + ku1/2.)*B*dt
        kv2 = -(U_RK2[-1] + kv1/2.)*B*dt
        kx2 = dt*(ku1 + U_RK2[-1])
        ky2 = dt*(kv1 + V_RK2[-1])
        U_RK2.append(U_RK2[-1] + ku2)
        V_RK2.append(V_RK2[-1] + kv2)
        X_RK2.append(X_RK2[-1] + kx2)
        Y_RK2.append(Y_RK2[-1] + ky2)
        E_RK2.append(0.5*m*(U_RK2[-1]**2 + V_RK2[-1]**2))
        # RK2 scheme ends here
        uminus = U_Boris[-1]
        vminus = V_Boris[-1]
        alpha = q*B*dt/(2.0*m)
        uplus = ((1-alpha*alpha)*uminus + 2*alpha*vminus)/(alpha*alpha + 1.0)
        vplus = (vminus*(1 - alpha*alpha) - 2*alpha*uminus)/(1 + alpha*alpha)
        U_Boris.append(uplus)
        V_Boris.append(vplus)
        X_Boris.append(X_Boris[-1] + U_Boris[-1]*dt)
        Y_Boris.append(Y_Boris[-1] + V_Boris[-1]*dt)
        E_Boris.append(0.5*m*(U_Boris[-1]**2 + V_Boris[-1]**2))
        # Boris scheme ends here
        t = t + dt
    pass
