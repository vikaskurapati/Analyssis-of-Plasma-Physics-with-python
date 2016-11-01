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
    X_Euler_Symplectic = [x]
    Y_Euler_Symplectic = [x]
    U_Euler_Symplectic = [u]
    V_Euler_Symplectic = [v]
    X_RK2 = [x]
    Y_RK2 = [x]
    U_RK2 = [u]
    V_RK2 = [v]
    X_Boris = [x]
    Y_Boris = [x]
    U_Boris = [u]
    V_Boris = [v]
    while t < tf:
        U_Euler.append(U_Euler[-1] + V_Euler[-1]*B)
        V_Euler.append(V_Euler[-1] - U_Euler[-1]*B)
        X_Euler.append(X_Euler[-1] + U_Euler[-2]*dt)
        Y_Euler.append(Y_Euler[-1] + V_Euler[-2]*dt)
        # Euler Scheme ends here
        U_Euler_Symplectic.append(U_Euler_Symplectic[-1] +
                                  V_Euler_Symplectic[-1]*B)
        V_Euler_Symplectic.append(V_Euler_Symplectic[-1] -
                                  U_Euler_Symplectic[-1]*B)
        X_Euler_Symplectic.append(X_Euler_Symplectic[-1] +
                                  U_Euler_Symplectic[-1]*dt)
        Y_Euler_Symplectic.append(Y_Euler_Symplectic[-1] +
                                  V_Euler_Symplectic[-1]*dt)
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
        # RK2 scheme ends here
        t = t + dt
    pass
