import numpy as np
# import pymetabiosis.auto
from matplotlib import pyplot as plt
import copy


def q1(Dt, m=1.0, q=1.0, B=1.0, xi=-1.0 + 0.0j, ui=0.0 + 1.0j, plot=True):  # Dt is the dt scaling factor, dt = Dt*T
    T = 2.0*np.pi*m/(q*B)
    dt = Dt*T
    tf = 10*T
    x = [xi]  # initial position of the particle
    vi = [ui]  # initial velocity of the particle
    x_euler = copy.copy(x)
    v_euler = copy.copy(vi)
    x_euler_simplicit = copy.copy(x)  # simplicit means semi implicit
    v_euler_simplicit = copy.copy(vi)
    x_RK2 = copy.copy(x)
    v_RK2 = copy.copy(vi)
    x_Boris = copy.copy(x)
    v_Boris = copy.copy(vi)
    T = [0.0]
    # Denoting a vector with a complex number where real part is i value and
    # imaginary part is j value
    t = 0.0
    while t < tf:
        v_euler += [v_euler[-1]+(q*B*(v_euler[-1].imag - v_euler[-1].real*1.0j)*dt)/m]
        x_euler += [x_euler[-1] + v_euler[-2]*dt]
        v_euler_simplicit += [v_euler_simplicit[-1]+(q*B*(v_euler_simplicit[-1].imag - v_euler_simplicit[-1].real*1.0j)*dt)/m]
        x_euler_simplicit += [x_euler_simplicit[-1] + v_euler_simplicit[-1]*dt]
        kv1 = q*B*(v_RK2[-1].imag - v_RK2[-1].real*1.0j)*dt/m
        kx1 = v_RK2[-1]*dt
        v = v_RK2[-1] + kv1/2.0
        kv2 = dt*q*B*(v.imag - v.real*1.0j)/m
        kx2 = dt*(kv1 + v_RK2[-1])
        v_RK2 += [v_RK2[-1] + kv2]
        x_RK2 += [x_RK2[-1] + kx2]
        vminus = v_Boris[-1]
        alpha = 0.5*q*B*dt/m
        uplus = (1.0-alpha*alpha)*vminus.real/(1.0+alpha*alpha)+2.0*alpha*vminus.imag/(1.0+alpha*alpha)
        vplus = -2.0*alpha*vminus.real/(1.0+alpha*alpha)+(1.0-alpha*alpha)*vminus.imag/(1.0+alpha*alpha)
        Vplus = uplus + vplus*1.0j
        v_Boris += [Vplus]
        x_Boris += [x_Boris[-1] + v_Boris[-1]*dt]
        t = t + dt
        T += [t]
    v_euler = np.array(v_euler)
    x_euler = np.array(x_euler)
    v_euler_simplicit = np.array(v_euler_simplicit)
    E_euler = 0.5*m*abs(v_euler)*abs(v_euler)
    diss_euler = abs(E_euler - 0.5*m*vi[0]*vi[0])
    x_euler_simplicit = np.array(x_euler_simplicit)
    E_euler_simplicit = 0.5*m*abs(v_euler_simplicit)*abs(v_euler_simplicit)
    diss_euler_simplicit = abs(E_euler_simplicit - 0.5*m*vi[0]*vi[0])
    v_RK2 = np.array(v_RK2)
    x_RK2 = np.array(x_RK2)
    E_RK2 = 0.5*m*abs(v_RK2)*abs(v_RK2)
    diss_RK2 = abs(E_RK2 - 0.5*m*vi[0]*vi[0])
    v_Boris = np.array(v_Boris)
    x_Boris = np.array(x_Boris)
    E_Boris = 0.5*m*abs(v_Boris)*abs(v_Boris)
    diss_Boris = abs(E_Boris - 0.5*m*vi[0]*vi[0])
    if plot is True:
        plt.plot(x_euler.real, x_euler.imag)
        plt.title('Path of particle in Euler Method')
        plt.savefig('path_euler.png')
        plt.close()
        plt.plot(x_euler_simplicit.real, x_euler_simplicit.imag)
        plt.title('Path of particle in Euler semi implicit Method')
        plt.savefig('path_euler_simplicit.png')
        plt.close()
        plt.plot(x_RK2.real, x_RK2.imag)
        plt.title('Path of particle in RK2 Method')
        plt.savefig('path_RK2.png')
        plt.close()
        plt.plot(x_Boris.real, x_Boris.imag)
        plt.title('Path of particle in Boris Method')
        plt.savefig('path_Boris.png')
        plt.close()
        plt.plot(T, E_euler)
        plt.title('Energy of particle in Euler Method')
        plt.savefig('energy_euler.png')
        plt.close()
        plt.plot(T, E_euler_simplicit)
        plt.title('Energy of particle in Euler semi implicit Method')
        plt.savefig('energy_euler_simplicit.png')
        plt.close()
        plt.plot(T, E_RK2)
        plt.title('Energy of particle in RK2 Method')
        plt.savefig('energy_RK2.png')
        plt.close()
        plt.plot(T, E_Boris)
        plt.title('Energy of particle in Boris Method')
        plt.savefig('energy_Boris.png')
        plt.close()
        plt.plot(T, diss_euler)
        plt.title('Disspiation of energy of particle in Euler Method')
        plt.savefig('diss_euler.png')
        plt.close()
        plt.plot(T, diss_euler_simplicit)
        plt.title('Disspiation of energy of particle in Euler semi implicit Method')
        plt.savefig('diss_euler_simplicit.png')
        plt.close()
        plt.plot(T, diss_RK2)
        plt.title('Disspiation of energy of particle in RK2 Method')
        plt.savefig('diss_RK2.png')
        plt.close()
        plt.plot(T, diss_Boris)
        plt.title('Disspiation of energy of particle in Boris Method')
        plt.savefig('diss_Boris.png')
        plt.close()
        pass
    else:
        # As the particles is expected to return to the initial position as tf is multiple of T
        return abs(x_euler[-1] - x[0]),abs(x_euler_simplicit[-1] - x[0]),abs(x_RK2[-1] - x[0]),abs(x_Boris[-1] - x[0])


def q2(DT=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]):
    Err_Euler = np.zeros(len(DT))
    Err_Euler_simplicit = np.zeros(len(DT))
    Err_RK2 = np.zeros(len(DT))
    Err_Boris = np.zeros(len(DT))
    for i, Dt in enumerate(DT):
        Err_Euler[i], Err_Euler_simplicit[i], Err_RK2[i], Err_Boris[i] = q1(Dt, plot=False)
    plt.loglog(DT, Err_Euler)
    plt.title('Error in Euler scheme with time step factors')
    plt.savefig('q2_euler.png')
    plt.close()
    plt.loglog(DT, Err_Euler_simplicit)
    plt.title('Error in Euler semi implicit scheme with time step factors')
    plt.savefig('q2_euler_simplicit.png')
    plt.close()
    plt.loglog(DT, Err_RK2, label='RK2')
    plt.title('Error in RK2 scheme with time step factors')
    plt.savefig('q2_RK2.png')
    plt.close()
    plt.loglog(DT, Err_Boris, label='Boris')
    plt.title('Error in Boris scheme with time step factors')
    plt.savefig('q2_Boris.png')
    plt.close()


def q3(Dt, m=1.0, q=1.0, B=1.0, xi=-1.0 + 0.0j, ui=0.0 + 1.0j, E=1.0+0.0j):
    # Give Electric field in vector form as a complex number
    T = 2.0*np.pi*m/(q*B)
    dt = Dt*T
    tf = 10*T
    x = [xi]  # initial position of the particle
    vi = [ui]  # initial velocity of the particle
    x_euler = copy.copy(x)
    v_euler = copy.copy(vi)
    x_euler_simplicit = copy.copy(x)  # simplicit means semi implicit
    v_euler_simplicit = copy.copy(vi)
    x_RK2 = copy.copy(x)
    v_RK2 = copy.copy(vi)
    x_Boris = copy.copy(x)
    v_Boris = copy.copy(vi)
    T = [0.0]
    # Denoting a vector with a complex number where real part is i value and
    # imaginary part is j value
    t = 0.0
    while t < tf:
        v_euler += [v_euler[-1]+((q*B*(v_euler[-1].imag - v_euler[-1].real*1.0j)+q*E)*dt/m)]
        x_euler += [x_euler[-1] + v_euler[-2]*dt]
        v_euler_simplicit += [v_euler_simplicit[-1]+((q*B*(v_euler_simplicit[-1].imag - v_euler_simplicit[-1].real*1.0j)+q*E)*dt/m)]
        x_euler_simplicit += [x_euler_simplicit[-1] + v_euler_simplicit[-1]*dt]
        kv1 = (q*B*(v_RK2[-1].imag - v_RK2[-1].real*1.0j)+q*E)*dt/m
        kx1 = v_RK2[-1]*dt
        v = v_RK2[-1] + kv1/2.0
        kv2 = dt*(q*B*(v.imag - v.real*1.0j)+q*E)/m
        kx2 = dt*(kv1 + v_RK2[-1])
        v_RK2 += [v_RK2[-1] + kv2]
        x_RK2 += [x_RK2[-1] + kx2]
        vminus = v_Boris[-1] + q*dt*E*0.5/m
        alpha = 0.5*q*B*dt/m
        uplus = (1.0-alpha*alpha)*vminus.real/(1.0+alpha*alpha)+2.0*alpha*vminus.imag/(1.0+alpha*alpha)
        vplus = -2.0*alpha*vminus.real/(1.0+alpha*alpha)+(1.0-alpha*alpha)*vminus.imag/(1.0+alpha*alpha)
        Vplus = uplus + vplus*1.0j + q*dt*E*0.5/m
        v_Boris += [Vplus]
        x_Boris += [x_Boris[-1] + v_Boris[-1]*dt]
        t = t + dt
        T += [t]
    # v_drift =
    v_euler = np.array(v_euler)
    x_euler = np.array(x_euler)
    v_euler_simplicit = np.array(v_euler_simplicit)
    E_euler = 0.5*m*abs(v_euler)*abs(v_euler)
    diss_euler = abs(E_euler - 0.5*m*vi[0]*vi[0])
    x_euler_simplicit = np.array(x_euler_simplicit)
    E_euler_simplicit = 0.5*m*abs(v_euler_simplicit)*abs(v_euler_simplicit)
    diss_euler_simplicit = abs(E_euler_simplicit - 0.5*m*vi[0]*vi[0])
    v_RK2 = np.array(v_RK2)
    x_RK2 = np.array(x_RK2)
    E_RK2 = 0.5*m*abs(v_RK2)*abs(v_RK2)
    diss_RK2 = abs(E_RK2 - 0.5*m*vi[0]*vi[0])
    v_Boris = np.array(v_Boris)
    x_Boris = np.array(x_Boris)
    E_Boris = 0.5*m*abs(v_Boris)*abs(v_Boris)
    diss_Boris = abs(E_Boris - 0.5*m*vi[0]*vi[0])
    plt.plot(x_euler.real, x_euler.imag)
    plt.title('Path of particle in Euler Method with Electric Field')
    plt.savefig('q3_path_euler.png')
    plt.close()
    plt.plot(x_euler_simplicit.real, x_euler_simplicit.imag)
    plt.title('Path of particle in Euler semi implicit Method with Electric Field')
    plt.savefig('q3_path_euler_simplicit.png')
    plt.close()
    plt.plot(x_RK2.real, x_RK2.imag)
    plt.title('Path of particle in RK2 Method with Electric Field')
    plt.savefig('q3_path_RK2.png')
    plt.close()
    plt.plot(x_Boris.real, x_Boris.imag)
    plt.title('Path of particle in Boris Method with Electric Field')
    plt.savefig('q3_path_Boris.png')
    plt.close()


def q4(Dt, m=0.01, q=0.1, Bi=1.0, xi=-1.0 + 0.0j, ui=0.0 + 1.0j):
    # Give gradient of magnetic field in vector form as a complex number
    T = 2.0*np.pi*m/(q*Bi)
    dt = Dt*T
    tf = 10*T
    x = [xi]  # initial position of the particle
    vi = [ui]  # initial velocity of the particle
    x_euler = copy.copy(x)
    v_euler = copy.copy(vi)
    x_euler_simplicit = copy.copy(x)  # simplicit means semi implicit
    v_euler_simplicit = copy.copy(vi)
    x_RK2 = copy.copy(x)
    v_RK2 = copy.copy(vi)
    x_Boris = copy.copy(x)
    v_Boris = copy.copy(vi)
    T = [0.0]
    # Denoting a vector with a complex number where real part is i value and
    # imaginary part is j value
    t = 0.0

    def B(x, Bi=1.0, xi=-1.0+0.0j):
        # Assuming linear magentic field variation and along z axis.
        kx = 1.0  # Change these k values to change the variation in magnetic field.
        ky = 0.0
        return kx*(x-xi).real + ky*(x-xi).imag + Bi

    while t < tf:
        v_euler += [v_euler[-1]+((q*B(x_euler[-1])*(v_euler[-1].imag - v_euler[-1].real*1.0j))*dt/m)]
        x_euler += [x_euler[-1] + v_euler[-2]*dt]
        v_euler_simplicit += [v_euler_simplicit[-1]+((q*B(x_euler[-1])*(v_euler_simplicit[-1].imag - v_euler_simplicit[-1].real*1.0j))*dt/m)]
        x_euler_simplicit += [x_euler_simplicit[-1] + v_euler_simplicit[-1]*dt]
        kv1 = (q*B(x_RK2[-1])*(v_RK2[-1].imag - v_RK2[-1].real*1.0j))*dt/m
        kx1 = v_RK2[-1]*dt
        v = v_RK2[-1] + kv1/2.0
        kv2 = dt*(q*B(x_RK2[-1]+0.5*kx1)*(v.imag - v.real*1.0j))/m
        kx2 = dt*(kv1 + v_RK2[-1])
        v_RK2 += [v_RK2[-1] + kv2]
        x_RK2 += [x_RK2[-1] + kx2]
        vminus = v_Boris[-1]
        alpha = 0.5*q*B(x_Boris[-1])*dt/m
        uplus = (1.0-alpha*alpha)*vminus.real/(1.0+alpha*alpha)+2.0*alpha*vminus.imag/(1.0+alpha*alpha)
        vplus = -2.0*alpha*vminus.real/(1.0+alpha*alpha)+(1.0-alpha*alpha)*vminus.imag/(1.0+alpha*alpha)
        Vplus = uplus + vplus*1.0j
        v_Boris += [Vplus]
        x_Boris += [x_Boris[-1] + v_Boris[-1]*dt]
        t = t + dt
        T += [t]
    # v_drift =
    v_euler = np.array(v_euler)
    x_euler = np.array(x_euler)
    v_euler_simplicit = np.array(v_euler_simplicit)
    E_euler = 0.5*m*abs(v_euler)*abs(v_euler)
    diss_euler = abs(E_euler - 0.5*m*vi[0]*vi[0])
    x_euler_simplicit = np.array(x_euler_simplicit)
    E_euler_simplicit = 0.5*m*abs(v_euler_simplicit)*abs(v_euler_simplicit)
    diss_euler_simplicit = abs(E_euler_simplicit - 0.5*m*vi[0]*vi[0])
    v_RK2 = np.array(v_RK2)
    x_RK2 = np.array(x_RK2)
    E_RK2 = 0.5*m*abs(v_RK2)*abs(v_RK2)
    diss_RK2 = abs(E_RK2 - 0.5*m*vi[0]*vi[0])
    v_Boris = np.array(v_Boris)
    x_Boris = np.array(x_Boris)
    E_Boris = 0.5*m*abs(v_Boris)*abs(v_Boris)
    diss_Boris = abs(E_Boris - 0.5*m*vi[0]*vi[0])
    plt.plot(x_euler.real, x_euler.imag)
    plt.title('Path of particle in Euler Method with gradient Magnetic Field')
    plt.savefig('q4_path_euler.png')
    plt.close()
    plt.plot(x_euler_simplicit.real, x_euler_simplicit.imag)
    plt.title('Path of particle in Euler semi implicit Method with gradient magnetic Field')
    plt.savefig('q4_path_euler_simplicit.png')
    plt.close()
    plt.plot(x_RK2.real, x_RK2.imag)
    plt.title('Path of particle in RK2 Method with gradient magnetic Field')
    plt.savefig('q4_path_RK2.png')
    plt.close()
    plt.plot(x_Boris.real, x_Boris.imag)
    plt.title('Path of particle in Boris Method with gradient magnetic Field')
    plt.savefig('q4_path_Boris.png')
    plt.close()


if __name__ == '__main__':
    # q1(0.05)
    # q2(np.linspace(0.01, 0.1, 100))
    # q3(0.05)
    q4(0.05)
