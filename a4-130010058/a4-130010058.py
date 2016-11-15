import numpy as np
from matplotlib import pyplot as plt
import copy

k = 1.38064e-23
e = 1.60271e-19
eps = 8.85419e-12
m_p = 1.67262e-27


class beam:
    '''Temperature is taken in electron volts for convinience'''
    def __init__(self, v, rho, t, mw=1.0):
        self.rho = rho
        self.v = v
        self.t = t*e/k
        self.mw = mw*m_p

    def get_beam(self):
        return self.rho, self.v, self.t, self.mw

    def v_dist(self, v):
        return self.rho*(self.mw/(2.0*np.pi*k*self.t))**0.5*np.exp(-self.mw*(v-self.v)**2.0/(2.0*k*self.t))

    def thermal_v(self):
        return (k*self.t/self.mw)**0.5

    def freq(self):
        return self.rho*e**4.0/(16.0*np.pi*eps**2.0*self.mw**2.0*self.thermal_v()**3.0)


def df(f, dx):
    df = np.zeros(len(f))
    df[0] = (f[1] - f[0])/dx  # forward difference for first element.
    for i in range(1, len(f)-1):
        df[i] = (f[i+1] - f[i-1])*0.5/dx
    df[len(f)-1] = (f[len(f)-1] - f[len(f)-2])/dx
    return df


def entropy(v, pdf):
    s = 0
    for i in range(len(v)):
        if(pdf[i] != 0.0):
            s += -1*pdf[i]*np.log(pdf[i])*(v[i] - v[i-1])
    return s


def q1(beam_matrix):
    fmax = 0.0
    for beam in beam_matrix:
        if fmax < beam.freq():
            fmax = beam.freq()
    return fmax


def q2(beam_matrix):
    v_min = beam_matrix[0].get_beam()[1] - 3.0*beam_matrix[0].thermal_v()
    v_max = beam_matrix[0].get_beam()[1] + 3.0*beam_matrix[0].thermal_v()
    for beam in beam_matrix:
        if v_min > beam.get_beam()[1] - 3.0*beam.thermal_v():
            v_min = beam.get_beam()[1] - 3.0*beam.thermal_v()
        if v_max < beam.get_beam()[1] + 3.0*beam.thermal_v():
            v_max = beam.get_beam()[1] + 3.0*beam.thermal_v()
    v_matrix = np.linspace(v_min, v_max, 1000)
    f = np.zeros(len(v_matrix))
    for beam in beam_matrix:
        f = f + beam.v_dist(v_matrix)
    plt.plot(v_matrix, f)
    plt.ylabel('Number Density')
    plt.xlabel('V')
    plt.title('Initial Distribution with respect to velocity')
    plt.savefig('q2.png')
    plt.close()
    return f, v_matrix


def q3(f, v_matrix, mw=1.0):
    dv = v_matrix[1] - v_matrix[0]
    rho = np.sum(f*v_matrix)
    v_mean = np.sum(v_matrix*f)*dv/rho
    sigma = np.sqrt(np.sum((v_matrix - v_mean)**2*f*dv)/rho)
    t = sigma*sigma*mw*m_p/e
    return rho, v_mean, t


def q4(beam, v):
    plt.plot(v, beam.v_dist(v))
    plt.xlabel('V')
    plt.ylabel('n(V)')
    plt.title('Equilibrium distribution with respect to velocity', y=1.05)
    plt.savefig('q4.png')
    plt.close()
    return beam.v_dist(v)


def q5(initial_f, mean_f, v_matrix, freq, m, F):
    dv = v_matrix[1] - v_matrix[0]
    dt = 0.1/freq
    f_mid = np.zeros((1, len(initial_f)))
    f_mid[0, :] = copy.copy(initial_f)
    while (np.sum((f_mid[-1] - mean_f)**2)**0.5/np.sum(mean_f) >= 0.01):
        f = copy.copy(f_mid[-1])
        bgk = freq*(mean_f - f)
        F_eff = F/(m*m_p)*df(mean_f, dv)
        f = f + (bgk + F)*dt
        f_mid = np.vstack([f_mid, copy.copy(f)])
    return f_mid, f_mid.shape


def q6(f, v, den):
    s = np.zeros(len(f[:, 0]))
    for i in range(len(f)):
        s[i] = entropy(v, f[i]/den)
    plt.plot(s)
    plt.savefig('q6.png')

if __name__ == '__main__':
    hot_beam = beam(0.0, 10**10, 1.0)
    thermal_v = hot_beam.thermal_v()
    cold_beam1 = beam(50.0*thermal_v, 10**10, 0.01, 1)
    cold_beam2 = beam(-50.0*thermal_v, 10**10, 0.01, 1)
    f_cold = q1([hot_beam, cold_beam1, cold_beam2])
    print "We can consider of the order of"+str(1/f_cold)+"s"
    f, v_matrix = q2([hot_beam, cold_beam1, cold_beam2])
    rho, v_mean, t = q3(f, v_matrix)
    print "Density = "+str(rho)+", temperature = "+str(t)
    eq_beam = beam(v_mean, rho, t)
    mean_f = q4(eq_beam, v_matrix)
    evolved_f, i = q5(f, mean_f, v_matrix, f_cold, 1.0, 0.0)
    print "It takes of the order of "+str(i[0]*0.1/f_cold)+"s to deviate from maxwell."
    q6(evolved_f, v_matrix, rho)
