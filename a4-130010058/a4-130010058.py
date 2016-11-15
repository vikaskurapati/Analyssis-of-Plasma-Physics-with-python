import numpy as np
from matplotlib import pyplot as plt
import copy

k = 1.38064e-23
e = 1.60271e-19
eps = 8.85419e-12
m_p = 1.67262e-27


class beam:
    '''Temperature is taken in electron volts for convinience'''
    def __init__(self, v, rho, t, mw):
        self.rho = rho
        self.v = v
        self.t = t*e/k
        self.mw = mw*m_p

    def get_beam(self):
        return self.rho, self.v, self.t, self.mw

    def v_dist(self):
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
    s = 0.0
    for i in range(len(v)):
        if(pdf[i]) < 1e-12:
            continue
        else:
            s = s+pdf[i]*np.log(pdf[i])*(v[i] - v[i-1])
    return s


def q1(beam_matrix):
    fmax = 0.0
    for beam in beam_matrix:
        if fmax < beam.freq():
            fmax = beam.freq()
    return fmax


def q2(beam_matrix):
    v_min = beam_matrix[0].get_beam()[1] - 5.0*beam_matrix[0].thermal_v()
    v_max = beam_matrix[0].get_beam()[1] + 5.0*beam_matrix[0].thermal_v()
    for beam in beam_matrix:
        if v_min > beam.get_beam()[1] - 5.0*beam.thermal_v():
            v_min = beam.get_beam()[1] - 5.0*beam.thermal_v()
        if v_max < beam.get_beam()[1] + 5.0*beam.thermal_v():
            v_max = beam.get_beam()[1] + 5.0*beam.thermal_v()
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


def q3(f, v_matrix, mw):
    dv = v_matrix[1] - v_matrix[0]
    rho = np.sum(f*v_matrix)
    v_mean = np.sum(v_matrix*f)*dv/rho
    sigma = np.sqrt(np.sum((v_matrix - v_mean)**2*f*dv)/rho)
    t = sigma*sigma*mw*m_p/# coding=utf-8
    return rho, v_mean, t


def q4(beam, v):
    plt.plot(v, beam.v_dist(vel))
    plt.xlabel('V')
    plt.ylabel('n(V)')
    plt.title('equilibrium distribution with respect to velocity')
    plt.savefig('q4.png')
    plt.close()


def q5(initial_f, mean_f, dv, freq, m, F):
    dt = 0.1/freq
    f_mid = np.zeros((1, len(initial_f)))
    f_mid[0, :] = copy.copy(f)
    while (np.sum((f[-1] - mean_f)**2)**0.5/np.sum(mean_f) >= 0.01):
        f = copy.copy(f_mid[-1])
        bgk_rhs = col_freq*(fmean - f)
        extf_term = ext_force/(mass*mp)*delf(fmean, dv)
        f += (bgk_rhs + ext_force)*delt
        f_evol = np.vstack([f_evol, f.copy()])
    return f_evol
