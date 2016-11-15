import numpy as np
from matplotlib import pyplot as plt

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
