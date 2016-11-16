import numpy as np
from matplotlib import pyplot as plt
epsilon0 = 8.85419e-12
nu0 = 4*np.pi*1e-7
e = 1.60271e-19
m_p = 1.67262e-27
m_e = 9.10938356e-31


def lambdad(kT, n):
    return np.sqrt(epsilon0*kT/(n*e))


def rhoc(kT, B):
    vp = np.sqrt(3.0*kT*e/m_p)
    return m_p*vp/(e*B)


def lambdamfp(kT, n):
    return 36.0*np.pi*epsilon0*kT/(n*e*e*e)


def omegape(n):
    return np.sqrt(n*e*e/(m_e*epsilon0))


def omegace(B):
    return e*B/m_e


def beta(kT, B, n):
    return 2*nu0*n*kT*e/(B*B)


def prop(n, kT, B):
    return np.array([lambdad(kT, n)/rhoc(kT, B), lambdad(kT, n)/lambdamfp(kT, n), n*lambdad(kT, n)**3, omegape(n)/omegace(B), beta(kT, B, n)])

solar_wind = prop(10**7, 10, 2.8e-9)
sunspot = prop(10**20, 1, 3)
fusion_reactor = prop(10**20, 10000, 5)
print solar_wind
print sunspot
print fusion_reactor
plt.plot(solar_wind[0], 'o', label='Solar Wind')
plt.plot(sunspot[0], '+', label='Sunspot')
plt.plot(fusion_reactor[0], '*', label='Fusion reactor')
plt.axis([-0.01, 0.01, 0.9*min(solar_wind[0],sunspot[0],fusion_reactor[0]), 1.1*max(solar_wind[0],sunspot[0],fusion_reactor[0])])
plt.legend()
plt.savefig('q1_0.png')
plt.close()
plt.plot(solar_wind[1], 'o', label='Solar Wind')
plt.plot(sunspot[1], '+', label='Sunspot')
plt.plot(fusion_reactor[1], '*', label='Fusion reactor')
plt.axis([-0.01, 0.01, 0.9*min(solar_wind[1],sunspot[1],fusion_reactor[1]), 1.1*max(solar_wind[1],sunspot[1],fusion_reactor[1])])
plt.legend()
plt.savefig('q1_1.png')
plt.close()
plt.plot(solar_wind[2], 'o', label='Solar Wind')
plt.plot(sunspot[2], '+', label='Sunspot')
plt.plot(fusion_reactor[2], '*', label='Fusion reactor')
plt.axis([-0.01, 0.01, 0.9*min(solar_wind[2],sunspot[2],fusion_reactor[2]), 1.1*max(solar_wind[2],sunspot[2],fusion_reactor[2])])
plt.legend()
plt.savefig('q1_2.png')
plt.close()
plt.plot(solar_wind[3], 'o', label='Solar Wind')
plt.plot(sunspot[3], '+', label='Sunspot')
plt.plot(fusion_reactor[3], '*', label='Fusion reactor')
plt.axis([-0.01, 0.01, 0.9*min(solar_wind[3],sunspot[3],fusion_reactor[3]), 1.1*max(solar_wind[3],sunspot[3],fusion_reactor[3])])
plt.legend()
plt.savefig('q1_3.png')
plt.close()
plt.plot(solar_wind[4], 'o', label='Solar Wind')
plt.plot(sunspot[4], '+', label='Sunspot')
plt.plot(fusion_reactor[4], '*', label='Fusion reactor')
plt.axis([-0.01, 0.01, 0.9*min(solar_wind[4],sunspot[4],fusion_reactor[4]), 1.1*max(solar_wind[4],sunspot[4],fusion_reactor[4])])
plt.legend()
plt.savefig('q1_4.png')
plt.close()
