import numpy as np
from matplotlib import pyplot as plt

def power10(eps):
	i = 0
	while(eps < 1.0):
		eps = eps*10
		i = i+1
	return i

def q3(eps,n,up):
	ETA = np.linspace(0.0,up,n)
	deta = (up - 0.0)/n
	V_gen = np.zeros((len(ETA)))
	for i in range(len(ETA) - 1):
		V_gen[i+1] = V_gen[i] + deta*np.sqrt(eps*eps + 2*(np.sqrt(1 + 2*V_gen[i]) + np.exp(-V_gen[i]) -2))
	dV_gen = np.sqrt(eps*eps + 2*(np.sqrt(1 + 2*V_gen) + np.exp(-V_gen) -2))
	dV_bohm = np.empty(len(ETA))
	dV_bohm.fill(eps)
	V_bohm = eps*ETA
	V_CL = ((3*ETA)**(4.0/3.0))/(2**(5.0/3.0))
	dV_CL = (6*ETA)**(1.0/3.0)
	plt.figure(1)
	plt.plot(ETA,dV_gen,label = 'General Case')
	plt.plot(ETA,dV_bohm,label = "Bohm Case")
	plt.plot(ETA,dV_CL, label = "Child_Langmuir Case")
	plt.legend()
	plt.title("Plot comparing normalized sheath electric field for epsilon = "+str(eps))
	plt.savefig('q3Eeps'+str(power10(eps))+'.png')
	plt.close()
	plt.figure(2)
	plt.plot(ETA,V_gen,label = 'General Case')
	plt.plot(ETA,V_bohm,label = "Bohm Case")
	plt.plot(ETA,V_CL, label = "Child_Langmuir Case")
	plt.legend()
	plt.title("Plot comparing normalized sheath Potential for epsilon = "+str(eps))
	plt.savefig('q3Veps'+str(power10(eps))+'.png')
	plt.close()

if __name__ == '__main__':
	q3(0.1,1000,100.0)
	q3(0.01,1000,100.0)
	q3(0.001,1000,100.0)