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
	plt.legend(loc = 'upper left')
	plt.title("Plot comparing normalized sheath electric field for epsilon = "+str(eps))
	plt.savefig('q3Eeps'+str(power10(eps))+'.png')
	plt.close()
	plt.figure(2)
	plt.plot(ETA,V_gen,label = 'General Case')
	plt.plot(ETA,V_bohm,label = "Bohm Case")
	plt.plot(ETA,V_CL, label = "Child_Langmuir Case")
	plt.legend(loc = 'upper left')
	plt.title("Plot comparing normalized sheath Potential for epsilon = "+str(eps))
	plt.savefig('q3Veps'+str(power10(eps))+'.png')
	plt.close()

def q4(eps,n,up):
	m_i = 1.6726219e-27
	m_e = 9.10938356e-31
	V_w = 0.5*np.log(m_i/(2*np.pi*m_e))
	ETA = np.linspace(0.0,up,n)
	deta = (up - 0.0)/n
	V_gen = np.zeros((len(ETA)))
	for i in range(len(ETA) - 1):
		V_gen[i+1] = V_gen[i] + deta*np.sqrt(eps*eps + 2*(np.sqrt(1 + 2*V_gen[i]) + np.exp(-V_gen[i]) -2))
	dV_gen = np.sqrt(eps*eps + 2*(np.sqrt(1 + 2*V_gen) + np.exp(-V_gen) -2))
	i = 0
	while V_gen[i] < V_w:
		i = i + 1
	return ETA[i]

def q5(eps,n,up):
	m_i = 1.6726219e-27
	m_e = 9.10938356e-31
	V_w = 0.5*np.log(m_i/(2*np.pi*m_e))
	ETA = np.linspace(0.0,up,n)
	deta = (up - 0.0)/n
	V_gen = np.zeros((len(ETA)))
	for i in range(len(ETA) - 1):
		V_gen[i+1] = V_gen[i] + deta*np.sqrt(eps*eps + 2*(np.sqrt(1 + 2*V_gen[i]) + np.exp(-V_gen[i]) -2))
	dV_gen = np.sqrt(eps*eps + 2*(np.sqrt(1 + 2*V_gen) + np.exp(-V_gen) -2))
	i = 0
	while V_gen[i] < V_w:
		i = i + 1
	return dV_gen[i]

def q6(eps,deta,V0):
	v_max = 1.6021e-19*V0/1.3806e-19			#Assuming T_e = 10000K
	eta = 0.0	
	V = 0.0
	j = True
	while V < v_max:
		V = V + V + deta*np.sqrt(eps*eps + 2*(np.sqrt(1 + 2*V) + np.exp(-V) -2))
		eta = eta + deta
		if (abs(V - eta*eps) > abs(V - 3.0**(4.0/3.0)/(2.0**(5.0/3.0))*eta**(4.0/3.0))) and j == True:
			print "Child Langmuir starts at eta = %f for epsilon = %f for V = %f" %(eta,eps,V0)
			j = False
	return eta

if __name__ == '__main__':
	q3(0.1,300,30.0)
	q3(0.01,300,30.0)
	q3(0.001,300,30.0)
	print "Sheath thickness normalized with debye length to reach wall potential for epsilon %f is %f"%(0.1,q4(0.1,500,50.0))
	print "Sheath thickness normalized with debye length to reach wall potential for epsilon %f is %f"%(0.01,q4(0.01,500,50.0))
	print "Sheath thickness normalized with debye length to reach wall potential for epsilon %f is %f"%(0.001,q4(0.001,500,50.0))
	print "Sheath thickness normalized with debye length to reach wall potential for epsilon %f is %f"%(0.0001,q4(0.0001,1000,100.0))
	print "Sheath thickness normalized with debye length to reach wall potential for epsilon %f is %f"%(0.00001,q4(0.00001,10000,1000.0))
	print "Sheath thickness normalized with debye length to reach wall potential for epsilon %f is %f"%(0.000001,q4(0.000001,10000,1000.0))
	print "The normalized electric field for epsilon %f is %f"%(0.1,q5(0.1,500,50.0))
	print "The normalized electric field for epsilon %f is %f"%(0.01,q5(0.01,500,50.0))
	print "The normalized electric field for epsilon %f is %f"%(0.001,q5(0.001,500,50.0))
	print "The normalized electric field for epsilon %f is %f"%(0.0001,q5(0.0001,1000,100.0))
	print "The normalized electric field for epsilon %f is %f"%(0.00001,q5(0.00001,10000,1000.0))
	print "The normalized electric field for epsilon %f is %f"%(0.000001,q5(0.000001,10000,1000.0))	
	print "The thickness of sheath for a neutral wall with epsilon = %f and electrode potential = %f is %f" %(0.1,1,q6(0.01,0.01,1))
	print "The thickness of sheath for a neutral wall with epsilon = %f and electrode potential = %f is %f" %(0.1,1,q6(0.01,0.01,10))
	print "The thickness of sheath for a neutral wall with epsilon = %f and electrode potential = %f is %f" %(0.1,1,q6(0.01,0.01,100))
	print "The thickness of sheath for a neutral wall with epsilon = %f and electrode potential = %f is %f" %(0.01,1,q6(0.01,0.01,1))
	print "The thickness of sheath for a neutral wall with epsilon = %f and electrode potential = %f is %f" %(0.01,1,q6(0.01,0.01,10))
	print "The thickness of sheath for a neutral wall with epsilon = %f and electrode potential = %f is %f" %(0.01,1,q6(0.01,0.01,100))
	print "The thickness of sheath for a neutral wall with epsilon = %f and electrode potential = %f is %f" %(0.01,1,q6(0.001,0.01,1))
	print "The thickness of sheath for a neutral wall with epsilon = %f and electrode potential = %f is %f" %(0.01,1,q6(0.001,0.01,10))
	print "The thickness of sheath for a neutral wall with epsilon = %f and electrode potential = %f is %f" %(0.01,1,q6(0.001,0.01,100))