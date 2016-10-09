#Assignment1
#Vikas Kurapati
#130010058

import numpy as np 
import matplotlib.pyplot as plt

N = 6.022e+23
k = 1.38e-23
amu = 1.66053904e-27

def pdf_velocity(v, gmw, rho, t):
	gmw = gmw*amu
	nden = rho/gmw
	return nden*(gmw/(2*np.pi*k*t))**1.5*np.exp(-1*gmw*v**2/(2*k*t))

def plot_pdf_velocity(low,up,npoints,gmw,rho,t):
	velocities = np.linspace(low,up,npoints,endpoint = True)
	pdfs = pdf_velocity(velocities,gmw,rho,t)
	v_avg = 0
	v_rms = (3*k*t / (amu*gmw))**0.5
	plt.plot(velocities,pdfs)
	plt.xlabel('vx')
	plt.ylabel('f(vx,vy,vz)')
	plt.vlines(v_avg, 0, pdf_velocity(v_avg,gmw,rho,t), label = "Mean", color = "red", linestyle = 'dashed')
	plt.vlines(v_rms, 0, pdf_velocity(v_rms,gmw,rho,t), label = "RMS", color = "blue", linestyle = 'dashed')
	plt.legend()
	plt.savefig("Q3vel.png")
	plt.show()

def pdf_speed(v, gmw, rho, t):
	gmw = gmw*amu
	nden = rho/gmw
	return nden*(gmw/(2*np.pi*k*t))**1.5*4*np.pi*v*v*np.exp(-1*gmw*v**2/(2*k*t))

def plot_pdf_speed(up,npoints,gmw,rho,t):
	velocities = np.linspace(0,up,npoints,endpoint = True)
	pdfs = pdf_speed(velocities,gmw,rho,t)
	v_avg = (8*k*t / (np.pi*amu*gmw))**0.5
	v_rms = (3*k*t / (amu*gmw))**0.5
	plt.plot(velocities,pdfs)
	plt.xlabel('v')
	plt.ylabel('f(v)')
	plt.vlines(v_avg, 0, pdf_speed(v_avg,gmw,rho,t), label = "Mean", color = "red", linestyle = 'dashed')
	plt.vlines(v_rms, 0, pdf_speed(v_rms,gmw,rho,t), label = "RMS", color = "blue", linestyle = 'dashed')
	plt.legend()
	plt.savefig("Q3speed.png")
	plt.show()

def pdf_energy(e, gmw, rho, t):
	gmw = gmw*amu
	nden = rho/gmw
	return nden*np.pi/(np.pi*k*t)**1.5*e**0.5*np.exp(-e/(k*t))

def plot_pdf_energy(up,npoints,gmw,rho,t):
	energies = np.linspace(0,up,npoints,endpoint = True)
	pdfs = pdf_energy(energies,gmw,rho,t)
	e_avg = 1.5*k*t
	plt.plot(energies,pdfs)
	plt.xlabel('e')
	plt.ylabel('f(e)')
	plt.vlines(e_avg, 0, pdf_energy(e_avg,gmw,rho,t), label = "Mean", color = "red", linestyle = 'dashed')
	plt.legend()
	plt.savefig("Q3energy.png")
	plt.show()

def pressure_mom(up_lim,npoints,gmw,rho,t):
	pressure = 0
	v = np.linspace(0,up_lim,npoints,endpoint = True)
	pdfs = pdf_speed(v,gmw,rho,t)
	for i in range(len(pdfs)):	
		pressure += ((v[i] - v[i-1])*gmw*amu*v[i]*v[i]*pdfs[i]/3)
	return pressure

def energy_mom(up_lim,npoints,gmw,rho,t):
	energy = 0
	e = np.linspace(0,up_lim,npoints,endpoint = True)
	pdfs = pdf_energy(e,gmw,rho,t)
	for i in range(len(pdfs)):
		energy += e[i] * pdf_energy(e[i],gmw, rho, t)*(e[i] - e[i-1])
	return energy

def vmean_mom(up_lim,npoints,gmw,rho,t):
	vmean = 0
	v = np.linspace(0,up_lim,npoints,endpoint = True)
	pdfs = pdf_speed(v,gmw,rho,t)
	for i in range(len(v)):
		vmean += v[i]*(v[i] - v[i-1]) * pdfs[i]
	return vmean*gmw*amu/rho

def entropy_mom(up_lim,npoints,gmw,rho,t,func):
	entropy = 0
	v = np.linspace(0,up_lim,npoints,endpoint = True)
	pdfs = func(v,gmw,rho,t)
	for i in range(len(v)):
		if(pdfs[i]!= 0.0):
			entropy += pdfs[i]*np.log(pdfs[i])*(v[i] - v[i-1])
	return entropy

def exponential(v,x,y,z):
	return np.exp(-1*v)

def exp_log(v, p, beta,z):
	return -1/np.log(p)*beta*(1-p)*np.exp(-beta*v)/(1 - (1 - p)*np.exp(-beta*v))

def standardcauchy(v,x,y,z):
	return 1/((1 + v*v)*np.pi)

def question6(uplim_vel,uplim_e,gmw, rho, t):      
	pressure_id = (rho/(gmw*amu))*k*t
	v_mean_id = (8*k*t/(np.pi * gmw*amu))**0.5
	nden = rho/(gmw*amu)
	energy_id = 1.5*nden*k*t
	ngrids = [50, 100, 200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
	normalised_error_p = np.zeros(len(ngrids))
	normalised_error_e = np.zeros(len(ngrids))
	normalised_error_v = np.zeros(len(ngrids))

	for i in range(len(ngrids)): 
		pressure = pressure_mom(uplim_vel,ngrids[i],gmw,rho,t)
		energy = energy_mom(uplim_e,ngrids[i],gmw,rho,t)
		vmean = vmean_mom(uplim_vel,ngrids[i],gmw,rho,t)
		normalised_error_p[i] = abs(pressure - pressure_id)/pressure_id
		normalised_error_e[i] = abs(energy - energy_id)/energy_id
		normalised_error_v[i] = abs( vmean- v_mean_id)/v_mean_id

	plt.figure(1)
	plt.plot(ngrids,normalised_error_p)
	plt.xlabel('Number of grid Points')
	plt.ylabel('Normalised error of pressure')
	plt.savefig('Q6pressure.png')
	plt.show()

	plt.figure(2)
	plt.plot(ngrids,normalised_error_e)
	plt.xlabel('Number of grid Points')
	plt.ylabel('Normalised error of Energy')
	plt.savefig('Q6energy.png')
	plt.show()

	plt.figure(3)
	plt.plot(ngrids,normalised_error_v)
	plt.xlabel('Number of grid Points')
	plt.ylabel('Normalised error of Mean Velocity')
	plt.savefig('Q6vel.png')
	plt.show()

def question7(gmw,rho,t,npoints,up_lim):
	gmw = gmw*amu
	nden = rho/gmw
	v = np.linspace(0.001,up_lim,npoints,endpoint = True)
	entropy_maxw_boltz = entropy_mom(up_lim,npoints,gmw,rho,t,pdf_speed)
	entropy_exp = entropy_mom(up_lim,npoints,gmw,rho,t,exponential)
	entropy_exp_log = entropy_mom(up_lim,npoints,gmw,rho,t,exp_log)
	entropy_std_cauch = entropy_mom(up_lim,npoints,gmw,rho,t,standardcauchy)
	return entropy_maxw_boltz,entropy_exp,entropy_exp_log,entropy_std_cauch

def proton_electron_plot(name, xdata, ydata, xlabel, ylabel, flag):
	figure, axis1 = plt.subplots()
	axis1.plot(xdata[0], ydata[0], 'r-')
	axis1.set_ylabel(ylabel[0])
	axis1.set_xlabel(xlabel)

	if flag == 0:
		axis1.set_xscale('symlog')

	for tl in axis1.get_yticklabels():
		tl.set_color('r')

	axis2 = axis1.twinx()
	axis2.plot(xdata[1], ydata[1], 'b-')
	axis1.set_ylabel(ylabel[1])
	axis1.set_xlabel(xlabel)

	if flag == 0:
		axis1.set_xscale('symlog')

	for tl in axis2.get_yticklabels():
		tl.set_color('b')

	plt.savefig(name)
	plt.show()

def question8(rho,t,npoints):
	mp = 1.67e-27/amu 	#As all the functions are taking gmw in amu so dividing it here to neutralise the multiplication inside the functions
	me = 9.11e-31/amu	#Same as above
	sigmap = (k*t/(amu*mp))**0.5
	sigmae = (k*t/(amu*me))**0.5
	velp = np.linspace(-5*sigmap,5*sigmap,npoints,endpoint = True)
	vele = np.linspace(-5*sigmae,5*sigmae,npoints,endpoint = True)
	speedp = np.linspace(0,10*sigmap,npoints,endpoint = True)
	speede = np.linspace(0,10*sigmae,npoints,endpoint = True)
	ep = 0.5*mp*amu*speedp*speedp
	ee = 0.5*me*amu*speede*speede
	pdf_velp = pdf_velocity(velp, mp, rho, t)
	pdf_vele = pdf_velocity(vele, me, rho, t)
	pdf_speedp = pdf_speed(speedp, mp, rho, t)
	pdf_speede = pdf_speed(speede, me, rho, t)
	pdf_energyp = pdf_energy(ep, mp, rho, t)
	pdf_energye = pdf_energy(ee, me, rho, t)
	proton_electron_plot("Plasma_vel.png", [velp, vele], [pdf_velp, pdf_vele], "log(v)", ["f(v_p)", "f(v_e)"], 0)
	proton_electron_plot("Plasma_speed.png", [speedp, speede], [pdf_speedp, pdf_speede], "log(v)", ["f(v_p)", "f(v_e)"], 0)
	proton_electron_plot("Plasma_energy.png", [ep[0:npoints/2], ee[0:npoints/2]], [pdf_energyp[0:npoints/2], pdf_energye[0:npoints/2]], "e", ["f(ve)", "f(e)"],1) 

plot_pdf_velocity(-1000,1000,10000,28.1,1.225,300)
plot_pdf_speed(2000,10000,28.1,1.225,300)
plot_pdf_energy(1.2e-19,1000,28.1,1.225,300)
print "Pressure calculated with velocities until %d with %d grid points is %d"%(2000,20000,pressure_mom(2000,20000,28.1,1.225,300))
print "Internal Energy calculated with energies until %s with %d grid points is %d"%('1.2e-19',1000,energy_mom(1.2e-19,1000,28.1,1.225,300))
print "Mean Velocity calculated with velocities until %d with %d grid points is %d"%(2000,20000,vmean_mom(2000,20000,28.1,1.225,300))
print "Entropy calculated with velocities until %d with %d grid points is %d"%(2000,20000,entropy_mom(2000,20000,28.1,1.225,300,pdf_speed))
question6(2000,1.2e-19,28.1,1.225,300)
print question7(28.1,1.225,300,20000,2000)
question8(1.225,300,100)