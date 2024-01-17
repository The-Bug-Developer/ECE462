""" Se1_1.py """

from math import pi,sqrt, exp, cos, sin
import numpy as np
from matplotlib import pyplot as plt
from numba import jit

NN = 400
N2 = int(NN/2)
hbar = 1.054e-34
m0 = 9.1e-31
ecoul = 1.6e-19
epsz = 8.85e-9
eV2J = 1.6e-19
J2eV = 1/eV2J

del_x = 0.1e-9
dt = 2e-17
ra = (0.5*hbar/m0)*(dt/del_x**2)
rd = dt/hbar
DX = del_x*1e9
    
XX = np.linspace(DX,DX*NN,NN)

""" Specify the potential """
V = np.zeros(NN)

#Barrier
V[N2:] = .1*eV2J

# Semiconductor conduction band
#V = 0.1*eV2J*np.ones(NN)
#V[N2:] = .2*eV2J

# Electric potential
#for n in range(NN):
#    V[n] = -(0.2/NN)*(n-NN)*eV2J


""" Initialize psi"""

lambdax = 35
sigma   = 25
nc      = 75         # Starting position

prl = np.zeros(NN)
pim = np.zeros(NN)
for n in range(NN):
    prl[n] = np.exp(-1.*((n-nc)/sigma)**2)*np.cos(2*pi*(n-nc)/lambdax)
    pim[n] = np.exp(-1.*(((n-nc)/sigma)**2))*np.sin(2*pi*(n-nc)/lambdax)
    
ptot = np.sum(prl**2 + pim**2)
pnorm = sqrt(ptot)
prl = np.divide(prl,pnorm)
pim = np.divide(pim,pnorm)

plt.subplot(211)
plt.plot(XX,J2eV*V,'k')    
plt.plot(XX,prl,'b')
plt.plot(XX,pim,'r--')
plt.grid()
plt.show()

@jit
def fdtd(prl, pim, n_step):
    print("FDTD: nstep = ",n_step)
    for _ in range(n_step):
        
        for n in range(NN-1):
            prl[n] = prl[n] - ra*(pim[n-1] - 2*pim[n] + pim[n+1])+rd*V[n]*pim[n]
            
        for n in range(NN-1): 
            pim[n] = pim[n] + ra*(prl[n-1] - 2*prl[n] + prl[n+1])-rd*V[n]*prl[n]
        
T = 0

#""" The core FDTD loop """

while True:
    n_step = int(input("How many time steps--> "))
    if n_step == 0:
        break
    
    T = T + n_step
    fdtd(prl, pim, n_step)
 
# Calculate the observables 

    print('Got here')
                          
    ptot = round(np.sum(prl**2 + pim**2),5)
    psi = prl + pim*1j
    PE = round(np.sum( (prl**2 + pim**2)*V)*J2eV,4)  
    ke = 0.
    for n in range(NN-2):
        ke = ke + (psi[n-1] - 2*psi[n] + psi[n+1])*np.conjugate(psi[n])
        
    KE = round(-J2eV*((hbar/del_x)**2/(2*m0))*ke.real,4)
    Etot = round(PE+KE,4)
    time = round(T*dt*1e15,5)
    
# Plot 

    plt.subplot(211)
    plt.title("Se1_1",fontsize=12)
    plt.axis([ 0, NN*DX , -.2, .3])
    plt.xlabel('nm')
    plt.text(2,0.15,"{} fs".format(time),fontsize=12)           
    plt.text(2,-0.15,"K.E.: {} eV".format(KE),fontsize=12)
    plt.text(20,-.15,"P.E.: {} eV".format(PE),fontsize=12)
    plt.text(20,0.15,"Energy : {} eV".format(Etot),fontsize=12)
    plt.yticks([-.1,0,.1,.2],fontsize=14)
    plt.plot(XX,J2eV*V,'k',label='V')
    plt.plot(XX,prl,'b',label='real')
    plt.plot(XX,pim,'r--',label='imag',linewidth=2)
    plt.grid()
    plt.savefig('se1_1.png',dpi=300)
    plt.show()
    print("T = ",T,"   ptot = ",ptot)