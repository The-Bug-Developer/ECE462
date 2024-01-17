""" Se2_eigen.py
This program finds the eigenenergies 
and eigenfunctions for a potential V
"""

from math import pi,sqrt, exp, cos, sin
import numpy as np
from matplotlib import pyplot as plt
from numba import jit

NN = 100
N2 = int(NN/2)

hbar = 1.054e-34
m0 = 9.1e-31
meff = 1.0
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

""" ------------ Specify the potential ------------"""

# Harmonic Oscillator
V = np.zeros(NN)
#Eref = 0.05*eV2J
#for n in range(0,NN):
#    V[n] = 0.5*m0*((Eref/hbar)**2)*(del_x**2)*(n-N2)**2
 
# V well
Eref = 0.05*eV2J
#for n in range(NN):
#    V[n] = eV2J*(0.001)*abs(N2-n)
    
plt.subplot(321)
plt.plot(XX,J2eV*V,'k')
plt.text(2,1.0,"Eref:   {}".format(round(Eref*J2eV,4)),fontsize=14)
plt.grid()
plt.title("Find_eig")
plt.ylabel("V (eV)")

"""------- Create the H matrix ------"""

chi0 = 0.5*(hbar/m0)*(hbar/del_x**2)

H = np.zeros( (NN,NN) )
H[0,0] = 2*chi0 + V[0]
H[0,1] = -1*chi0
for n in range(1,NN-1):
    H[n,n-1] = -1*chi0
    H[n,n] = 2*chi0 + V[n]
    H[n,n+1] -1*chi0

H[NN-1,NN-2]   = -1*chi0
H[NN-1,NN-1] = 2*chi0 + V[NN-1]

"""Determine the eigenvalues and eigenfunctions"""
D, chi = np.linalg.eigh(H) 
eps = J2eV*D

print("Eref:", round(J2eV*Eref,4))
print("eps :", round(eps[0],4))

plt.subplot(322)
plt.plot(eps,'ok')
plt.ylabel("eps (eV)")
plt.grid()
plt.axis([0,10,0,1])

plt.subplot(312)
plt.plot(XX,chi[:,0],'b')
plt.plot(XX,chi[:,1],'r--')
plt.plot(XX,chi[:,2],'g')
plt.text(2,0.1,"e[0]:   {}".format(round(eps[0],4)),fontsize=14)
plt.grid()
plt.show()

""" Use one of the eigenfunctions to initialize the FDTD"""
    
prl = np.zeros(NN)
pim = np.zeros(NN)

prl = chi[:,0]
    
ptot = np.sum(prl**2 + pim**2)
pnorm = sqrt(ptot)
prl = np.divide(prl,pnorm)
pim = np.divide(pim,pnorm)

plt.subplot(311)
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

# ---------- The core FDTD loop ----------------------

while True:
    n_step = int(input("How many time steps--> "))
    if n_step == 0:
        break
    
    T = T + n_step
    fdtd(prl, pim, n_step)
 
# ----------- Calculate the observables --------------
                          
    ptot = round(np.sum(prl**2 + pim**2),5)
    psi = prl + pim*1j
    PE = round(np.sum( (prl**2 + pim**2)*V)*J2eV,4)
  
    ke = 0.
    for n in range(NN-2):
        ke = ke + (psi[n-1] - 2*psi[n] + psi[n+1])*np.conjugate(psi[n])
        
    KE = round(-J2eV*((hbar/del_x)**2/(2*m0))*ke.real,4)
    Etot = round(PE+KE,4)
    time = round(T*dt*1e15,5)
    
# ------------- Plot ---------------------------------

    plt.subplot(211)
    plt.title("Se2eigen",fontsize=14)
    plt.axis([ 0, NN*DX , -.2, .2])
    plt.xlabel('nm')
    plt.text(2,0.15,"{} fs".format(time),fontsize=14)           
    plt.text(2,0.1,"K.E.: {} eV".format(KE),fontsize=14)
    plt.text(2,0.05,"P.E.: {} eV".format(PE),fontsize=14)
    plt.text(2,-0.05,"Energy : {} eV".format(Etot),fontsize=14)
    plt.yticks([-.1,0,.1],fontsize=14)
    plt.plot(XX,J2eV*V,'k',label='V')
    plt.plot(XX,prl,'b',label='real')
    plt.plot(XX,pim,'r--',label='imag',linewidth=2)
#    plt.legend(loc="lower right",fontsize=14)
    plt.grid()
    plt.savefig('se1_1.png',dpi=300)
    plt.show()
    print("T = ",T,"   ptot = ",ptot)