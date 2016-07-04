#!/usr/bin/python

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def main():


    global mu,Re,a,n

    mu = 3.986004e14
    Re = 6378
    a = 5000 + Re

    # n = np.sqrt(mu / a**3)
    #
    # t = np.linspace(0,100,100)
    # x0 = np.array([a, 0.0, 0.0])
    # v0 = np.array([0, n, 0])
    #
    #
    # # clohessey Wiltshire closed form solution
    # # xt = (4-3*np.cos(n*t))*x0[0] + (np.sin(n*t)/n)*v0[0]
    # # yt = 6*(np.sin(n*t)-n*t)*x0[0] + x0[1] - (2/n)*(1-np.cos(n*t))*v0[0] + (4*np.sin(n*t)-(3*n*t))/n * v0[1]
    # # zt = x0[2]*np.cos(n*t) + v0[2]/n * np.sin(n*t)
    #
    # y = odeint(clohessyWiltshire, [a, .5*a, .5*a, 0.0, n, 0], t)
    #
    #
    # plt.figure(1)
    # plt.plot(t, y[:,0])
    # plt.xlabel('t')
    # plt.ylabel('xyz')
    # plt.show()

    # Example 7.2 from Curtis

    # Target
    aTarg = Re + 300
    TTarg = 1.508   # hr
    thetaTarg = 60
    iTarg = 40
    OmegaTarg = 20
    omegaTarg = 0

    oeTarg = np.Array([aTarg,TTarg,thetaTarg,iTarg,OmegaTarg,omegaTarg])

    # Chaser
    aChase = Re + (318.50+515.51)/2
    TChase = 1.548
    thetaChase = 349.65
    iChase = 40.130
    OmegaChase = 19.819
    omegaChase = 70.662

    oeChase = np.array([aChase,TChase,thetaChase,iChase,OmegaChase,omegaChase])

    rv0 = oe_to_rv(oeTarg)
    rvChase = oe_to_rv(oeChase)

    ihat = np.array([rv0[0], rv0[1], rv0[2]])/np.linalg.norm([rv0[0],rv0[1],rv0[2]])
    jhat = np.array([rv0[3], rv0[4], rv0[5]])/np.linalg.norm([rv0[3],rv0[4],rv0[5]])
    khat = np.cross(ihat,jhat)

    # clohessey Wiltshire closed form solution
    # xt = (4-3*np.cos(n*t))*x0[0] + (np.sin(n*t)/n)*v0[0]
    # yt = 6*(np.sin(n*t)-n*t)*x0[0] + x0[1] - (2/n)*(1-np.cos(n*t))*v0[0] + (4*np.sin(n*t)-(3*n*t))/n * v0[1]
    # zt = x0[2]*np.cos(n*t) + v0[2]/n * np.sin(n*t)

    # matrix form
    rr= np.array([[4-3*np.cos(n*t), 0, 0], [6*(np.sin(n*t)-n*t), 1, 0], [0, 0, np.cos(n*t)]])
    rv = np.array([[(1/n*np.sin(n*t)), 2/n*(1-np.cos(n*t)), 0], [0,0,1/n*np.sign(n*t)]])
    vr = np.array([[3*n*np.sin(n*t),0,0],[6*n*(np.cos(n*t)-1), 0, 0], [0, 0, -m*np.sin(n*t)]])
    vv = np.array([[np.cos(n*t), 2*np.sin(n*t), 0],[-2*np.sin(n*t), 4*np.cos(n*t)-3, 0], [0, 0, np.cos(n*t)]])


def oe_to_rv(oe):
    a=oe[0]
    T=oe[1]
    theta=oe[3]
    i=oe[4]
    Omega=oe[5]

    r = (h**2/mu) * (1/(1+e*np.cos(theta))) * np.array([np.cos(theta), np.sin(theta), 0]).T
    v = mu/h*np.array([-np.sin(theta), e + np.cos(theta), 0]).T

    return np.array([r, v])


# Clohessy Wilshire model differential
def clohessyWiltshire(x,t):

    # set up the 2nd order odes as 1st order
    dx=x[3]  # xdot
    dy=x[4]  # ydot
    dz=x[5]  # zdot

    ddx = 3*(n**2)*x[0] + 2*n*x[4]
    ddy = -2*n*x[3]
    ddz = -n**2*x[2]

    # print(x)

    return np.array([x[0], x[1], x[2], dx, dy, dz])

if __name__ == '__main__':
    main()
