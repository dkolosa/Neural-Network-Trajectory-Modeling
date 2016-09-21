#!/usr/bin/python

import numpy as np
import math
import csv

def main():

    
    global mu,Re,a,n

    Re = 6378
    mu = 398600

#    # Example 7.2 from Curtis
#
#    # Target
#    aTarg = Re + 300
#    TTarg = 1.508   # hr
#    thetaTarg = 60
#    iTarg = 40
#    OmegaTarg = 20
#    omegaTarg = 0
#
#    oeTarg = np.array([aTarg,TTarg,thetaTarg,iTarg,OmegaTarg,omegaTarg])
#
#    # Chaser
#    aChase = Re + (318.50+515.51)/2
#    TChase = 1.548
#    thetaChase = 349.65
#    iChase = 40.130
#    OmegaChase = 19.819
#    omegaChase = 70.662
#
#    oeChase = np.array([aChase,TChase,thetaChase,iChase,OmegaChase,omegaChase])
#
#    rv0 = oe_to_rv(oeTarg)
#    rvChase = oe_to_rv(oeChase)

    #Assume a known state vector (from curtis)

    # generate a random set of chaser and target values

    for x in range(1,100):

        r0random = np.random.uniform(low=100, high=10000.0, size=3)
        v0random = np.random.uniform(low=-5.0, high=10.0, size=3)

        # range for random value generation
        rdiff = 40
        vdiff = 0.1

        rtargrandom = np.array([np.random.uniform(low=r0random[0] - rdiff, high=r0random[0] + rdiff),
                                np.random.uniform(low=r0random[1] - rdiff, high=r0random[1] + rdiff),
                                np.random.uniform(low=r0random[2] - rdiff, high=r0random[2] + rdiff)])

        vtargrandom = np.array([np.random.uniform(low=v0random[0] - vdiff, high=v0random[0] + vdiff),
                                np.random.uniform(low=v0random[1] - vdiff, high=v0random[1] + vdiff),
                                np.random.uniform(low=v0random[2] - vdiff, high=v0random[2] + vdiff)])


        # Space station (target vechicle)
        # r0 = np.array([1622.39, 5305.10, 3717.44])
        # v0 = np.array([-7.29977, 0.492357, 2.48318])
        #
        # # Chaser
        # rtarg = np.array([1612.5,5310.19,3750.33])
        # vtarg = np.array([-7.35321, 0.463856, 2.46920])

        r0 = r0random
        v0 = v0random

        rtarg = rtargrandom
        vtarg = vtargrandom

        ihat = np.array([r0[0], r0[1], r0[2]])/np.linalg.norm(r0)
        jhat = np.array([v0[0], v0[1], v0[2]])/np.linalg.norm(v0)
        khat = np.cross(ihat, jhat)

        #coordinate transformation

        Q = np.concatenate((ihat, jhat, khat)).reshape((3, 3))

        deltar = rtarg - r0

        # Mean motion of the space station
        n = np.linalg.norm(v0)/np.linalg.norm(r0)


        deltav = vtarg - v0 - n*(np.cross(khat,deltar))

        deltar0 = Q.dot(deltar)
        deltav0 = Q.dot(deltav)

        #time of the 2-impulse maneuver
        t = 8 * 60**2  # hours to seconds


        # matrix form CW
        rr = np.array([4-3*np.cos(n*t), 0, 0, 6*(np.sin(n*t)-n*t), 1, 0, 0, 0, np.cos(n*t)]).reshape((3, 3))
        rv = np.array([(1/n*np.sin(n*t)), 2/n*(1-np.cos(n*t)), 0, 2/n*(np.cos(n*t)-1), (1/n)*(4*np.sin(n*t)-3*n*t), 0, 0,0,1/n*np.sign(n*t)]).reshape((3,3))
        vr = np.array([3*n*np.sin(n*t),0,0,6*n*(np.cos(n*t)-1), 0, 0, 0, 0, -n*np.sin(n*t)]).reshape((3,3))
        vv = np.array([np.cos(n*t), 2*np.sin(n*t), 0, -2*np.sin(n*t), 4*np.cos(n*t)-3, 0, 0, 0, np.cos(n*t)]).reshape((3,3))

        # print("rr =\n", rr)
        # print("rv =\n", rv)
        # print("vr =\n", vr)
        # print("vv = \n", vv)

        #First CW equation deltarf
        deltarf = rr.dot(deltar0) + rv.dot(deltav0)

        #Second CW equation deltavf
        deltavf = vr.dot(deltar0) + vv.dot(deltav0)
        # deltav = deltavf - deltav0

        print("deltarf =\n", deltarf)
        print("deltavf =\n", deltavf)

        # Save the initial and final states in the csv file
        trainingFile = open("trainingData.csv", 'w')
        saveTrainingData = csv.writer(trainingFile)

        saveTrainingData.writerow(deltar0)
        saveTrainingData.writerow(deltav0)
        saveTrainingData.writerow(deltarf)
        saveTrainingData.writerow(deltavf)
        trainingFile.close()

        del r0random, v0random, rtargrandom, vtargrandom


def oe_to_rv(oe):
    
    """Input a set of orbital elements and returns a state vector"""

    a=oe[0]
    e=oe[1]
    i=oe[2]
    w=oe[3]
    Omega=oe[4]
    theta=oe[5]

    if a<0 or e<0 or e>1 or math.fabs(1) > 2*math.pi or math.fabs(Omega)> 2*math.pi or math.fabs(w) > 2*math.pi:

        print('Invalid orbital elements')

    xhat = np.array([1, 0, 0])
    yhat = np.array([0, 0, 0]) 
    zhat = np.array([1, 0, 0])
    
    r = (h**2/mu) * (1/(1+e*np.cos(theta))) * np.array([np.cos(theta), np.sin(theta), 0]).T
    v = mu/h*np.array([-np.sin(theta), e + np.cos(theta), 0]).T

    nu = kepler(oe,t)
    nhat = math.cos(Omega)*xhat+math.sin(Omega)*yhat
    rhatT=-math.cos(i)*math.sin(Omega)*xhat+math.cos(i)*math.cos(Omega)*yhat+math.sin(i)*zhat
    rmag=a*(1-e**2)/(1+e*math.cos(nu))
    vmag=math.sqrt(mu/rmag*(2-rmag/a))
    gamma=math.atan2(e*math.sin(nu),1+e*math.cos(nu))
    u=w+nu
    rhat=math.cos(u)*nhat+math.sin(u)*rhatT
    vhat=math.sin(gamma-u)*nhat+math.cos(gamma-u)*rhatT
    r=rmag*rhat
    v=vmag*vhat

    return np.array([r, v])


#def diffCW():
    
   # """Solves the difeerential form of the CW equationsi"""
   # a = 5000 + Re
    # n = np.sqrt(mu / a**3)
    #
    # t = np.linspace(0,100,100)
    # x0 = np.array([a, 0.0, 0.0])
    # v0 = np.array([0, n, 0])
    #
    # y = odeint(clohessyWiltshireode, [a, .5*a, .5*a, 0.0, n, 0], t)
    #
    # plt.figure(1)
    # plt.plot(t, y[:,0])
    # plt.xlabel('t')
    # plt.ylabel('xyz')
    # plt.show()

# Clohessy Wilshire model differential
#def clohessyWiltshireode(x,t):
#
#    # set up the 2nd order odes as 1st order
#    dx=x[3]  # xdot
#    dy=x[4]  # ydot
#    dz=x[5]  # zdot
#
#    ddx = 3*(n**2)*x[0] + 2*n*x[4]
#    ddy = -2*n*x[3]
#    ddz = -n**2*x[2]
#
#    # print(x)
#
#    return np.array([x[0], x[1], x[2], dx, dy, dz])

if __name__ == '__main__':
    main()
