#!/usr/bin/python3

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from net_demo import *
from sklearn.preprocessing import MinMaxScaler

def main():
    """Solving the linear CW equations given a set of state vectors
       for a target and chaser vehicle"""
    global mu, Re, hr_to_sec

    Re = 6378
    mu = 398600
    hr_to_sec = 60 ** 2

    a = 300 + Re  # semi-major axis
    n = np.sqrt(mu/a**3)   # mean motion circular orbit

    tfinal = 2*np.pi/n*2

    time_span = np.arange(0, tfinal, 10)


# """
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

    # Assume a known state vector (from curtis)

    # Delete file contents
    # remove_data('trainingData.csv')

    # do the ode of the CW equations

    y_CW = diffCW(time_span, a, n)

    # Use the CW equations as input and output for a neural network
    test_regression(y_CW, time_span, True)

    pass


# generate a random set of chaser and target values
def cw_Linear():
    for x in range(20):
        # Space Station
        r0random = np.random.uniform(low=10, high=50.0, size=3)
        v0random = np.random.uniform(low=0.10, high=2.0, size=3)

        # range for random value generation
        rdiff = 10
        vdiff = 0.1

        # Spacecraft
        rtargrandom = np.array([np.random.uniform(low=r0random[0] - rdiff, high=r0random[0] + rdiff),
                                np.random.uniform(low=r0random[1] - rdiff, high=r0random[1] + rdiff),
                                np.random.uniform(low=r0random[2] - rdiff, high=r0random[2] + rdiff)])

        vtargrandom = np.array([np.random.uniform(low=v0random[0] - vdiff, high=v0random[0] + vdiff),
                                np.random.uniform(low=v0random[1] - vdiff, high=v0random[1] + vdiff),
                                np.random.uniform(low=v0random[2] - vdiff, high=v0random[2] + vdiff)])

        # # Space station (target vechicle)
        # r0 = np.array([1622.39, 5305.10, 3717.44])
        # v0 = np.array([-7.29977, 0.492357, 2.48318])
        # # Chaser
        # rtarg = np.array([1612.75, 5310.19, 3750.33])
        # vtarg = np.array([-7.35321, 0.463856, 2.46920])

        # space Station
        r0, v0 = r0random, v0random
        rtarg, vtarg = rtargrandom, vtargrandom

        ihat = r0 / np.linalg.norm(r0)
        jhat = v0 / np.linalg.norm(v0)
        khat = np.cross(ihat, jhat)

        # coordinate transformation from geocentric to equatorial to space station
        Q = np.concatenate(([ihat], [jhat], [khat]))

        deltar = rtarg - r0
        # Mean motion of the space station
        n = np.linalg.norm(v0)/np.linalg.norm(r0)

        Omega_space_station = n*khat
        deltav = vtarg - v0 - np.cross(Omega_space_station, deltar)
        deltar0 = Q.dot(deltar)
        deltav0 = Q.dot(deltav)

        # time of the 2-impulse maneuver
        # solve cw over a timespan
        tfinal = 8 * 60**2  # hours to seconds
        # t = np.linspace(0, tfinal, 100)
        t = tfinal

        # deltavf = np.zeros((len(t), 3))
        # deltarf = np.zeros((len(t), 3))

        # for index in range(len(t)):
        # matrix form CW
        rr = np.array([4 - 3 * np.cos(n * t), 0, 0,
                       6 * (np.sin(n * t) - n * t), 1, 0,
                       0, 0, np.cos(n * t)]).reshape((3, 3))
        rv = np.array([(1 / n * np.sin(n * t)), 2 / n * (1 - np.cos(n * t)), 0,
                       2 / n * (np.cos(n * t) - 1), (1 / n) * (4 * np.sin(n * t) - 3 * n * t), 0,
                       0, 0, 1 / n * np.sin(n * t)]).reshape((3, 3))
        vr = np.array([3 * n * np.sin(n * t), 0, 0,
                       6 * n * (np.cos(n * t) - 1), 0, 0,
                       0, 0, -n * np.sin(n * t)]).reshape((3, 3))
        vv = np.array([np.cos(n * t), 2 * np.sin(n * t), 0,
                       -2 * np.sin(n * t), 4 * np.cos(n * t) - 3, 0,
                       0, 0, np.cos(n * t)]).reshape((3, 3))

        deltav0p = -np.linalg.inv(rv).dot(np.dot(rr, deltar0))

        # # First CW equation deltarf is assumed zero
        # deltarf[t:] = [rr.dot(deltar0) + rv.dot(deltav0)]
        # # Second CW equation deltavf
        # deltavf[t:] = [vr.dot(deltar0) + vv.dot(deltav0)]

        # First CW equation deltarf is assumed zero
        deltarf = rr.dot(deltar0) + rv.dot(deltav0p)
        # Second CW equation deltavf
        deltavf = vr.dot(deltar0) + vv.dot(deltav0p)

        deltarv0 = np.concatenate((deltar0, deltav0p)).reshape((1, 6))
        deltarvf = np.concatenate((deltarf, deltavf)).reshape((1, 6))
        delta = np.array([[deltarv0], [deltarvf]])
        # Save the initial and final states in the csv file to use for training
        save_data('trainingData.csv', delta)

        # Save the test data and the output test data
        save_data('testData.csv', np.reshape(deltar0, (1, 3)))
        save_data('outputTestData.csv', np.reshape(deltavf, (1, 3)))

        del r0random, v0random, rtargrandom, vtargrandom
    pass


def save_data(file, data):
    """Save the file contents"""
    with open(file, 'ab') as data_file:
        np.savetxt(data_file, data, delimiter=',')


def remove_data(file):
    """Delete the file contents only"""
    with open(file, 'w') as f:
        pass


def diffCW(time_span, a, n):
    """
    Solves the difeerential form of the CW equations

    Arguments:
            time_span: time span of interest
            a : semi-major axis
            n : mean motion

    Units:
        r = km
        time = seconds
    """

    # ode parameters
    abserr = 1.0e-8
    relerr = 1.0e-6

    # Chaser
    delta_r0 = [0.0, 0.0, 0.0]
    delta_v0 = [0, -0.01, 0]
    # delta_r0 = [1.2, 2.3, 1.26]
    # delta_v0 = [0.31667, 0.11199, 1.247]
    delta_y0 = delta_r0 + delta_v0

    y_rel = odeint(CW_ode, delta_y0, time_span, args=(n,),
                   atol=abserr, rtol=relerr)

    # target
    x0 = [a, 0, 0]
    v0 = [0, np.sqrt(mu/a), 0]
    y0 = x0 + v0
    y_abs = odeint(two_body, y0, time_span,
                   atol=abserr, rtol=relerr)

    # plots(time_span, y_rel, y_abs)

    return y_rel


def plots(time_span, y_rel, y_abs):
    """
    Generates a set of plots for the CW equations
    :param time_span: time duration of interest
    :param y_rel: output of the CW equations
    :param y_abs: output of the two body problem
    :return: null
    """
    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(time_span, y_rel[:, 0], label='x')
    plt.plot(time_span, y_rel[:, 1], label='y')
    plt.plot(time_span, y_rel[:, 2], label='z')
    plt.xlabel('time (s)')
    plt.ylabel('position')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(y_rel[:, 0], y_rel[:, 1])
    plt.ylabel('y')
    plt.xlabel('x')

    plt.figure(2)
    plt.axes(projection='3d')
    plt.plot(y_rel[:, 0], y_rel[:, 1], y_rel[:, 2])

    plt.figure(3)
    plt.axes(projection='3d')
    plt.plot(y_abs[:,0], y_abs[:,1], y_abs[:,2])
    plt.title('two boundary')

    plt.show()


def cw_cloed_form(time_span, n, delta_r0, delta_v0):
    """
        Computes the closed form solution of the CW equations in the LVLH frame

        Arguments:
            time_span : time_span of the orbit
            n : mean motion
            delta_r0 : Initial conditions of position
                delta_r0 = [x, y, z]
            delta_v0 : initial conditions of velocity
                delta_v0 = [u, v, w]
    """

    xt = (4-3*np.cos(n*time_span))*delta_r0[0] + (np.sin(n*time_span)/n)*delta_v0[0] + (2/n)*(1-np.cos(n*time_span))*delta_v0[1]
    yt = 6*(np.sin(n*time_span) - n*time_span)*delta_r0[0] + delta_r0[1]+ (2/n)*(np.cos(n*time_span) - 1)*delta_v0[0] + (1/n)*(4*np.sin(n*time_span)-3*n*time_span)*delta_v0[1]
    zt = np.cos(n*time_span)*delta_r0[2] + (1/n)*np.sin(n*time_span)*delta_v0[2]
    ut = 3*n*np.sin(n*time_span)*delta_r0[0] + np.cos(n*time_span)*delta_v0[0] + 2*np.sin(n*time_span)*delta_v0[1]
    vt = 6*n*(np.cos(n*time_span) - 1)*delta_r0[0] - 2*np.sin(n*time_span)*delta_v0[0] + (4*np.cos(n*time_span) - 3)*delta_v0[1]
    wt = -n*np.sin(n*time_span)*delta_r0[2] + np.cos(n*time_span)*delta_v0[2]

    y_rel = np.column_stack((xt, yt, zt))
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(time_span, xt, label='x')
    plt.plot(time_span, yt, label='y')
    plt.plot(time_span, zt, label='z')

    plt.subplot(3,1,2)
    plt.plot(time_span, ut, label='u')
    plt.plot(time_span, vt, label='v')
    plt.plot(time_span, wt, label='w')

    plt.show()
    return y_rel


def CW_ode(w, t, n):
    """
    Defines the differential equation in terms of the moving frame (CW frame LVLH)
    Arguments:
        w : vector of state variables
            w = [x,y,z,dx,dy,dz]
        t : time
        n : mean motion
    """

    x, y, z, dx, dy, dz = w

    ddx = 3*(n ** 2)*x + 2*n*dy
    ddy = -2*n*dx
    ddz = -n**2*z

    return [dx, dy, dz, ddx, ddy, ddz]


def two_body(y, t):
    """
    Solves the two body problem
    :param y: state vector
        y = [rx,ry,rz,vx,vy,vz]
    :param t: time
    :return: dy
    """
    rx, ry, rz = y[0], y[1], y[2]
    vx, vy, vz = y[3], y[4], y[5]

    r = np.array([rx, ry, rz])
    v = np.array([vx, vy, vz])

    r_mag = np.linalg.norm(r)
    c = -mu / (r_mag ** 3)

    dy = np.zeros(6)

    dy[0] = y[3]
    dy[1] = y[4]
    dy[2] = y[5]
    dy[3] = c*y[0]
    dy[4] = c*y[1]
    dy[5] = c*y[2]

    return dy


def test_regression(y_CW, X, plots=True):
    """
    Creates, trains, and tests a neural network
    :param y_CW: the input dataset
    :param X: time span of the dataset
    :param plots: generate the plots of training and test data
    :return: null
    """
    #First create the data.
    # n = 100   # Number of points
    # t = np.linspace(0, 2*np.pi, num=n)  # Function range (time span)
    # t.shape = (n, 1)
    # y_org = np.sin(t)
    X = X/ (60**2)
    X.shape = (-1, 1)
    x0 = np.ones(len(X)) * 1
    x0.shape = (-1,1)
    in_data = np.hstack((X, x0))
    y = y_CW[:,0]
    # X = 2*((X-X.min())/(X.max()-X.min()))-1


    #We make a neural net with 2 hidden layers, 20 neurons in each, using logistic activation
    #functions.
    # param=((1,0,0),(10, expit, logistic_prime),(10, expit, logistic_prime),(1,identity, identity_prime))
    param = ((2, 0, 0), (15, hyp_tan, hyp_tan_prime), (15, hyp_tan, hyp_tan_prime), (1, identity, identity_prime))

    #Set learning rate.
    rates = [0.001]
    predictions=[]
    for rate in rates:
        N=NeuralNetwork(in_data,y,param)
        N.train(100, learning_rate=rate)
        predictions.append([rate, N.predict(in_data)])
    fig, ax = plt.subplots(1, 1)
    if plots:
        ax.plot(X, y, label='Sine', linewidth=2, color='black')
        for data in predictions:
            ax.plot(X, data[1], label="Learning Rate: "+str(data[0]))
        ax.legend()
        plt.show()


if __name__ == '__main__':
    main()
