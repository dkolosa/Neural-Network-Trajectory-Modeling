#!/usr/bin/python3

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
import time
from net_demo import *

Re = 6378
mu = 398600
hr_to_sec = 60 ** 2
a = 300 + Re  # semi-major axis
n = np.sqrt(mu / a ** 3)  # mean motion circular orbit

def main():
    """Solving the linear CW equations given a set of state vectors
       for a target and chaser vehicle"""

    # delta_r0 = [.5,0,.2]
    delta_r0 = [[0.0, 0.0, 0.0], [.80, 0.30, 0.40], [0.10, 0.20, 0.10]]

    delta_r0_std = np.asarray(delta_r0)
    deltar0_test = [.4, .4, .5]
    tfinal = 2*np.pi/n*1
    time_span = np.arange(0, tfinal, 2)

    x_CW = np.zeros((len(time_span), 6))
    xy_CW = np.zeros((len(time_span), 6))
    i = 0
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
    for r0 in delta_r0:
        y_CW = diffCW(time_span, a, n, r0)

        x_CW[:,i] = y_CW[:,0]
        xy_CW[:,i] = y_CW[:,1]
        i += 1

    print('Generated training data\n')

    y_test_CW = diffCW(time_span, a, n, deltar0_test)
    # y_test = y_test_CW[:,0]
    print('Generated test data\n Beginning training...')
    test_regression(x_CW, xy_CW, delta_r0, time_span, y_test_CW, deltar0_test, plots=True)

    # Use the CW equations as input and output for a neural network
    # test_regression(x_CW, delta_r0, time_span, True)

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


def diffCW(time_span, a, n, delta_r0):
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
    # delta_r0 = [0.0, 0.0, 0.0]
    delta_v0 = [0, -0.01, .56]
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


def test_regression(x_CW, xy_CW, delta_r0, X, y_test, deltar0_test, plots=False):
    """
    Creates, trains, and tests a neural network
    :param x_CW: the output dataset (used for training)
    :param delta_r0: initial conditions of the training set
    :param X: time span of the dataset (input)
    :param y_test: the test data of the CW equation
    :param plots: generate the plots of training and test data
    :return: null
    """
    # create the data.
    # n = 500   # Number of points
    # X = np.linspace(0, 2*np.pi, num=n)  # Function range (time span)
    # X.shape = (n, 1)
    # y = np.sin(X)

    # Process training data for neural network
    X = X / (60**2)
    X.shape = (-1, 1)
    y = x_CW
    yy = xy_CW
    x0 = np.ones((len(X), 1))


    x0_test = np.ones((len(X), 1)) * deltar0_test[0]
    y0_test = np.ones((len(X), 1)) * deltar0_test[1]
    z0_test = np.ones((len(X), 1)) * deltar0_test[2]

    # x0_test.shape = (-1, 1)
    test_set = np.hstack((X, x0_test))
    test_sety = np.hstack((X, y0_test))
    test_setz = np.hstack((X, z0_test))


    # make a neural net with 2 hidden layers, 20 neurons in each, using hyperbolic tan activation
    # functions.
    # param=((1,0,0),(10, expit, logistic_prime),(10, expit, logistic_prime),(1,identity, identity_prime))
    # param = ((1, 0, 0), (40, hyp_tan, hyp_tan_prime), (40, hyp_tan, hyp_tan_prime), (1, identity, identity_prime))
    # param = ((3, 0, 0), (43, hyp_tan, hyp_tan_prime), (43, hyp_tan, hyp_tan_prime), (1, identity, identity_prime))

    param = ((2, 0, 0), (12, hyp_tan, hyp_tan_prime), (12, hyp_tan, hyp_tan_prime), (1, identity, identity_prime))


    param_y = ((2, 0, 0), (80, logistic, logistic_prime), (80, logistic, logistic_prime), (1, identity, identity_prime))
    #
    # param_z = ((2, 0, 0), (60, hyp_tan, hyp_tan_prime), (60, hyp_tan, hyp_tan_prime),
    #            (60, hyp_tan, hyp_tan_prime), (1, identity, identity_prime))

    #Set learning rate.
    rates = [0.005]
    predictions = []
    predictions_y = []
    predictions_z = []
    predictions_test, predictions_y_test, predictions_z_test = [], [], []

    # mse_test = np.empty(len(X))
    j=0
    for rate in rates:
        epochs = 5
        for r0 in delta_r0:
            # train_input = x0[:, j] * r0
            # train_input.shape = (-1,1)
            # train = np.column_stack((X, x0 * delta_r0[0], x0 * delta_r0[1]))
            train = np.hstack((X,  x0 * r0[0]))
            train_y = np.hstack((X,  x0 * r0[1]))
            # train_z = np.hstack((X, x0 * r0[2]))

            if j == 0:  # Set up for the 1st time
                N=NeuralNetwork(train, y[:,0], param)
                N_y = NeuralNetwork(train_y, yy[:,0], param_y)
                # N_z = NeuralNetwork(train_z, y[:,2], param_z)

            # start_train = time.time()
            N.train(5, train, y[:,j], learning_rate=rate)
            N_y.train(3, train_y, yy[:,j], learning_rate=0.005)
            # N_z.train(5, train_z, y[:,2], learning_rate=0.001)

            print("initial cond: ", r0)
            j += 1

    # end_train = time.time()
    print('Testing network')
    predictions.append(N.predict(train))
    predictions_y.append(N_y.predict(train_y))
    # predictions_z.append(N_z.predict(train_z))

    predictions_test.append(N.predict(test_set))
    predictions_y_test.append(N_y.predict(test_sety))
    # predictions_z_test.append(N_z.predict(test_setz))


    # np.append(predictions,N.predict(train))
    # plt.figure(4)
    if plots:
        fig, ax = plt.subplots(1, 1)
        # ax.plot(X, y[:,0], label='x value', linewidth=2, color='black')
        # ax.plot(X, y[:,1], label='y value', linewidth=2, color='red')
        # ax.plot(X,y[:,2], label='z value', linewidth=2, color='blue')
        # ax.plot(X,np.asarray(predictions).flatten(), label="NN x")
        # ax.plot(X,np.asarray(predictions_y).flatten(), label="NN y")
        # ax.plot(X,np.asarray(predictions_z).flatten(), label="NN z")
        ax.plot(X, y_test[:, 0], label='x test', linewidth=3)
        ax.plot(X, y_test[:, 1], label='y test', linewidth=3)
        # ax.plot(X, y_test[:, 2], label='z test', linewidth=3)

        ax.plot(X, np.asarray(predictions_test).flatten(), label="NN x test")
        ax.plot(X, np.asarray(predictions_y_test).flatten(), label="NN y test")
        # ax.plot(X, np.asarray(predictions_z_test).flatten(), label="NN z test")

        # for data in predictions:
        #     ax.plot(X, data[1], label="NN x "+str(rate))
            # np.append(mse_test, data[1])
        # for data in predictions_y:
        #     ax.plot(X, data[1], label='NN y set')
        # for data in predictions_z:
        #     ax.plot(X, data[1], label='NN z set')

        ax.legend()
        plt.xlabel('Time (hr)')
        plt.ylabel('Position (km)')

        print('MSE training X: ', mean_squared_error(y[:,0], np.asarray(predictions).flatten()))
        print('r2 training X: ', r2_score(y[:,0], np.asarray(predictions).flatten()))

        print('MSE training Y: ', mean_squared_error(y[:,1], np.asarray(predictions_y).flatten()))
        print('r2 training Y: ', r2_score(y[:,1], np.asarray(predictions_y).flatten()))

        # print('MSE training Z: ', mean_squared_error(y[:,2], np.asarray(predictions_z).flatten()))



        print('MSE test X: ', mean_squared_error(y_test[:,0], np.asarray(predictions_test).flatten()))
        print('r2 test X: ', r2_score(y_test[:,0], np.asarray(predictions_test).flatten()))

        print('MSE test Y: ', mean_squared_error(y_test[:,1], np.asarray(predictions_y_test).flatten()))
        # print('MSE test Z: ', mean_squared_error(y_test[:,2], np.asarray(predictions_z_test).flatten()))

        # print(mean_squared_error(y[:,1], np.asarray(predictions_z).flatten())
        plt.show()


if __name__ == '__main__':
    main()
