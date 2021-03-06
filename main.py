#!/usr/bin/python3

import numpy as np
from scipy.integrate import odeint
from sklearn.neural_network import MLPRegressor
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import time
import random
from net_demo import *
#matplotlib.use('GTKAgg')

Re = 6378
mu = 398600
sec_to_hr = 60 ** 2
a = 300 + Re  # semi-major axis
n = np.sqrt(mu / a ** 3)  # mean motion circular orbit
cw_problems = 100
cw_test_problems = 6


def main():
    """Solving the linear CW equations given a set of state vectors
       for a target and chaser vehicle"""

    # delta_r0 = [.5,0,.2]
    delta_r0 = [[random.random() for i in range(3)] for j in range(cw_problems)]
    # delta_r0 = [[0.197, 0.145, 0.110], [0.574, 0.522, 0.648], [0.548, 0.375, 0.815],
    #             [0.836, 0.082, 0.260], [0.352, 0.867, 0.882], [0.866, 0.674, 0.558]]


    # delta_r0_std = np.asarray(delta_r0)
    # deltar0_test = [.4, .4, .2]

    # deltar0_test = [[random.random() for i in range(3)] for j in range(cw_test_problems)]

    deltar0_test = [[0.197, 0.145, 0.110], [0.574, 0.522, 0.648], [0.548, 0.375, 0.815],
                    [0.836, 0.082, 0.260], [0.352, 0.867, 0.882], [0.866, 0.674, 0.558]]

    tfinal = 2*np.pi/n*1
    time_span = np.arange(0, tfinal, 2)

    x_CW = np.zeros((len(time_span), cw_problems))
    xy_CW = np.zeros((len(time_span), cw_problems))
    xz_CW = np.zeros((len(time_span), cw_problems))

    x_test = np.zeros((len(time_span), cw_test_problems))
    y_test = np.zeros((len(time_span), cw_test_problems))
    z_test = np.zeros((len(time_span), cw_test_problems))
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
    print('Generating training data\n')
    i = 0
    for r0 in delta_r0:
        y_CW = diffCW(time_span, a, n, r0)

        x_CW[:,i] = y_CW[:,0]
        xy_CW[:,i] = y_CW[:,1]
        xz_CW[:,i] = y_CW[:,2]
        i += 1

    print('Generated training data\n')
    i = 0
    for r0_test in deltar0_test:
        y_test_CW = diffCW(time_span, a, n, r0_test)
        x_test[:,i] = y_test_CW[:,0]
        y_test[:,i] = y_test_CW[:,1]
        z_test[:,i] = y_test_CW[:,2]
        i += 1

    print('Generated test data\n Beginning training...')
    test_regression(x_CW, xy_CW, xz_CW, delta_r0, time_span, x_test, y_test, z_test, deltar0_test, plots=True)

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
    """
    Save the file contents
    :param file: name of the file to save to
    :param data: the object or data to save
    """

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
    delta_v0 = [0, -0.01, 0.56]
    # delta_r0 = [1.2, 2.3, 1.26]
    # delta_v0 = [0.31667, 0.11199, 1.247]
    delta_y0 = delta_r0 + delta_v0

    y_rel = odeint(CW_ode, delta_y0, time_span, args=(n,),
                   atol=abserr, rtol=relerr)

    # # target
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
    plt.subplot(1, 1, 1)
    plt.plot(time_span, y_rel[:, 3], label='x')
    plt.plot(time_span, y_rel[:, 4], label='y')
    plt.plot(time_span, y_rel[:, 5], label='z')
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


def test_regression(x_cw, xy_cw, xz_cw, delta_r0, X, x_test, y_test, z_test, deltar0_test, plots=False):
    """
    Creates, trains, and tests a neural network
    :param x_cw: the output dataset (used for training, x component)
    :param xy_cw: the output dataset (used for training, y component)
    :param xz_cw: the output dataset (used for training, z component)
    :param delta_r0: initial conditions of the training set
    :param X: time span of the dataset (input)
    :param y_test: the test data of the CW equation
    :param deltar0_test: the test initial positions
    :param plots: generate the plots of training and test data
    :return: null
    """
    # create the data.
    # n = 500   # Number of points
    # X = np.linspace(0, 2*np.pi, num=n)  # Function range (time span)
    # X.shape = (n, 1)
    # y = np.sin(X)

    z_max = xz_cw.max()

    y_norm = 200
    z_norm = 500

    # Process training data for neural network

    X = X / (60 ** 2)
    X.shape = (-1, 1)
    y = x_cw
    yy = xy_cw / y_norm
    yz = xz_cw / z_norm
    x0 = np.ones((len(X), 1))



    # make a neural net with 2 hidden layers, 20 neurons in each, using hyperbolic tan activation
    # functions.
    # param=((1,0,0),(10, expit, logistic_prime),(10, expit, logistic_prime),(1,identity, identity_prime))
    # param = ((1, 0, 0), (40, hyp_tan, hyp_tan_prime), (40, hyp_tan, hyp_tan_prime), (1, identity, identity_prime))
    # param = ((3, 0, 0), (43, hyp_tan, hyp_tan_prime), (43, hyp_tan, hyp_tan_prime), (1, identity, identity_prime))

    param = ((5, 0, 0), (20, hyp_tan, hyp_tan_prime), (20, hyp_tan, hyp_tan_prime), (1, identity, identity_prime))

    param_y = ((5, 0, 0), (35, hyp_tan, hyp_tan_prime), (35, hyp_tan, hyp_tan_prime), (1, identity, identity_prime))

    param_z = ((3, 0, 0), (45, hyp_tan, hyp_tan_prime), (45, hyp_tan, hyp_tan_prime), (1, identity, identity_prime))

    #Set learning rate.
    rates = [0.05]
    predictions = []
    predictions_y = []
    predictions_z = []
    predictions_test, predictions_y_test, predictions_z_test = [], [], []

    # mse_test = np.empty(len(X))

    # fig_train, ax_train = plt.subplots(1, 1)
    # xyMLP = MLPRegressor(hidden_layer_sizes=(30, 30), activation='tanh', solver='adam')

    j=0
    for rate in rates:
        epochs = 5
        j = 0
        for r0 in delta_r0:
            # train_input = x0[:, j] * r0
            # train_input.shape = (-1,1)
            # train = np.column_stack((X, x0 * delta_r0[0], x0 * delta_r0[1]))
            train = np.hstack((X,  x0 * r0[0], x0 * r0[1], x0*0, x0*-0.01))
            train_y = np.hstack((X,  x0 * r0[1], x0 * r0[0], x0*-0.01, x0*0))
            train_z = np.hstack((X, x0 * r0[2], x0*0.56))

            x_train =  y[:,j]
            y_train =  yy[:,j]
            z_train = yz[:,j]

            x_train.shape = (-1,1)
            y_train.shape = (-1,1)
            z_train.shape = (-1,1)

            train_xy = np.hstack((x_train, y_train))

            if j == 0:  # Set up for the 1st time
                N=NeuralNetwork(train, y[:,0], param)
                N_y = NeuralNetwork(train_y, yy[:,0], param_y)
                N_z = NeuralNetwork(train_z, yz[:,0], param_z)

            # N_y = NeuralNetwork(train_y, yy[:,0], param_y)

            # start_train = time.time()
            # while True:
            N.train(3, train, y[:,j], learning_rate=0.001)
            N_y.train(4, train_y, yy[:,j], learning_rate=0.05)
            N_z.train(5, train_z, yz[:,j], learning_rate=0.03)

            # n = xyMLP.fit(train, train_xy)

            print(j+1, " of ", cw_problems, " initial cond: [{0:.3f}, {1:.3f}, {2:.3f}]".format(r0[0], r0[1], r0[2]))
            print("MSE train X: {0:.5f}".format(mean_squared_error(y[:,j], np.asarray(N.predict(train)).flatten())))
            print("MSE train y: {0:.5f}".format(mean_squared_error(yy[:, j], np.asarray(N_y.predict(train_y)).flatten())))
            print("MSE train z: {0:.5f} \n".format(mean_squared_error(yz[:, j], np.asarray(N_z.predict(train_z)).flatten())))

            # print('r2 training Z: ', r2_score(yz[:, j], np.asarray(N_z.predict(train_z)).flatten()))

            j += 1

    # end_train = time.time()
    # predictions.append(N.predict(train))
    # predictions_y.append(N_y.predict(train_y))
    # predictions_z.append(N_z.predict(train_z))

    fig, ax = plt.subplots(1, 1)

    # fig, (ax_x, ax_y, ax_z) = plt.subplots(3, sharex=True)
    # ax_x = plt.subplot((2,2,1))
    # ax_y = plt.subplot((2,2,2))
    # ax_z = plt.subplot((2,2,3))

    print('Testing network')
    i = 0
    for r0_test in deltar0_test:

        predictions_test, predictions_y_test, predictions_z_test = [], [], []

        out_x = np.zeros(len(X))
        out_y = np.zeros(len(X))
        out_z = np.zeros(len(X))

        test_set = np.hstack((X, x0 * r0_test[0], x0*r0_test[1], x0*0.0, x0*-0.01))
        test_sety = np.hstack((X, x0 * r0_test[1], x0*r0_test[0], x0*-0.01, x0*0.0))
        test_setz = np.hstack((X, x0*r0_test[2], x0*0.56))

        # testMPL = xyMLP.predict(test_set)

        out_x = np.asarray(N.predict(test_set)).flatten()
        out_y = np.asarray(N_y.predict(test_sety)).flatten() * y_norm
        out_z = np.asarray(N_z.predict(test_setz)).flatten() * z_norm

        # ax_x.plot(X, x_test[:,i], label="x test " + str(i))
        # ax_y.plot(X, y_test[:,i], label="y test " + str(i))
        # ax_z.plot(X, z_test[:,i], label="z test " + str(i))

        ax.plot(X, x_test[:,i], label="x test " + str(i))
        ax.plot(X, y_test[:,i], label="y test " + str(i))
        ax.plot(X, z_test[:,i], label="z test " + str(i))

        # ax.plot(X, testMPL[:,0], label='NN x',linestyle='-.')
        # ax.plot(X, testMPL[:,1], label='NN y',linestyle='-.')
        # ax.plot(X, testMPL[:,2], label='NN z',linestyle='-.')
        # ax_x.plot(X, out_x, label="NN x test " + str(i), linestyle='-.')
        # ax_y.plot(X, out_y, label="NN y test " + str(i), linestyle='dotted')
        # ax_z.plot(X, out_z, label="NN z test " + str(i), linestyle='dashed')

        ax.plot(X, out_x, label="NN x test " + str(i), linestyle='-.')
        ax.plot(X, out_y, label="NN y test " + str(i), linestyle='dotted')
        ax.plot(X, out_z, label="NN z test " + str(i), linestyle='dashed')

        # print('Initial conditions (test): ', r0_test)
        print(i+1, " of ", cw_test_problems, " initial cond: [{0:.3f}, {1:.3f}, {2:.3f}]".format(r0_test[0], r0_test[1], r0_test[2]))

        print("MSE test X: {0:.5f}".format(mean_squared_error(x_test[:,i], out_x)))
        print("MSE test y: {0:.5f}".format(mean_squared_error(y_test[:,i], out_y)))
        print("MSE test z: {0:.5f}".format(mean_squared_error(z_test[:,i],out_z)))

        print('r2 test X: ', r2_score(x_test[:, i], out_x))
        print('r2 test y: ', r2_score(y_test[:, i], out_y))
        print('r2 test z: ', r2_score(z_test[:, i], out_z), "\n")

        # mean_squared_error(x_test[:, i].reshape((-1, 1)), testMPL[:, 0])
        # print("MSE test X: {0:.5f}".format(mean_squared_error(x_test[:, i].reshape((-1, 1)), testMPL[:, 0])))
        # print("MSE test y: {0:.5f}".format(mean_squared_error(y_test[:, i].reshape((-1, 1)), testMPL[:, 1])))

        i += 1

    #Store the matplotlib file
    plt.xlabel('time (hr)')
    plt.ylabel('position (km)')
    plt.title('Neural Network test output')
    pickle.dump(fig, open('FigNN_final4.fig.pkl', 'wb'))
    plt.show()
    pass

    # save the NN state
    pickle.dump(N, open('NN_x_model.pkl', 'wb'))
    pickle.dump(N_y, open('NN_y_model.pkl', 'wb'))
    pickle.dump(N_z, open('NN_z_model.pkl', 'wb'))

    # save the training and test inital conditions
    save_data('training_ini.csv', delta_r0)
    save_data('test_ini.csv', deltar0_test)

    # save the output training and test data
    save_data('train_outx.csv', x_cw)
    save_data('train_outy.csv', xy_cw)
    save_data('train_outz.csv', xz_cw)

    save_data('test_outx.csv', x_test)
    save_data('test_outy.csv', y_test)
    save_data('test_outz.csv', z_test)

    pass
    # np.append(predictions,N.predict(train))
    # plt.figure(4)
    # if plots:
    #     fig, ax = plt.subplots(1, 1)
    #     # ax.plot(X, y[:,0], label='x value', linewidth=2, color='black')
    #     # ax.plot(X, y[:,1], label='y value', linewidth=2, color='red')
    #     # ax.plot(X,y[:,2], label='z value', linewidth=2, color='blue')
    #     # ax.plot(X,np.asarray(predictions).flatten(), label="NN x")
    #     # ax.plot(X,np.asarray(predictions_y).flatten(), label="NN y")
    #     # ax.plot(X,np.asarray(predictions_z).flatten(), label="NN z")
    #     ax.plot(X, y_test[:, 0], label='x test', linewidth=3)
    #     ax.plot(X, y_test[:, 1], label='y test', linewidth=3)
    #     ax.plot(X, y_test[:, 2], label='z test', linewidth=3)
    #
    #
    #     ax.plot(X, np.asarray(predictions_test).flatten(), label="NN x test")
    #     ax.plot(X, np.asarray(predictions_y_test).flatten(), label="NN y test")
    #     ax.plot(X, np.asarray(predictions_z_test).flatten()*z_norm, label="NN z test")
    #
    #     # for data in predictions:
    #     #     ax.plot(X, data[1], label="NN x "+str(rate))
    #         # np.append(mse_test, data[1])
    #     # for data in predictions_y:
    #     #     ax.plot(X, data[1], label='NN y set')
    #     # for data in predictions_z:
    #     #     ax.plot(X, data[1], label='NN z set')
    #
    #     ax.legend()
    #     plt.xlabel('Time (hr)')
    #     plt.ylabel('Position (km)')
    #
    #     print('---Training---')
    #     print('MSE training X: ', mean_squared_error(y[:,0], np.asarray(predictions).flatten()))
    #     print('r2 training X: ', r2_score(y[:,0], np.asarray(predictions).flatten()))
    #
    #     print('MSE training Y: ', mean_squared_error(yy[:,0], np.asarray(predictions_y).flatten()))
    #     print('r2 training Y: ', r2_score(yy[:,1], np.asarray(predictions_y).flatten()))
    #
    #     print('MSE training Z: ', mean_squared_error(yz[:,0], np.asarray(predictions_z).flatten()))
    #     print('r2 training Z: ', r2_score(yz[:,2], np.asarray(predictions_z).flatten()))
    #
    #
    #     print('---Testing---')
    #     print('MSE test X: ', mean_squared_error(y_test[:,0], np.asarray(predictions_test).flatten()))
    #     # print('r2 test X: ', r2_score(y_test[:,0], np.asarray(predictions_test).flatten()))
    #
    #     print('MSE test Y: ', mean_squared_error(y_test[:,1], np.asarray(predictions_y_test).flatten()))
    #     print('MSE test z: ', mean_squared_error(y_test[:,2], np.asarray(predictions_z_test).flatten()*z_norm))
    #
    #     # print(mean_squared_error(y[:,1], np.asarray(predictions_z).flatten())
    #     plt.show()
    #
    #     pass


if __name__ == '__main__':
    main()
