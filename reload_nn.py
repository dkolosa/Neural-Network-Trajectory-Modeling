import numpy as np
import pickle
import matplotlib.pyplot as plt
from net_demo import *

Re = 6378
mu = 398600
sec_to_hr = 60 ** 2
a = 300 + Re  # semi-major axis
n = np.sqrt(mu / a ** 3)  # mean motion circular orbit
tfinal = 2*np.pi/n*1
time_span = np.arange(0, tfinal, 2)

with open('NN_x_model.pkl', 'rb') as fid:
    NN_x = pickle.load(fid)

with open('NN_y_model.pkl', 'rb') as fid:
    NN_y = pickle.load(fid)

with open('NN_z_model.pkl', 'rb') as fid:
    NN_z = pickle.load(fid)