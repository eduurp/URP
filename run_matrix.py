import sys
import datetime
import time
import torch
sys.path.append(r'Z:\URP')

from utils.base import *
from utils.matrix import *
from utils.tools import *

import numpy as np
import os

file_dir = "./params_realworld/"
rank = 4
model_name = "LN"
epochs = 100

R = np.load(file_dir + model_name + '_Dataset.npy')
P, Q = learn_PQ(rank, model_name, epochs, R)
P, Q = P.numpy(), Q.numpy()

np.save("./params_realworld/Theta_" + model_name + "_" + str(rank) + ".npy", P)
np.save("./params_realworld/Query_" + model_name + "_" + str(rank) + ".npy", Q)