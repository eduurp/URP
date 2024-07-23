import sys
import datetime
import time
sys.path.append(r'Z:\URP')

from utils.base import *
from utils.synthetic import *
from utils.tools import *

import numpy as np
import os

model_name = "LN" # "MIRT" 

if model_name == "LN" :
    alpha = 10
    f = LN
    Algo = DURM_LN(alpha)

else :
    f = MIRT
    Algo = DURM_MIRT()

iterations = 20
dim = 4
num_t = 10

true_theta = np.load("./params_synthetic/true_theta_" + str(dim) + "dim_" + model_name + ".npy")
MAP_0 = np.load("./params_synthetic/MAP_0_" + str(dim) + "dim_" + model_name + ".npy")
nHess_0 = np.load("./params_synthetic/nHess_0_" + str(dim) + "dim_" + model_name + ".npy")

exp = Experiment_Synthetic(Algo, dim, f, num_t, 
                           true_theta, MAP_0, nHess_0)

current_time = datetime.datetime.now().strftime('%y%m%d-%H-%M')
create_directory(current_time)

for iter in range(1, iterations+1) :
    data = exp.iter_t()
    data = np.array(data)
    file_path = "./" + current_time + "/" + str(iter) + "_" + str(dim) + "dim_" + model_name + ".npy"
    np.save(file_path, data, allow_pickle = True)