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

iterations = 100
dim = 4
num_t = 300

true_theta = np.load("Demo/params_synthetic/true_theta_" + str(dim) + "dim_" + model_name + ".npy")
MAP_0 = np.load("Demo/params_synthetic/MAP_0_" + str(dim) + "dim_" + model_name + ".npy")
nHess_0 = np.load("Demo/params_synthetic/nHess_0_" + str(dim) + "dim_" + model_name + ".npy")


exp = Experiment_Synthetic(Algo, dim, f, num_t, 
                           true_theta, MAP_0, nHess_0)

### Gaussian Mixture Model Case ###

'''
gmm_mu = np.load("Demo/params_synthetic/gmm_mu_"+ str(dim) + "dim_" + model_name + ".npy")
gmm_cov_inv = np.load("Demo/params_synthetic/gmm_inv_"+ str(dim) + "dim_" + model_name + ".npy")
gmm_pi = np.load("Demo/params_synthetic/gmm_pi_"+ str(dim) + "dim_" + model_name + ".npy")

exp = Experiment_Synthetic(Algo, dim, f, num_t, 
                           true_theta, MAP_0, nHess_0, prior_set='Gmm',
                           gmm_K = 3, gmm_mu= gmm_mu, gmm_cov_inv= gmm_cov_inv, gmm_pi= gmm_pi)
'''

current_time = datetime.datetime.now().strftime('%y%m%d-%H-%M')
folder_directory = current_time + '_' + model_name + '_synthetic'
create_directory(folder_directory)

for iter in tqdm(range(1, iterations+1)) :
    data = exp.iter_t()
    data = np.array(data)
    file_path = "./" + folder_directory + "/" + str(iter) + "_" + str(dim) + "dim_" + model_name + ".npy"
    np.save(file_path, data, allow_pickle = True)
