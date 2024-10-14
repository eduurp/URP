import sys
import datetime
import time
sys.path.append(r'Z:\URP')

from utils.base import *
from utils.realworld import *
from utils.tools import *

import numpy as np
import os

Algo_name = "BALD"
Grade = 7

for model_name in ["LN"] :
    if model_name == "LN" :
        model = LN
    else :
        model = MIRT

    Algo_list = [BALD, DURM, MaxEnt, VarRatio, MeanStd]
    Algo_name_list = ["BALD", "DURM", "MaxEnt", "VarRatio", "MeanStd"]
    algorithm = Algo_list[Algo_name_list.index(Algo_name)]
    dim = 4

    dir_p = 'Demo/Grade' + str(Grade)
    R = np.load(dir_p + "/params_realworld/" + model_name+ "_Dataset.npy")
    P = np.load(dir_p +"/params_realworld/Theta_" +  model_name + "_" + str(dim) + ".npy")
    Q = np.load(dir_p +"/params_realworld/Query_" +  model_name + "_" + str(dim) + ".npy")
    PQI, max_query_num = return_PQI(R)

    ### Gaussian Mixture model case ###
    '''
    gmm_mu = np.load(dir_p + "/params_realworld/gmm_mu_" + str(dim) + "dim_" + model_name + ".npy")
    gmm_pi = np.load(dir_p + "/params_realworld/gmm_pi_" + str(dim) + "dim_" + model_name + ".npy")
    gmm_cov_inv = np.load(dir_p + "/params_realworld/gmm_inv_" + str(dim) + "dim_" + model_name + ".npy")
    '''
    
    num_t = max_query_num # num_t <= max_query_num
    MAP_0 = np.mean(P,axis = 0)

    nHess_0 = np.linalg.inv(np.cov(P.T))

    exp = Experiment_Realworld(algorithm, dim, model, num_t, MAP_0, nHess_0, R, P, Q, PQI)
    ### Gaussian Mixture model case ###
    '''
    exp = Experiment_Realworld(algorithm, dim, model, num_t, MAP_0, nHess_0, R, P, Q, PQI,prior_set = 'Gmm',
                           gmm_K=3,gmm_mu = gmm_mu, gmm_pi = gmm_pi, gmm_cov_inv= gmm_cov_inv)
    '''
    
    current_time = datetime.datetime.now().strftime('%y%m%d-%H-%M')
    folder_dir = dir_p + "/Grade" + str(Grade) + "_realworld_" + model_name + "_" + Algo_name
    create_directory(folder_dir)

    pop_num = len(P)//10

    for i in tqdm(range(pop_num)) :
        data = exp.do(i)
        file_path = folder_dir + "/" + str(i) + "_" + str(dim) + "dim_" + model_name + "_" + Algo_name + ".npy"
        np.save(file_path, data, allow_pickle = True)