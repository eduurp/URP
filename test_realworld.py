import sys
import datetime
import time
sys.path.append(r'Z:\URP')

from utils.base import *
from utils.realworld import *
from utils.tools import *

import numpy as np
import os

model_name = "LN" # "MIRT" 
if model_name == "LN" :
    model = LN
else :
    model = MIRT

Algo_name = "BALD"

Algo_list = [BALD, DURM, MaxEnt, VarRatio, MeanStd]
Algo_name_list = ["BALD", "DURM", "MaxEnt", "VarRatio", "MeanStd"]
algorithm = Algo_list[Algo_name_list.index(Algo_name)]

dim = 4

R = np.load("./params_realworld/" + model_name+ "_Dataset.npy")
P = np.load("./params_realworld/Theta_" +  model_name + "_" + str(dim) + "_MSE.npy")
Q = np.load("./params_realworld/Query_" +  model_name + "_" + str(dim) + "_MSE.npy")
PQI, max_query_num = return_PQI(R)

num_t = 10 # num_t <= max_query_num

MAP_0 = np.mean(P,axis = 0)
nHess_0 = np.linalg.inv(np.cov(P.T))

# num_population <= len(P)
exp = Experiment_Realworld(algorithm, dim, model, num_t, MAP_0, nHess_0, R, P, Q, PQI)
current_time = datetime.datetime.now().strftime('%y%m%d-%H-%M')
create_directory(current_time)

pop_num = 10 # pop_num < len(P)
for i in range(pop_num) :
    print(len(PQI[i]))
    print(Q[PQI[i]].shape)
    data = exp.do(i)
    file_path = "./" + current_time + "/" + str(i+1) + "_" + str(dim) + "dim_" + model_name + "_" + Algo_name + ".npy"
    np.save(file_path, data, allow_pickle = True)
