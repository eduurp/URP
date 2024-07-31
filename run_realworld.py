import sys
import datetime
import time
sys.path.append(r'Z:\URP')

from utils.base import *
from utils.realworld import *
from utils.tools import *

import numpy as np
import os

model_name = "MIRT" # "MIRT" 
Algo_name = "BALD"


if model_name == "LN" :
    model = LN
else :
    model = MIRT

Algo_list = [BALD, DURM, MaxEnt, VarRatio, MeanStd]
Algo_name_list = ["BALD", "DURM", "MaxEnt", "VarRatio", "MeanStd"]
algorithm = Algo_list[Algo_name_list.index(Algo_name)]

dim = 4

R = np.load("./params_realworld/" + model_name+ "_Dataset.npy")
P = np.load("./params_realworld/Theta_" +  model_name + "_" + str(dim) + ".npy")
Q = np.load("./params_realworld/Query_" +  model_name + "_" + str(dim) + ".npy")
PQI, max_query_num = return_PQI(R)

num_t = max_query_num # num_t <= max_query_num

MAP_0 = np.mean(P,axis = 0)
nHess_0 = np.linalg.inv(np.cov(P.T))

# num_population <= len(P)
exp = Experiment_Realworld(algorithm, dim, model, num_t, MAP_0, nHess_0, R, P, Q, PQI)
current_time = datetime.datetime.now().strftime('%y%m%d-%H-%M')
# folder_dir = current_time + "_realworld_" + model_name + "_" + Algo_name
folder_dir = "realworld_" + model_name + "_" + Algo_name
create_directory(folder_dir)

pop_num = 50 # pop_num < len(P)
for i in tqdm(range(pop_num)) :
    # print(len(PQI[i]))
    # print(Q[PQI[i]].shape)
    data = exp.do(i)
    file_path = "./" + folder_dir + "/" + str(i+1) + "_" + str(dim) + "dim_" + model_name + "_" + Algo_name + ".npy"
    np.save(file_path, data, allow_pickle = True)