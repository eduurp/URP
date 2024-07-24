import sys
import datetime
import time
sys.path.append(r'Z:\URP')

from utils.base import *
from utils.realworld import *
from utils.tools import *

import numpy as np
import os

model = LN
model_name = "LN"
Algo_name = MaxEnt
dim = 4
R = np.load("./params_realworld/" + model_name+ "_Dataset.npy")
P = np.load("./params_realworld/Theta_" +  model_name + "_" + str(dim) + "_MSE.npy")
Q = np.load("./params_realworld/Query_" +  model_name + "_" + str(dim) + "_MSE.npy")
PQI, max_query_num = return_PQI(R)

num_t = 10 # num_t <= max_query_num

MAP_0 = np.mean(P,axis = 0)
nHess_0 = np.linalg.inv(np.cov(P.T))


belief = Belief(model, dim, MAP_0, nHess_0)
pop_idx = 0
algo = VarRatio(LN)
algo.query(belief, Q)
algo = MaxEnt(LN)
algo.query(belief, Q)
algo = MeanStd(LN)
algo.query(belief, Q)