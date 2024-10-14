
############ Matrix Factorization Code ############
'''
This Code is for Matrix Factorization, with Bianary item-response table
Input : file_dir of dataset ==> R is the Binary item-response table with 0, 1, nan
Output : P, Q matrix ==> Q is for Item, P is for User
model_name = "LN" or "MIRT"
'''
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

file_dir = "./Figures/Figure_3/params_realworld/"
rank = 4
model_name = "LN"
epochs = 300

R = np.load(file_dir + model_name + '_Dataset.npy')
P, Q = learn_PQ(rank, model_name, epochs, R)
P, Q = P.numpy(), Q.numpy()

np.save(file_dir + "Theta_" + model_name + "_" + str(rank) + ".npy", P)
np.save(file_dir + "Query_" + model_name + "_" + str(rank) + ".npy", Q)