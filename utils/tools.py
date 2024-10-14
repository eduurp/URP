import datetime
import time
import os
import json

import numpy as np

def create_directory(directory_name):
    try:
        # make directory
        os.makedirs(directory_name)
        print(f"Directory successfully created : {directory_name}")
    except FileExistsError:
        print(f"Directory already existed : {directory_name}")
    except Exception as e:
        print(f"Erorr occured : {e}")

def return_PQI(R) :
    obs = ~np.isnan(R)
    possible_query_idx = []
    for i in range(obs.shape[0]):
        true_indices = np.where(obs[i])
        possible_query_idx.append(true_indices[0])
            
    PQIs = []

    for i in range(len(possible_query_idx)) :
        PQIs.append(possible_query_idx[i].tolist())
    PQI = PQIs
    max_query_num = np.min([np.sum(obs[j]) for j in range(len(obs))])
    return PQI, max_query_num
