import numpy as np
from Utils import sigmoid, log_gaussian, MIRT, LN, Record, Belief, SIR
from Utils import DURM_LN, DURM_MIRT, Experiment_Synthetic, Run_Synthetic
from Utils import DURM, BALD, Experiment_Realworld, Run_RealWorld

##
''' Synthetic Case Experiments '''
#'''
dim = 4
alpha = 10
model_name = 'LN'
true_theta = np.array([0.9, 0.3, 0.3, 0.1])
#true_theta = np.array([0, 0, 0, 0])
num_t = 500
MAP_0 = np.array([1.0, 0.0, 0.6, 1.0])
nHess_0 = np.array([
    [ 0.58396333, -0.36175317,  0.08181892, -0.01999662],
    [-0.36175317,  0.47667193,  0.02797142, -0.05730899],
    [ 0.08181892,  0.02797142,  0.2620719 ,  0.03275013],
    [-0.01999662, -0.05730899,  0.03275013,  0.27729285]
    ]) # Eigvals : 0.9, 0.3, 0.3, 0.1
iterations = 10

Run_Synthetic(model_name, dim, alpha, true_theta, num_t, MAP_0, nHess_0, iterations) 
#'''

''' Realworld Case Experiments '''
'''
algorithm_name = 'DURM'
model_name = 'MIRT'
dim = 4
pop_num = 30

Run_RealWorld(algorithm_name, model_name, dim, pop_num)
#'''