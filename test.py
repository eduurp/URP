from utils.model import *
from utils.base import *

import numpy as np
import os

with open('vrf_MIRT.txt', 'w') as file: pass

MyAlgo = PCM_MIRT()
Test = Experiment(MyAlgo, 4)
Test.make()

'''
base = "[231118 05-15-00] MIRT 4 100 10 10 PCM_MIRT"
_ = np.load(os.path.join(base, "nHess_0.npy"))
'''

'''
nHesses = []
for i in range(10000):
    A = np.random.rand(4, 4)
    nHesses.append(A@A.T)

np.save('init_nhesses.npy', nHesses)
'''