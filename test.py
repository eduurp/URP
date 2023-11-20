from utils.model import *
from utils.base import *

import numpy as np
import os

# A = np.random.rand(4, 4)

base = "[231118 05-15-00] MIRT 4 100 10 10 PCM_MIRT"

MyAlgo = PCM_LN(1/2)
Test = Experiment(MyAlgo, LN, 4, nHess_0=np.load(os.path.join(base, "nHess_0.npy")))

Test.make()
