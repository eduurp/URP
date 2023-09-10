from utils import *

def ones_query(Belief):
    return np.ones(Belief.dim)/4

def linear(theta, x):
    return sigmoid(theta.T@x)

Test = Query(ones_query, linear, 4, np.ones(4), verbose=1)
Test.exam()
Test.save()