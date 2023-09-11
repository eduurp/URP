from utils import *

def ones_query(Belief):
    return np.ones(Belief.dim)/4

def linear(theta, x):
    return sigmoid(theta.T@x)

make_answers(100, 10)