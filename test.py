from utils import *
from record import *

bound = 1/2 # ???

def fixed_query(Belief):
    return np.ones(Belief.dim)/4

def linear(theta, x):
    return sigmoid(theta.T@x)

def gradient_based(Belief):
    global bound
    val, vec = np.linalg.eig(Belief.neg_hessian)
    return np.append(bound * vec[-1], -bound * vec[-1].T@Belief.theta_hat)

def linear_intercept(theta, x):
    return sigmoid(theta.T@x[:-1] + x[-1])

Test = Exam(gradient_based, linear_intercept, 4, np.ones(4), 'GB', verbose=1)
Test.exam()