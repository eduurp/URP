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
    return np.append(bound * vec[:, np.argmin(val)], -bound * vec[:, np.argmin(val)].T@Belief.theta_hat)

def linear_intercept(theta, x):
    return sigmoid(theta.T@x[:-1] + x[-1])

'''
A = np.random.rand(4, 4)
A = A@A.T
det = np.linalg.det(A)
A = A/(det**(1/4))
'''

A = np.diag(np.array([1/2, 1, 1, 2]))

Test = Exam(gradient_based, linear_intercept, 4, np.zeros(4), 'GB3', verbose=1, init_theta_hat=np.zeros(4), init_neg_hessian=A)
Test.exam()
