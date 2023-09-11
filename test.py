from utils import *
from record import *

def ones_query(Belief):
    return np.ones(Belief.dim)/4

def linear(theta, x):
    return sigmoid(theta.T@x)

def 
    val,vec = np.linalg.eig(Belief_test.neg_hessian)
    query = np.insert(query_bound_norm * vec[-1], len(vec[-1]),-query_bound_norm * vec[-1].T@Belief_test.theta_hat)

Test = Exam(ones_query, linear, 4, np.ones(4), verbose=1)
Test.exam()
Test.save()