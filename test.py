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

class Asymptotic_Belief(Belief):
    def bayesian_update(self, x, y):
        # Log Likelihood
        log_likelihood = lambda theta : np.log( self.correct(theta, x) if y==1 else 1-self.correct(theta, x) )
        self.log_likelihood_t.append(log_likelihood)
        self.x_t.append(x)
        self.y_t.append(y)

        func = self.neg_log_posterior
        hessian_func = nd.Hessian(func) #

        # Laplace Approximation
        self.theta_hat = np.zeros(self.dim)
        self.neg_hessian = hessian_func(self.theta_hat)

        self.theta_hat_t.append(self.theta_hat) # record theta_hat, hessian
        self.neg_hessian_t.append(self.neg_hessian)
        self.t += 1

        self.explain()

Test = Exam(gradient_based, linear_intercept, 4, np.ones(4), 'GB_Asym', verbose=1)
Test.Belief = Asymptotic_Belief(linear_intercept, 4, verbose=1)

Test.exam()