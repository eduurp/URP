import random
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

def normalize(belief):   return belief/sum(belief)
def sigmoid(x): return 1 / (1 + np.exp(-x))

class Belief():
    def __init__(self, correct, dim=4, verbose=False):
        self.verbose = verbose
        self.correct, self.dim = correct, dim # set correct rate function
        self.initialize()

    def initialize(self): # clear history
        self.theta_hat, self.hessian_inv = np.zeros(self.dim), np.eye(self.dim)
        self.t, self.theta_hat_t, self.hessian_inv_t = 1, [self.theta_hat], [self.hessian_inv]
        
        self.prior = lambda theta : multivariate_normal.pdf(theta, mean=self.theta_hat, cov=self.hessian_inv)
        self.log_likelihood_t = [self.prior] # record hessian_inv at each step
        self.neg_log_posterior = lambda theta : -sum(log_likelihood(theta) for log_likelihood in self.log_likelihood_t)

        self.explain()

    def bayesian_update(self, x, y):
        # Log Likelihood
        log_likelihood = lambda theta : np.log( self.correct(theta, x) if y==1 else 1-self.correct(theta, x) )
        self.log_likelihood_t.append(log_likelihood)

        # Laplace Approximation
        opt_result = minimize(self.neg_log_posterior, self.theta_hat)
        self.theta_hat = opt_result.x
        self.hessian_inv = opt_result.hess_inv

        self.theta_hat_t.append(self.theta_hat) # record theta_hat, hessian_inv
        self.hessian_inv_t.append(self.hessian_inv)
        self.t += 1

        self.explain()

    def explain(self):
        if self.verbose:
            print('---------------------------')
            print(f'step t : {self.t}')
            print(f'MAP : \n {self.theta_hat}')
            print(f'Hessian_inv : \n {self.hessian_inv}')
            print('---------------------------')

    # TODO : imagine, realization of update