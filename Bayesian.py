import random
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

def normalize(belief):   return belief/sum(belief)
def sigmoid(x): return 1 / (1 + np.exp(-x))


class Belief():
    def __init__(self, correct, dim=4, num=10000, verbose=False):
        self.correct, self.dim, self.num = correct, dim, num # set correct rate function
        self.verbose = verbose
        self.initialize()

    def initialize(self): # clear history
        self.samples = np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim), self.num) # sample by initial Gaussian
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(self.samples)

        self.theta_hat, self.hessian_inv = np.zeros(self.dim), np.eye(self.dim)
        self.t, self.theta_hat_t, self.hessian_inv_t = 1, [np.zeros(self.dim)], [np.eye(self.dim)] # record hessian_inv at each step

        self.explain()

    def bayesian_update(self, x, y):
        # KDE sampling
        log_likelihood = lambda theta : np.log( self.correct(theta, x) if y==1 else 1-self.correct(theta, x) )
        neg_log_posterior = lambda theta : -( log_likelihood(theta) + self.kde.score_samples(theta.reshape(1, -1)) )
        
        self.samples = self.kde.sample(self.num) # resampling
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(self.samples)

        # Laplace Approximation
        opt_result = minimize(neg_log_posterior, self.theta_hat)
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

def linear(theta, x):
    return sigmoid(theta.T@x)

Belief_test = Belief(linear, verbose=True)
Belief_test.bayesian_update(np.ones(4), 1)

