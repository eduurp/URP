import os
import re

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numdifftools as nd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


def normalize(belief):   return belief/sum(belief)
def sigmoid(x): return 1 / (1 + np.exp(-x))

def log_gaussian(x, mean, cov_inv):
    return -(1/2) * (x - mean).T @ (cov_inv) @ (x - mean)

class Belief():
    def __init__(self, correct, dim, verbose=0, init_theta_hat=None, init_neg_hessian=None):
        self.verbose = verbose
        self.initialize()

    def initialize(self): # clear history
        self.theta_hat, self.neg_hessian = self.init_theta_hat, self.init_neg_hessian
        self.t, self.theta_hat_t, self.neg_hessian_t = 1, [self.theta_hat], [self.neg_hessian]
        self.x_t, self.y_t = [], []
        
        self.log_prior = lambda theta : log_gaussian(theta, mean=self.init_theta_hat, cov_inv=self.init_neg_hessian)
        self.log_likelihood_t = [ self.log_prior ] # record hessian at each step
        self.neg_log_posterior = lambda theta : -sum(log_likelihood(theta) for log_likelihood in self.log_likelihood_t)

        self.explain()

    def bayesian_update(self, x, y):
        # Log Likelihood
        log_likelihood = lambda theta : np.log( self.correct(theta, x) if y==1 else 1-self.correct(theta, x) )
        self.log_likelihood_t.append(log_likelihood)
        self.x_t.append(x)
        self.y_t.append(y)

        func = self.neg_log_posterior
        gradient_func = nd.Gradient(func) # 
        hessian_func = nd.Hessian(func) #

        # Laplace Approximation
        opt_result = minimize(func, self.theta_hat, jac=gradient_func, hess=hessian_func, method='trust-ncg')
        self.theta_hat = opt_result.x
        self.neg_hessian = hessian_func(self.theta_hat)

        self.theta_hat_t.append(self.theta_hat) # record theta_hat, hessian
        self.neg_hessian_t.append(self.neg_hessian)
        self.t += 1

        self.explain()

    def explain(self):
        if self.verbose>1:
            print('------------------------------------------')
            print(f'[ t : {self.t} ]')
            if self.t>1: print(f'y : {self.y_t[-1]}')
            print(f'MAP : \n {self.theta_hat}')
            print(f'Determinant : \n {np.linalg.det(self.neg_hessian)}')
            print(f'Eigenvalues : \n {np.linalg.eig(self.neg_hessian)[0]}')
            print('------------------------------------------')

    # TODO : imagine, realization of update


class Query():
    def __init__(self, query, correct, dim, true_theta, answers, verbose=1, init_theta_hat=None, init_neg_hessian=None):
        self.verbose = verbose
        self.Belief = Belief(correct, dim, verbose, init_theta_hat, init_neg_hessian)
        
        self.answers = answers
        self.num_t, self.num_seed = self.answers.shape[1], self.answers.shape[0]
        self.seed, self.theta_hat_seed_t, self.neg_hessian_seed_t = 0, [], []
        self.x_seed_t, self.y_seed_t = [], []
        self.correct_rate_t, self.correct_rate_seed_t = [], []

        self.query = query
        self.true_theta = true_theta

    def result(self, x, t, seed):
        correct_rate = self.Belief.correct(self.true_theta, x)
        self.correct_rate_t.append(correct_rate)
        if self.verbose>1:
            print(f'rate : {correct_rate}')
        return 1 if self.answers[seed][t] < correct_rate else 0
    
    def exam(self):
        if self.verbose>0:
            print(f' num_t : {self.num_t}')
            print(f' num_seed : {self.num_seed}')

        while self.seed<self.num_seed:
            self.Belief.initialize()
            self.correct_rate_t = []

            while self.Belief.t < self.num_t:
                x = self.query(self.Belief)
                y = self.result(x, self.Belief.t, self.seed)
                self.Belief.bayesian_update(x, y)

            self.x_seed_t.append(self.Belief.x_t)
            self.y_seed_t.append(self.Belief.y_t)
            self.correct_rate_seed_t.append(self.correct_rate_t)

            self.theta_hat_seed_t.append(self.Belief.theta_hat_t)
            self.neg_hessian_seed_t.append(self.Belief.neg_hessian_t)
            self.seed += 1

            self.explain()
            self.save()

    def explain(self):
        if self.verbose>0:
            print('------------------------------------------')
            print(f'[ seed : {self.seed}/{self.num_seed} ]')
            print(f'mean correct rate : {np.mean(self.correct_rate_seed_t[-1])}')
            for t, cov_det in enumerate([1/np.linalg.det(neg_hessian) for neg_hessian in self.neg_hessian_seed_t[-1]]):
                print(f't, det : {t}, {cov_det}')
            print('------------------------------------------')

    def save(self):
        pass