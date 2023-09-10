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


def make_answers(num_t, num_seed, seed=0):
    np.random.seed(seed)
    answers = np.random.random((num_seed, num_t))

    numbers = [int(re.search(r"answers_\d+_\d+_(\d+).npy", f).group(1)) for f in os.listdir('.') if re.search(r"answers_\d+_\d+_(\d+).npy", f)]
    np.save(f"answers_{num_t}_{num_seed}_{max(numbers, default=0) + 1}.npy", answers)


class Belief():
    def __init__(self, correct, dim, verbose=0):
        self.verbose = verbose
        self.correct, self.dim = correct, dim # set correct rate function
        self.initialize()

    def initialize(self): # clear history
        self.init_theta_hat, self.init_neg_hessian = np.zeros(self.dim), np.eye(self.dim)

        self.theta_hat, self.neg_hessian = self.init_theta_hat, self.init_neg_hessian
        self.t, self.theta_hat_t, self.neg_hessian_t = 1, [self.theta_hat], [self.neg_hessian]
        self.x_t, self.y_t = [None], [None]
        
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
        gradient_func = nd.Gradient(func)
        hessian_func = nd.Hessian(func)

        # Laplace Approximation
        opt_result = minimize(func, self.theta_hat, jac=gradient_func, hess=hessian_func, method='trust-ncg')
        self.theta_hat = opt_result.x
        self.neg_hessian = -hessian_func(self.theta_hat)

        self.theta_hat_t.append(self.theta_hat) # record theta_hat, hessian
        self.neg_hessian_t.append(self.neg_hessian)
        self.t += 1

        self.explain()

    def explain(self):
        if self.verbose>1:
            print('------------------------------------------')
            print(f'[ t : {self.t} ]')
            print(f'y : {self.y_t[-1]}')
            print(f'MAP : \n {self.theta_hat}')
            print(f'-Hessian : \n {self.neg_hessian}')
            print('------------------------------------------')

    # TODO : imagine, realization of update


class Query():
    def __init__(self, query, correct, dim, true_theta, verbose=1, file_num=None):
        self.verbose = verbose
        file_dict = {int(re.search(r"answers_\d+_\d+_(\d+).npy", f).group(1)): f for f in os.listdir('.') if re.match(r"answers_\d+_\d+_(\d+).npy", f)}
        self.file_num = max(file_dict.keys()) if not file_num else file_num
        self.answers = np.load(file_dict[self.file_num])

        self.initialize(query, true_theta)
        self.Belief = Belief(correct, dim, verbose)
        
        self.num_t, self.num_seed = self.answers.shape[1], self.answers.shape[0]
        self.seed, self.theta_hat_seed_t, self.neg_hessian_seed_t = 0, [], []
        self.x_seed_t, self.y_seed_t = [], []
        self.correct_rate_t, self.correct_rate_seed_t = [], []
        
    def initialize(self, query, true_theta):
        self.query = query
        self.true_theta = true_theta

    def result(self, x, t, seed):
        correct_rate = self.Belief.correct(self.true_theta, x)
        self.correct_rate_t.append(correct_rate)
        if self.verbose>1:
            print(f'rate : {correct_rate}')
        return 0 if self.answers[seed][t] < correct_rate else 1
    
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

    def explain(self):
        if self.verbose>0:
            print('------------------------------------------')
            print(f'[ seed : {self.seed}/{self.num_seed} ]')
            print(f'mean correct rate : {np.mean(self.correct_rate_seed_t[-1])}')
            for t, cov_det in enumerate([1/np.linalg.det(neg_hessian) for neg_hessian in self.neg_hessian_seed_t[-1]]):
                print(f't, det : {t}, {cov_det}')
            print('------------------------------------------')

    def save(self, about='test'):
        numbers = [int(re.search(rf"exam_{self.num_t}_{self.num_seed}_{self.file_num}_(\d+).npy", f).group(1)) for f in os.listdir('.') if re.search(rf"exam_{self.num_t}_{self.num_seed}_{self.file_num}_(\d+).npy", f)]
        folder = f"exam_{self.num_t}_{self.num_seed}_{self.file_num}_{max(numbers, default=0) + 1}"
        os.makedirs(folder, exist_ok=True)

        np.save(os.path.join(folder, "x_seed_t.npy"), self.x_seed_t)
        np.save(os.path.join(folder, "y_seed_t.npy"), self.y_seed_t)
        np.save(os.path.join(folder, "correct_rate_seed_t.npy"), self.correct_rate_seed_t)
        np.save(os.path.join(folder, "theta_hat_seed_t.npy"), self.theta_hat_seed_t)
        np.save(os.path.join(folder, "neg_hessian_seed_t.npy"), self.neg_hessian_seed_t)

        # plot
        for seed, neg_hessian_t in enumerate(self.neg_hessian_seed_t):
            determinants = [1/np.linalg.det(neg_hessian) for neg_hessian in neg_hessian_t]

            plt.plot(range(len(determinants)), determinants)
            plt.xlabel("step t")
            plt.ylabel("det of covariance")
            plt.title(f"Rule : {about}")
            if self.verbose>1: plt.show()
            plt.savefig(os.path.join(folder, f"det_by_t_{seed}.png"))
            plt.close()

        