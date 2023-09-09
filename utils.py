import os
import re

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import multivariate_normal


def normalize(belief):   return belief/sum(belief)
def sigmoid(x): return 1 / (1 + np.exp(-x))


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
        self.theta_hat, self.hessian_inv = np.zeros(self.dim), np.eye(self.dim)
        self.t, self.theta_hat_t, self.hessian_inv_t = 1, [self.theta_hat], [self.hessian_inv]
        self.x_t, self.y_t = [None], [None]
        
        self.prior = lambda theta : multivariate_normal.pdf(theta, mean=self.theta_hat, cov=self.hessian_inv)
        self.log_likelihood_t = [self.prior] # record hessian_inv at each step
        self.neg_log_posterior = lambda theta : -sum(log_likelihood(theta) for log_likelihood in self.log_likelihood_t)

        self.explain()

    def bayesian_update(self, x, y):
        # Log Likelihood
        log_likelihood = lambda theta : np.log( self.correct(theta, x) if y==1 else 1-self.correct(theta, x) )
        self.log_likelihood_t.append(log_likelihood)
        self.x_t.append(x)
        self.y_t.append(y)

        # Laplace Approximation
        opt_result = minimize(self.neg_log_posterior, self.theta_hat)
        self.theta_hat = opt_result.x
        self.hessian_inv = opt_result.hess_inv

        self.theta_hat_t.append(self.theta_hat) # record theta_hat, hessian_inv
        self.hessian_inv_t.append(self.hessian_inv)
        self.t += 1

        self.explain()

    def explain(self):
        if self.verbose>1:
            print('---------------------------')
            print(f'step t : {self.t}')
            print(f'MAP : \n {self.theta_hat}')
            print(f'Hessian_inv : \n {self.hessian_inv}')
            print('---------------------------')

    # TODO : imagine, realization of update


class Query():
    def __init__(self, query, correct, dim, true_theta, verbose=1, file_num=None):
        self.verbose = verbose
        file_dict = {int(re.search(r"answers_\d+_\d+_(\d+).npy", f).group(1)): f for f in os.listdir('.') if re.match(r"answers_\d+_\d+_(\d+).npy", f)}
        self.answers = np.load(file_dict[max(file_dict.keys())] if not file_num else file_dict[file_num])

        self.initialize(query, true_theta)
        self.Belief = Belief(correct, dim, verbose)
        
        self.num_t, self.num_seed = self.answers.shape[1], self.answers.shape[0]
        self.seed, self.theta_hat_seed_t, self.hessian_inv_seed_t = 0, [], []
        self.x_seed_t, self.y_seed_t = [], []
        
    def initialize(self, query, true_theta):
        self.query = query
        self.true_theta = true_theta

    def result(self, x, t, seed):
        return 0 if self.answers[seed][t] < self.Belief.correct(self.true_theta, x) else 1
    
    def exam(self):
        if self.verbose>0:
            print(f' num_t : {self.num_t}')
            print(f' num_seed : {self.num_seed}')

        while self.seed<self.num_seed:
            self.Belief.initialize()

            while self.Belief.t < self.num_t:
                x = self.query(self.Belief)
                y = self.result(x, self.Belief.t, self.seed)
                self.Belief.bayesian_update(x, y)

            self.x_seed_t.append(self.Belief.x_t)
            self.y_seed_t.append(self.Belief.y_t)

            self.theta_hat_seed_t.append(self.Belief.theta_hat_t)
            self.hessian_inv_seed_t.append(self.Belief.hessian_inv_t)
            self.seed += 1

            self.explain()

    def explain(self):
        if self.verbose>0:
            print(f'step seed : {self.seed}/{self.num_seed}')

    def save(self):
        determinants = [1/np.linalg.det(hessian_inv) for hessian_inv in self.hessian_inv_seed_t[0]]

        # 그래프 그리기
        plt.plot(range(len(determinants)), determinants)
        plt.xlabel("Index of Matrix")
        plt.ylabel("Determinant Value")
        plt.title("Determinants of Hessian Inverse Matrices")
        plt.show()

def ones_query(Belief):
    return np.ones(Belief.dim)/20000

def linear(theta, x):
    return sigmoid(theta.T@x)

Test = Query(ones_query, linear, 4, np.ones(4))
Test.exam()
Test.save()
print(Test.y_seed_t[0])
print(Test.theta_hat_seed_t[0])
print(Test.hessian_inv_seed_t[0])

# 값이 날아간다는 사실 발견 : 왜냐면 클 수록 맞을 확률이 늘어나니까 (하나만 맞춰도 유석주라고 생각, 그래서 Bayesian한건데도 Prior가 실패했다는 뜻)