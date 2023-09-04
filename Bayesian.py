import random
import numpy as np
import pandas as pd

import pymc3 as pm
from scipy import optimize

def normalize(belief):   return belief/sum(belief)
def sigmoid(x): return 1 / (1 + np.exp(-x))

class Belief():
    def __init__(self, correct, dim=4, num=10000):
        self.dim, self.correct = dim, correct # set correct rate function

        self.model = pm.Model()
        self.num, self.samples = num, np.random.multivariate_normal(np.zeros(dim), np.eye(dim), num) # sample by initial Gaussian
        
        self.t, self.theta_hat_t, self.Sigma_t = 1, [np.zeros(dim)], [np.eye(dim)] # record Sigma at each step

    def monte_carlo(self, x, y):
        with self.model:
            mu = pm.Empirical("mu", self.samples) # 이 부분이 바뀌어야하는데 
            likelihood = lambda theta : self.correct(theta, x) if y==1 else 1-self.correct(theta, x)
            obs = pm.Bernoulli('likelihood', p=likelihood, observed=self.samples)
            self.samples = pm.sample()

    def laplace_approx(self):
        with self.model:
            self.theta_hat = pm.find_MAP(method='L-BFGS-B')
            self.Sigma = pm.find_hessian(self.theta_hat, vars=[mu])

        self.theta_hat_t.append(self.theta_hat)
        self.Sigma_t.append(self.Sigma)
        self.t += 1

    def initialize(self): # clear history
        self.belief = normalize( np.ones(self.num) )
        self.t, self.theta_hat_t, self.Sigma_t = 0, [], []
        self.laplace_approx()

    def imagine_update(self, x, y): # expect next update
        r_hat = np.array([self.correct(sample, x) for sample in self.samples])
        return normalize( self.belief * (r_hat if y == True else (1-r_hat)) ) # 이 부분이 Belief로 바뀔듯

    def update(self, x, y): # realization of update
        self.belief = self.imagine_update(x, y)
        self.monte_carlo(x, y)
        self.laplace_approx()

def linear(theta, x): # TODO : 여러 개 들어오면 여러개 내 보내줘야함 or r_hat이 list이거나
    return sigmoid(theta.T@x)

Belief_test = Belief(linear)
Belief_test.update(np.ones(4), 1)
