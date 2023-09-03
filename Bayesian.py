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

        self.num, self.samples = num, np.random.multivariate_normal(np.zeros(dim), np.eye(dim), num) # sample by initial Gaussian
        self.belief = normalize( np.ones(self.num) ) # initial belief, uniform 
        
        self.t, self.theta_hat_t, self.Sigma_t = 1, [np.zeros(dim)], [np.eye(dim)] # record Sigma at each step

    def monte_carlo(self, x, y):
        with pm.Model() as model:
            mu = pm.Empirical("mu", self.samples)
            obs = pm.DensityDist("obs", logp=(self.correct if y==1 else 1-self.correct), observed={"theta":mu,"x":x})
            self.samples = pm.sample()
        self.model = model

    def laplace_approx(self):
        self.theta_hat = pm.find_MAP(model=self.model, method='L-BFGS-B')
        self.Sigma = pm.find_hessian(self.theta_hat, vars=[mu])

        self.theta_hat_t.append(self.theta_hat)
        self.Sigma_t.append(self.Sigma)
        self.t += 1

    def initialize(self): # clear history
        self.belief = normalize( np.ones(self.num) )
        self.t, self.theta_hat_t, self.Sigma_t = 0, [], []
        self.laplace_approx()

    def imagine_update(self, x, y): # expect next update
        r_hat = self.correct(self.sample, x) # TODO : theta_hat이 아니라 각 sample의 정답률인 거 같음 -> 어 그러면 belief는 필요가 없어지는데?
        return normalize( self.belief * (r_hat if y == True else (1-r_hat)) ) # return next belief

    def update(self, x, y): # realization of update
        self.belief = self.imagine_update(x, y)
        self.monte_carlo(x, y)
        self.laplace_approx()

def linear(theta, x):
    return sigmoid(theta.T@x)

Belief_test = Belief(linear)
Belief_test.update(np.ones(4), 1)
