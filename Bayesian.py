import random
import numpy as np
import pandas as pd

import pymc3 as pm
from scipy import optimize

def normalize(belief):   return belief/sum(belief)

class Belief():
    def __init__(self, dim, correct, num=10000):
        self.dim, self.correct = dim, correct # set correct rate function

        self.num, self.samples = num, np.random.multivariate_normal(np.zeros(dim), np.eye(dim), num) # sample by initial Gaussian
        self.belief = normalize( np.ones(self.P.shape[0]) ) # initial belief, uniform 
        
        self.t, self.theta_hat_t, self.Sigma_t = 0, [], [] # record Sigma at each step
        self.laplace_approx()

    def monte_carlo(self, x, y):
        self.belief
        with pm.Model() as model:
            mu = pm.Empirical("mu", self.samples)
            obs = pm.DensityDist("obs", logp=(self.correct if y==1 else 1-self.correct), observed={"theta":mu,"x":x})
            self.samples = pm.sample()
        self.model = model

    def laplace_approx(self):
        self.theta_hat = pm.find_MAP(model=self.model, method='L-BFGS-B') # 아니면 그냥 최대값 찾기
        self.Sigma = pm.find_hessian(self.theta_hat, vars=[mu])

        self.theta_hat_t[self.t], self.Sigma_t[self.t] = self.theta_hat, self.Sigma
        self.t += 1

    def initialize(self): # clear history
        self.belief = normalize( np.ones(self.P.shape[0]) )
        self.t, self.theta_hat_t, self.Sigma_t = 0, [], []
        self.laplace_approx()

    def imagine_update(self, x, y): # expect next update
        r_hat = self.correct(self.sample, x) # TODO : theta_hat이 아니라 각 sample의 정답률인 거 같음 -> 어 그러면 belief는 필요가 없어지는데?
        return normalize( self.belief * (r_hat if y == True else (1-r_hat)) ) # return next belief

    def update(self, x, y): # realization of update
        self.belief = self.imagine_update(self, x, y)
        self.monte_carlo(x, y)
        self.laplace_approx()
