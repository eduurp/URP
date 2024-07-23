# Utils #
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import itertools

from scipy.optimize import minimize
from scipy.linalg import sqrtm
import time

''' Fundemental Operations'''

# Functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def log_gaussian(x, mean, cov_inv):
    if np.array(x).ndim==1:  x = x.reshape(1,-1)
    return -(1/2) * np.sum((x - mean) @ (cov_inv) * (x - mean), axis=1)

# Models
def MIRT(ps, qs):
    ps_, qs_ = ps, qs
    if np.array(qs).ndim==1:  qs_ = [qs]
    if np.array(ps).ndim==1:  ps_ = [ps]
    r = np.prod(sigmoid(np.array([qs_] * len(ps_)) + np.array([ps_] * len(qs_)).swapaxes(0,1)), axis=2)
    return r

def LN(ps, qs):
    ps_, qs_ = ps, qs
    if np.array(qs).ndim==1:  qs_ = qs.reshape(1,-1)
    if np.array(ps).ndim==1:  ps_ = ps.reshape(1,-1)
    r = sigmoid(ps_@qs_.T[:-1] + qs_.T[1])
    return r

# Basic Pipelines
class Record():
    def __init__(self):
        self.typs = ['MAP', 'nHess', 'f', 'true_f', 'time', 'ent']
        self.data = { typ:[] for typ in self.typs }

    def merge(self, subRecord):
        for typ in self.typs:
            self.data[typ].append(subRecord.data[typ])

class Belief():
    """
    Class Representation of Bayesian Belief.
    """
    def __init__(self, f, dim, MAP_0, nHess_0):
        self.dim = dim # dimension
        self.f = f # set correct rate function
        
        if self.f == LN :
            self.func_name = 'LN'
        else :
            self.func_name = 'MIRT'

        self.MAP_0 = MAP_0
        self.nHess_0 = nHess_0
        self.ent_0 = 0
        self.LPrior = lambda theta : log_gaussian(theta, mean=self.MAP_0, cov_inv=self.nHess_0)
        
        self.sample_points = self.pre_sampling_points(num_sampling=3000)
        self.initialize(self.MAP_0, self.nHess_0)
        self.par_0 = {'MAP_0' : self.MAP_0, 'nHess_0' : self.nHess_0, 'ent_0' : self.ent_0}

    def initialize(self, MAP_0, nHess_0): # clear history
        self.t, self.MAP, self.nHess = 0, MAP_0, nHess_0
        self.LL_t = [self.LPrior] # Log Likelihood at step t
        self.x_t = []
        self.y_t = []

        self.ent_0 = self.calculate_diff_ent()
        self.ent = self.ent_0

        self.Record = Record()
        self.Record.data['MAP'].append(self.MAP)
        self.Record.data['nHess'].append(self.nHess)
        self.Record.data['ent'].append(self.ent)
    
    def pre_sampling_points(self,num_sampling) :
        return np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim), num_sampling)

    def calculate_diff_ent(self) :
        p_pdf = lambda theta: np.exp(np.sum([self.LL_t[L_idx](theta) for L_idx in range(len(self.LL_t))], axis=0))
        q_pdf = lambda theta: ((2*np.pi) ** (-self.dim/2)) * (np.linalg.det(self.nHess)**0.5) * np.exp(-(1/2) * np.sum((theta - self.MAP) @ (self.nHess) * (theta - self.MAP), axis=1))
        theta_points = self.MAP + (sqrtm(np.linalg.inv(self.nHess)) @ self.sample_points.T).T
        p_pdf_vals = p_pdf(theta_points)
        q_pdf_vals = q_pdf(theta_points)
        epsilon = 1e-40
        p_pdf_vals = np.maximum(p_pdf_vals,epsilon)
        q_pdf_vals = np.maximum(q_pdf_vals,epsilon)
        
        diff_ent = np.mean(- (p_pdf_vals / q_pdf_vals) * np.log(p_pdf_vals))
        return diff_ent
    
    def bayesian_update(self, x, y):
        LL = lambda theta : np.log( self.f(theta, x).reshape(-1) if y==1 else 1-self.f(theta, x).reshape(-1) ) # Log Likelihood
        self.LL_t.append(LL)
        self.x_t.append(x)
        self.y_t.append(y)
        
        NLPosterior = lambda theta : -sum([self.LL_t[L_idx](theta) for L_idx in range(len(self.LL_t))])
        func = NLPosterior

        # Analytic gradient and hessian of each models
        if self.func_name == 'LN' :
            gradient = lambda theta : (theta - self.MAP_0) @ self.nHess_0 + sum([ self.x_t[idx][:-1] * (self.f(theta,self.x_t[idx]).reshape(-1) - self.y_t[idx]) 
                                                                                 for idx in range(len(self.x_t))])
            hessian = lambda theta : self.nHess_0 + sum([ self.x_t[idx][:-1].T @ self.x_t[idx][:-1] * (1 - self.f(theta,self.x_t[idx]).reshape(-1)) * self.f(theta,self.x_t[idx]).reshape(-1) 
                                                         for idx in range(len(self.x_t))])
        
        elif self.func_name == 'MIRT' : 
            gradient = lambda theta : ((theta - self.MAP_0) @ self.nHess_0).reshape(-1) - sum([ (self.y_t[idx]  - (1 - self.y_t[idx]) * self.f(theta, self.x_t[idx] / (1-self.f(theta, self.x_t[idx])))) * (1 - sigmoid(self.x_t[idx] + theta)) 
                                                                                               for idx in range(len(self.y_t))]).reshape(-1)
            hessian = lambda theta : self.nHess_0 + sum([( (1 - self.y_t[idx]) * self.f(theta,self.x_t[idx]) / (1-self.f(theta,self.x_t[idx])) + self.y_t[idx]) * np.diag(sigmoid(self.x_t[idx] + theta).reshape(-1) * (1 - sigmoid(self.x_t[idx] + theta).reshape(-1))) 
                                                         + (1 - self.y_t[idx]) * (1 - sigmoid(self.x_t[idx] + theta)).T @ (1 - sigmoid(self.x_t[idx] + theta)) * self.f(theta,self.x_t[idx]) / ((1-self.f(theta,self.x_t[idx]))**2) 
                                                         for idx in range(len(self.x_t))])

        # Laplace Approximation
        opt_response = minimize(func, self.MAP, jac=gradient, hess=hessian, method='trust-ncg')
        self.MAP = opt_response.x
        self.nHess = hessian(self.MAP)
        self.ent = self.calculate_diff_ent()
        self.t += 1
        self.Record.data['MAP'].append(self.MAP)
        self.Record.data['nHess'].append(self.nHess)
        self.Record.data['ent'].append(self.ent)

        '''
        print("time step " + str(self.t))
        print("differential Entropy : ")
        print(self.ent)
        print("approx differential Entropy : ")
        print(0.5 * (self.dim * np.log(2 * np.pi * np.e) - np.log(np.linalg.det(self.nHess))))
        print("L2 norm : ")
        true_theta = np.array([0.9, 0.3, 0.3, 0.1])

        #true_theta = np.array([0, 0, 0, 0])
        print(np.sum((true_theta - self.MAP)**2))
        '''