# Utils #
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t
import itertools

from scipy.optimize import minimize
from scipy.linalg import sqrtm
import time

# Validations
import numdifftools as nd

''' Fundemental Operations'''

# Functions
def sigmoid(x): return 1 / (1 + np.exp(-x))

def safe_log(x):
    return np.log(np.clip(x, 1e-10, None))  

def log_gaussian(x, mean, cov_inv):
    if np.array(x).ndim==1:  x = x.reshape(1,-1)
    return -(1/2) * np.sum(((x - mean) @ (cov_inv)) * (x - mean), axis=1)

def gaussian(x, mean, cov_inv) :
    return ((2*np.pi)**(-len(mean)/2)) * np.sqrt(np.linalg.det(cov_inv)) * np.exp(log_gaussian(x, mean, cov_inv))
    #return ((2*np.pi)**(-2)) * np.sqrt(np.linalg.det(cov_inv)) * np.exp(log_gaussian(x, mean, cov_inv))

def L2norm(v1,v2) :
    return np.sum((v1.reshape(-1) - v2.reshape(-1))**2)

def nucnorm(mat1,mat2) :
    mat = mat1 - mat2
    val, vac = np.linalg.eig(mat)
    return np.sum(val)

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
    r = sigmoid(ps_@qs_.T[:-1] + qs_.T[-1])
    return r

# Basic Pipelines
class Record():
    def __init__(self):
        self.typs = ['MAP', 'nHess', 'f', 'true_f', 'time', 'ent', 'update_time']
        self.data = { typ:[] for typ in self.typs }

    def merge(self, subRecord):
        for typ in self.typs:
            self.data[typ].append(subRecord.data[typ])

class Belief():
    """
    Class Representation of Bayesian Belief.
    """
    def __init__(self, f, dim, MAP_0, nHess_0, prior_set = 'Gaussian', gmm_K = 3, gmm_mu = np.zeros([3, 4]), gmm_cov_inv = np.ones([3, 4, 4]), gmm_pi = (1/3) * np.ones(3)):
        self.dim = dim # dimension
        self.f = f # set correct rate function
        
        if self.f == LN :
            self.func_name = 'LN'
        else :
            self.func_name = 'MIRT'

        self.MAP_0 = MAP_0
        self.nHess_0 = nHess_0
        # self.ent_0 = 0
        
        if prior_set == 'Gaussian' :
            self.LPrior = lambda theta : log_gaussian(theta, mean=self.MAP_0, cov_inv=self.nHess_0)
            self.prior_set = 'Gaussian'
        
        elif prior_set == 'Student-t' :
            self.LPrior = lambda theta : np.log(multivariate_t.pdf(theta, loc=self.MAP_0, shape=self.nHess_0, df=5))
            self.prior_set = 'Student-t'
            
            self.Lprior_gradient = nd.Gradient(self.LPrior)
            self.Lprior_hessian = nd.Hessian(self.LPrior)

        elif prior_set == 'Gmm' :
            self.LPrior = lambda theta : sum([gmm_pi[idx] * gaussian(theta, gmm_mu[idx], gmm_cov_inv[idx]) for idx in range(gmm_K)])
            self.prior_set = 'Gmm'
            self.nLPrior = lambda theta : sum([gmm_pi[idx] * gaussian(theta, gmm_mu[idx], gmm_cov_inv[idx]) for idx in range(gmm_K)])
            opt_response = minimize(self.nLPrior, self.MAP_0, method='Nelder-Mead')
            self.MAP_0 = opt_response.x
            
            self.Lprior_gradient = lambda theta : sum([gmm_pi[idx] *  gaussian(theta, gmm_mu[idx], gmm_cov_inv[idx]) * (gmm_cov_inv[idx] @ (theta - gmm_mu[idx])) for idx in range(gmm_K)]) / (self.LPrior(theta)) 
            self.Lprior_hessian = nd.Hessian(self.LPrior)
            self.nHess_0 =  - self.Lprior_hessian(self.MAP_0)

        self.sample_points = self.pre_sampling_points(num_sampling=3000)
        self.initialize(self.MAP_0, self.nHess_0)
        # self.par_0 = {'MAP_0' : self.MAP_0, 'nHess_0' : self.nHess_0, 'ent_0' : self.ent_0}
        self.par_0 = {'MAP_0' : self.MAP_0, 'nHess_0' : self.nHess_0}

    def initialize(self, MAP_0, nHess_0): # clear history
        self.t, self.MAP, self.nHess = 0, MAP_0, nHess_0
        self.LL_t = [self.LPrior] # Log Likelihood at step t
        self.x_t = []
        self.y_t = []

        #self.ent_0 = self.calculate_diff_ent()
        #self.ent = self.ent_0

        self.Record = Record()
        self.Record.data['MAP'].append(self.MAP)
        self.Record.data['nHess'].append(self.nHess)
        #self.Record.data['ent'].append(self.ent)
    
    def pre_sampling_points(self,num_sampling) :
        return np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim), num_sampling)
    '''
    def calculate_diff_ent(self) :
        p_pdf = lambda theta: np.exp(np.sum([self.LL_t[L_idx](theta) for L_idx in range(len(self.LL_t))], axis=0))
        q_pdf = lambda theta: ((2*np.pi) ** (-self.dim/2)) * (np.linalg.det(self.nHess)**0.5) * np.exp(-(1/2) * np.sum((theta - self.MAP) @ (self.nHess) * (theta - self.MAP), axis=1))
        theta_points = self.MAP + (sqrtm(np.linalg.inv(self.nHess)) @ self.sample_points.T).T
        p_pdf_vals = p_pdf(theta_points)
        q_pdf_vals = q_pdf(theta_points)
        epsilon = 1e-10
        p_pdf_vals = np.maximum(p_pdf_vals,epsilon)
        q_pdf_vals = np.maximum(q_pdf_vals,epsilon)
        
        diff_ent = np.mean(- (p_pdf_vals / q_pdf_vals) * np.log(p_pdf_vals))
        return diff_ent
    '''
    def numerical_gradient(self, f, x, h=1e-8):
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_i_plus_h = np.copy(x)
            x_i_minus_h = np.copy(x)
            x_i_plus_h[i] += h
            x_i_minus_h[i] -= h
            grad[i] = (f(x_i_plus_h) - f(x_i_minus_h)) / (2 * h)
        return grad


    def bayesian_update(self, x, y):
        start_time = time.time()
        LL = lambda theta: safe_log(self.f(theta, x).reshape(-1) if y == 1 else 1 - self.f(theta, x).reshape(-1))
        
        self.LL_t.append(LL)
        self.x_t.append(x)
        self.y_t.append(y)

        NLPosterior = lambda theta : -sum([self.LL_t[L_idx](theta) for L_idx in range(len(self.LL_t))])
        
        func = NLPosterior

        # Analytic gradient and hessian of each models
        if self.prior_set == 'Gaussian' :
            if self.func_name == 'LN' :
                self.gradient = lambda theta : self.nHess_0 @ (theta - self.MAP_0).T + sum([ self.x_t[idx][:-1] * (self.f(theta,self.x_t[idx]).reshape(-1) - self.y_t[idx]) 
                                                                                    for idx in range(len(self.x_t))])
                            
                self.hessian = lambda theta : self.nHess_0 + sum([ np.outer(self.x_t[idx][:-1], self.x_t[idx][:-1]) * (1 - self.f(theta,self.x_t[idx]).reshape(-1)) * self.f(theta,self.x_t[idx]).reshape(-1) 
                                                            for idx in range(len(self.x_t))])

                opt_response = minimize(func, self.MAP, jac=self.gradient, hess=self.hessian, method='trust-ncg')

            elif self.func_name == 'MIRT' : 
                self.gradient = lambda theta : (self.nHess_0 @ (theta - self.MAP_0).T).reshape(-1) - sum([ (self.y_t[idx]  - (1 - self.y_t[idx]) * self.f(theta, self.x_t[idx] / (1-self.f(theta, self.x_t[idx])))) * (1 - sigmoid(self.x_t[idx] + theta)) 
                                                                                                for idx in range(len(self.y_t))]).reshape(-1)
                self.hessian = lambda theta : self.nHess_0 + sum([( (1 - self.y_t[idx]) * self.f(theta,self.x_t[idx]) / (1-self.f(theta,self.x_t[idx])) + self.y_t[idx]) * np.diag(sigmoid(self.x_t[idx] + theta).reshape(-1) * (1 - sigmoid(self.x_t[idx] + theta).reshape(-1))) 
                                                            + (1 - self.y_t[idx]) * np.outer((1 - sigmoid(self.x_t[idx] + theta)), (1 - sigmoid(self.x_t[idx] + theta))) * self.f(theta,self.x_t[idx]) / ((1-self.f(theta,self.x_t[idx]))**2) 
                                                            for idx in range(len(self.x_t))])
                
                opt_response = minimize(func, self.MAP, method='Nelder-Mead')
        else : 
            if self.func_name == 'LN' :
                    
                self.gradient = lambda theta : - self.Lprior_gradient(theta) + sum([ self.x_t[idx][:-1] * (self.f(theta,self.x_t[idx]).reshape(-1) - self.y_t[idx]) 
                                                                                        for idx in range(len(self.x_t))])
                                
                self.hessian = lambda theta : - self.Lprior_hessian(theta) + sum([ np.outer(self.x_t[idx][:-1], self.x_t[idx][:-1]) * (1 - self.f(theta,self.x_t[idx]).reshape(-1)) * self.f(theta,self.x_t[idx]).reshape(-1) 
                                                                for idx in range(len(self.x_t))])

                if self.prior_set == 'Gmm' :
                    opt_response = minimize(func, self.MAP, method='Nelder-Mead')
                else :
                    opt_response = minimize(func, self.MAP, jac=self.gradient, hess=self.hessian, method='trust-ncg')

            elif self.func_name == 'MIRT' : 
                self.gradient = lambda theta : - self.Lprior_gradient(theta) - sum([ (self.y_t[idx]  - (1 - self.y_t[idx]) * 
                                                                                                                self.f(theta, self.x_t[idx] / (1-self.f(theta, self.x_t[idx])))) * (1 - sigmoid(self.x_t[idx] + theta)) 
                                                                                                    for idx in range(len(self.y_t))]).reshape(-1)
                self.hessian = lambda theta : - self.Lprior_hessian(theta) + sum([( (1 - self.y_t[idx]) * self.f(theta,self.x_t[idx]) / (1-self.f(theta,self.x_t[idx])) + self.y_t[idx]) * np.diag(sigmoid(self.x_t[idx] + theta).reshape(-1) * (1 - sigmoid(self.x_t[idx] + theta).reshape(-1))) 
                                                                + (1 - self.y_t[idx]) * np.outer((1 - sigmoid(self.x_t[idx] + theta)), (1 - sigmoid(self.x_t[idx] + theta))) * self.f(theta,self.x_t[idx]) / ((1-self.f(theta,self.x_t[idx]))**2) 
                                                                for idx in range(len(self.x_t))])
                    
                opt_response = minimize(func, self.MAP, method='Nelder-Mead')
                
        self.MAP = opt_response.x
        end_time = time.time()

        self.nHess = self.hessian(self.MAP)
        # self.ent = self.calculate_diff_ent()
        self.t += 1
        self.Record.data['MAP'].append(self.MAP)
        self.Record.data['nHess'].append(self.nHess)
        # self.Record.data['ent'].append(self.ent)

        return end_time - start_time