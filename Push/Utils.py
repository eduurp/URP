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

        print(self.func_name)

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
        
        #### sometimes occur that hessian's determinant became negative, please chech the BNS formulations for this types of error occuring.
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

        print("time step " + str(self.t))
        print("differential Entropy : ")
        print(self.ent)
        print("approx differential Entropy : ")
        print(0.5 * (self.dim * np.log(2 * np.pi * np.e) - np.log(np.linalg.det(self.nHess))))
        print("L2 norm : ")
        true_theta = np.array([0.9, 0.3, 0.3, 0.1])

        #true_theta = np.array([0, 0, 0, 0])
        print(np.sum((true_theta - self.MAP)**2))

# Sampling Importance Sampling
def SIR(Belief, iterations) :
    true_pdf = lambda theta: np.exp(np.sum([Belief.LL_t[L_idx](theta) for L_idx in range(len(Belief.LL_t))], axis=0))
    proposal_pdf = lambda theta: ((2*np.pi) ** (-Belief.dim/2)) * (np.linalg.det(Belief.nHess)**0.5) * np.exp(-(1/2) * np.sum((theta - Belief.MAP) @ (Belief.nHess) * (theta - Belief.MAP), axis=1))
    
    samples = Belief.MAP + (sqrtm(np.linalg.inv(Belief.nHess)) @ Belief.sample_points.T).T
    N = len(samples)

    for _ in range(iterations) :
        weights = true_pdf(samples) / proposal_pdf(samples)
        weights = weights / np.sum(weights)
        indices = np.random.choice(N, N, p = weights)
        samples = samples[indices]
    
    return samples

''' Synthetic case Operations '''

# Algo
class DURM_LN():
    def __init__(self, alpha):
        self.alpha = alpha

    def query(self, Belief):
        val, vec = np.linalg.eig(Belief.nHess)
        x_1 = self.alpha * vec[:, np.argmin(val)]
        x_0 = - x_1.T @ Belief.MAP
        return np.append(x_1,x_0)

class DURM_MIRT():
    def __init__(self, alpha):
        self.alpha = alpha
    
    def query(self, Belief):
        ### We regarding Only for diagonal component cases ###
        component_idx = np.argmin(Belief.nHess @ np.ones(len(Belief.nHess)).T)
        x = 1e7 * np.ones(len(Belief.nHess))
        x[component_idx] = - Belief.MAP[component_idx]
        return x

# Experiments
class Experiment_Synthetic():
    def __init__(self, Algo, dim, f, true_theta, num_t, MAP_0, nHess_0):
        self.Algo = Algo
        self.dim = dim
        self.f = f
        self.true_theta = true_theta
        self.num_t = num_t
        self.MAP_0 = MAP_0
        self.nHess_0 = nHess_0
    
    def iter_t(self):
        self.Belief = Belief(self.f, self.dim, self.MAP_0, self.nHess_0)
        self.Belief.initialize(self.MAP_0, self.nHess_0)
        self.answers = np.random.uniform(0,1,self.num_t)
        
        while self.Belief.t < self.num_t:
            start_time = time.time()
            
            # Query suggestion by Algorithm
            x = self.Algo.query(self.Belief)
            self.Belief.Record.data['true_f'].append(true_f := self.Belief.f(self.true_theta, x))
            self.Belief.Record.data['f'].append(self.Belief.f(self.Belief.MAP, x))

            # Belief by Bayesian update
            y = 1 if self.answers[self.Belief.t] < true_f else 0
            self.Belief.bayesian_update(x, y)

            end_time = time.time()
            self.Belief.Record.data['time'].append(end_time-start_time)
        return self.Belief.Record
    
def Run_Synthetic(model_name, dim, alpha, true_theta, num_t, MAP_0, nHess_0, iterations) :
    if model_name == 'MIRT' :
        f = MIRT
        Algo = DURM_MIRT(alpha)
    else :
        f = LN
        Algo = DURM_LN(alpha)

    Data = []
    exp = Experiment_Synthetic(Algo, dim, f, true_theta, num_t, MAP_0, nHess_0)
    for iter in tqdm(range(iterations)) :
        data = exp.iter_t()
        Data.append(data)
        print(iter)
    Data = np.array(Data)
    np.save("Synthetic_" + model_name + "_" + str(dim) + ".npy",Data)

''' Realworld case Operations '''

# Algo
class DURM():
    def __init__(self, model):
        self.f = model
        
    def query(self, Belief, queries):
        MAP = Belief.MAP
        val, vec = np.linalg.eig(Belief.nHess)
        vec_min = vec[:,np.argmin(val)]
        delta = 1e-4
        idx = np.argmax(np.array([(self.f(MAP + delta * vec_min,queries[q_idx]) - self.f(MAP - delta * vec_min,queries[q_idx]))**2 for q_idx in range(len(queries))]).reshape(-1))
        return idx

class BALD():
    def __init__(self, model):
        self.f = model

    def query(self, Belief, queries):
        samples = SIR(Belief, 10)
        
        # f(samples, queries) 결과 한 번 계산
        f_samples_queries = self.f(samples, queries)
        M = len(samples)
        H1_pos = (np.sum(f_samples_queries, axis=0) / M) * np.log(np.sum(f_samples_queries, axis=0)/M) 
        H1_neg = (np.sum(1 - f_samples_queries, axis=0) / M) * np.log(np.sum(1 - f_samples_queries, axis=0)/M)
        H1 = H1_pos + H1_neg
        
        H2 = np.sum(f_samples_queries * np.log(f_samples_queries) + (1 - f_samples_queries) * np.log(1 - f_samples_queries), axis=0) / M
        
        gain = H1 - H2
        idx = np.argmax(gain)
        return idx

# Experiments
class Experiment_Realworld():
    def __init__(self, Algo, dim, f, num_t, MAP_0, nHess_0, R, P, Q, PQI):
        self.Algo = Algo
        self.dim = dim
        self.f = f
        self.num_t = num_t
        self.MAP_0 = MAP_0
        self.nHess_0 = nHess_0
        
        self.R = R
        self.P = P
        self.Q = Q
        self.PQI = PQI
    
    def do(self, populate_idx) :
        Data = []
        for i in tqdm(populate_idx) :
            
            self.Belief = Belief(self.f, self.dim, self.MAP_0, self.nHess_0)
            
            if self.num_t < len(self.PQI[i]) :
                self.true_theta = self.P[i]
                
                self.Belief.initialize(self.MAP_0, self.nHess_0)
                while self.Belief.t < self.num_t:
                    start_time = time.time()
                    
                    # Query suggestion by Algorithm
                    idx = self.Algo.query(self.Belief, self.Q[self.PQI[i]])
                    self.Belief.Record.data['true_f'].append(true_f := self.Belief.f(self.true_theta, self.Q[self.PQI[i][idx]]))
                    self.Belief.Record.data['f'].append(self.Belief.f(self.Belief.MAP, self.Q[self.PQI[i][idx]]))

                    self.Belief.bayesian_update(self.Q[self.PQI[i][idx]], self.R[i][self.PQI[i][idx]])
                    self.PQI[i].pop(idx)
                    
                    end_time = time.time()
                    self.Belief.Record.data['time'].append(end_time - start_time)

            Data.append(self.Belief.Record)
        return Data

def Run_RealWorld(algorithm_name, model_name, dim, pop_num) :
    if model_name == "MIRT" :
        model = MIRT
    else :
        model = LN

    R = np.load("RealWorld_dataset/" + model_name+ "_Dataset.npy")
    n_p, n_q = R.shape
    obs = ~np.isnan(R)
    max_query_num = np.min([np.sum(obs[j]) for j in range(len(obs))])

    possible_query_idx = []
    for i in range(obs.shape[0]):
        true_indices = np.where(obs[i])
        possible_query_idx.append(true_indices[0])
        
    PQIs = []

    for i in range(len(possible_query_idx)) :
        PQIs.append(possible_query_idx[i].tolist())
    PQI = PQIs

    P = np.load("RealWorld_dataset/Theta_" + model_name + "_" + str(dim) + "_MSE.npy")
    Q = np.load("RealWorld_dataset/Query_" + model_name + "_" + str(dim) + "_MSE.npy")

    num_t = 100

    mean = np.mean(P,axis=0)
    cov = np.cov(P.T)
    MAP_0 = mean
    nHess_0 = np.linalg.inv(cov)

    # pop_num < len(P)
    populate_idx = np.array([p_id for p_id in range(pop_num)])

    if algorithm_name == 'DURM':
        algorithm = DURM(model)

    elif algorithm_name == 'BALD':
        algorithm = BALD(model)
    '''
    elif algorithm_name == 'MaxEnt':
        algorithm = Max_Ent(model)
    
    elif algorithm_name == 'VarRatio':
        algorithm = Variation_Ratios(model)

    elif algorithm_name == 'MeanStd' :
        algorithm = Mean_std(model)
    '''
    
    exp = Experiment_Realworld(algorithm, dim, model, num_t, MAP_0, nHess_0, R, P, Q, PQI)

    Data = exp.do(populate_idx)
    
    Data = np.array(Data)
    np.save("Realworld_" + model_name + "_" + str(dim) +"_" + algorithm_name + ".npy",Data)