import datetime
import time

import numpy as np
import numdifftools as nd
from scipy.optimize import minimize

from .tools import save
from .config import *

# normalize
def normalize(belief):   return belief/sum(belief)
# logistic
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_inv(x): return np.log(x) - np.log(1-x)
# log of Gaussian
def log_gaussian(x, mean, cov_inv):
    return -(1/2) * (x - mean).T @ (cov_inv) @ (x - mean)


class Record():
    def __init__(self):
        self.typs = ['MAP', 'nHess', 'f', 'true_f', 'time']
        self.data = { typ:[] for typ in self.typs }

    def merge(self, subRecord):
        for typ in self.typs:
            self.data[typ].append(subRecord.data[typ])


class Belief():
    """
    Class Representation of Bayesian Belief.
    """
    def __init__(self, f, dim, MAP_0=None, nHess_0=None):
        self.dim = dim # dimension
        self.f = f # set correct rate function

        self.MAP_0 = np.zeros(self.dim) if MAP_0 is None else MAP_0 # initial MAP
        self.nHess_0 = np.eye(self.dim) if nHess_0 is None else nHess_0 # initial Hessian
        self.par_0 = {'MAP_0' : self.MAP_0, 'nHess_0' : self.nHess_0}
        
        self.LPrior = lambda theta : log_gaussian(theta, mean=self.MAP_0, cov_inv=self.nHess_0)
        self.initialize()

    def initialize(self): # clear history
        self.t, self.MAP, self.nHess = 0, self.MAP_0, self.nHess_0
        self.LL_t = [self.LPrior] # Log Likelihood at step t

        self.Record = Record()
        self.Record.data['MAP'].append(self.MAP)
        self.Record.data['nHess'].append(self.nHess)

    def NLPosterior(self, theta):
        return -sum(LL(theta) for LL in self.LL_t)

    def bayesian_update(self, x, y):
        LL = lambda theta : np.log( self.f(theta, x) if y==1 else 1-self.f(theta, x) ) # Log Likelihood
        self.LL_t.append(LL)

        func = self.NLPosterior
        gradient = nd.Gradient(func)
        hessian = nd.Hessian(func)

        # Laplace Approximation
        opt_response = minimize(func, self.MAP, jac=gradient, hess=hessian, method='trust-ncg')
        self.MAP = opt_response.x
        self.nHess = hessian(self.MAP)

        self.t += 1

        self.Record.data['MAP'].append(self.MAP)
        self.Record.data['nHess'].append(self.nHess)


class Algo():
    def __init_subclass__(cls):
        check, problem = False, []

        for func_name in ['query', 'f']:
            not_exist = not hasattr(cls, func_name) or not callable(getattr(cls, func_name))
            if not_exist: 
                check = True
                problem.append(f"{cls.__name__} must implement the {func_name} method")
                                           
        if check:
            raise NotImplementedError('\n'.join(problem))


class Experiment():
    def __init__(self, Algo, dim):
        self.Algo = Algo
        self.dim = dim

        self.answers_page = np.load(ANSWER_SHEET)
        self.true_theta_seed = np.load(TRUE_THETA)
        self.init_nhess_sbj = np.load(INIT_NHESS)

        self.num_t, self.num_page, self.num_seed, self.num_sbj = self.answers_page.shape[1], self.answers_page.shape[0], self.true_theta_seed.shape[0], self.init_nhess_sbj.shape[0]

        self.page, self.seed, self.sbj = 0, 0, 0
    
    def iter_t(self):
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
            if VERBOSE>3: print(f't = {self.Belief.t} : {end_time-start_time:2f}')
        return self.Belief.Record

    def iter_page(self):
        Record_page = Record()
        while self.page < self.num_page:
            start_time = time.time()

            self.answers = self.answers_page[self.page]
            self.Belief.initialize() #MAP_0 받기
            self.page += 1

            Record_page.merge(self.iter_t())
            end_time = time.time()
            if VERBOSE>2: print(f'page = {self.page} : {end_time-start_time:2f}')
        return Record_page

    def iter_seed(self):
        Record_seed = Record()
        while self.seed < self.num_seed:
            start_time = time.time()

            self.true_theta = self.true_theta_seed[self.seed]
            self.page = 0
            Record_seed.merge(self.iter_page())
            self.seed += 1
            
            end_time = time.time()
            if VERBOSE>1: print(f'seed = {self.seed} : {end_time-start_time:2f}')
        return Record_seed
    
    def iter_sbj(self):
        Record_sbj = Record()
        while self.sbj < self.num_sbj:
            start_time = time.time()

            self.Belief = Belief(self.Algo.f, self.dim, MAP_0=None, nHess_0=self.init_nhess_sbj[self.sbj])
            self.seed = 0
            Record_sbj.merge(self.iter_seed())
            self.sbj += 1
            
            end_time = time.time()
            if VERBOSE>0: print(f'sbj = {self.sbj} : {end_time-start_time:2f}')
        return Record_sbj

    @save
    def make(self):
        info = {'Algo' : self.Algo.__class__.__name__, 'f' : self.Algo.f.__name__, 'dim' : self.dim, 'num_t':self.num_t, 'num_page':self.num_page, 'num_seed':self.num_seed, 'num_sbj':self.num_sbj}
        hyp = self.Algo.__dict__
        print(info, hyp)
        
        result = self.iter_sbj().data
        return info, hyp, result