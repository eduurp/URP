import datetime
import time
import os 

import numpy as np
import numdifftools as nd
from scipy.optimize import minimize

from .tools import save
from .config import *



'''
Predefined functions
'''
# normalize
def normalize(belief):   return belief/sum(belief)
# logistic
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_inv(x): return np.log(x) - np.log(1-x)
# log of Gaussian
def log_gaussian(x, mean, cov_inv):
    return -(1/2) * (x - mean).T @ (cov_inv) @ (x - mean)



'''
Predefined functions
'''
class Record():
    def __init__(self):
        self.typs = ['x', 'y', 'f', 'true_f', 'MAP', 'nHess', 'time']
        self.data = { typ:[] for typ in self.typs }

    def merge(self, subRecord):
        for typ in self.typs:
            self.data[typ].append(subRecord.data[typ])


class Belief():
    """
    Class Representation of Bayesian Belief.
    """
    def __init__(self, f, dim):
        self.dim = dim # dimension
        self.f = f # set correct rate function
    
        self.t, self.MAP, self.nHess, self.NLL = 0, None, None, None
        self.initialize(np.zeros(self.dim), np.eye(self.dim))

    def initialize(self, MAP_0=None, nHess_0=None): # clear history
        if MAP_0 is not None:   self.MAP_0 = MAP_0
        if nHess_0 is not None: self.nHess_0 = nHess_0
        self.par_0 = {'MAP_0' : self.MAP_0, 'nHess_0' : self.nHess_0}

        NLPrior = lambda theta :  - log_gaussian(theta, mean=self.MAP_0, cov_inv=self.nHess_0)
        self.NLL = NLPrior
        self.NLP_t = [NLPrior]
        self.t, self.MAP, self.nHess = 0, self.MAP_0, self.nHess_0

        self.Record = Record()
        self.Record.data['MAP'].append(self.MAP)
        self.Record.data['nHess'].append(self.nHess)

    # NLP is depreciated. Use NLL method
    def NLP(self, theta):
        return sum(NLP(theta) for NLP in self.NLP_t)

    def bayesian_update(self, x, y):
        NLLikelihood = lambda theta : - np.log(self.f(theta, x) if y == 1 else 1 - self.f(theta, x))
        self.NLP_t.append( NLLikelihood )

        current_NLL = self.NLL
        def NLL_next(theta):    return current_NLL(theta) + NLLikelihood(theta)
        self.NLL = NLL_next

        func = self.NLP # or self.NLL
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
    def __init__(self, Algo, dim, num_t = None, num_page = None, num_seed = None, num_sbj = None):
        self.Algo = Algo
        self.dim = dim

        params_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'params'))

        self.answers_page = np.load( os.path.join(params_path, ANSWER_SHEET) )
        self.true_theta_seed = np.load( os.path.join(params_path, TRUE_THETA) )
        self.init_nhess_sbj = np.load( os.path.join(params_path, INIT_NHESS) )

        self.num_t = min(num_t or self.answers_page.shape[1], self.answers_page.shape[1])
        self.num_page = min(num_page or self.answers_page.shape[0], self.answers_page.shape[0])
        self.num_seed = min(num_seed or self.true_theta_seed.shape[0], self.true_theta_seed.shape[0])
        self.num_sbj = min(num_sbj or self.init_nhess_sbj.shape[0], self.init_nhess_sbj.shape[0])
        
        self.Belief = Belief(self.Algo.f, self.dim)
        self.answers, self.true_theta = None, None
        self.set_anwser(0)
        self.set_true_theta(0)
        self.set_nhess(0)


    def do_test(self):
        while self.Belief.t < self.num_t:
            start_time = time.time()

            # Query suggestion by Algorithm
            x = self.Algo.query(self.Belief)
            self.Belief.Record.data['true_f'].append(true_f := self.Belief.f(self.true_theta, x))
            self.Belief.Record.data['f'].append(self.Belief.f(self.Belief.MAP, x))
            self.Belief.Record.data['x'].append(x)

            # Belief by Bayesian update
            y = 1 if self.answers[self.Belief.t] < true_f else 0
            self.Belief.bayesian_update(x, y)
            self.Belief.Record.data['y'].append(y)

            end_time = time.time()
            self.Belief.Record.data['time'].append(end_time-start_time)
            if VERBOSE>3: print(f't = {self.Belief.t} : {end_time-start_time:2f}')
        return self.Belief.Record


    def iter_dim(self, dim_size, set_param, iterate_function, depth=0):
        record = Record()
        for index in range(dim_size):
            start_time = time.time()

            set_param(index)
            result = iterate_function()
            record.merge(result)

            end_time = time.time()
            if VERBOSE > depth: print(f'{dim_size.__name__} = {index + 1} : {end_time - start_time:.2f}s')
        return record
    
    def set_anwser(self, index):    self.answers = self.answers_page[index]
    def iter_page(self):    return self.iter_dim(self.num_page, self.set_anwser, self.do_test, 2)

    def set_true_theta(self, index):    self.true_theta = self.true_theta_seed[index]
    def iter_seed(self):    return self.iter_dim(self.num_seed, self.set_true_theta, self.iter_page, 1)

    def set_nhess(self, index):    self.Belief.initialize(MAP_0=None, nHess_0=self.init_nhess_sbj[index])
    def iter_sbj(self):    return self.iter_dim(self.num_sbj, self.set_nhess, self.iter_seed, 0)


    @save
    def make(self):
        info = {'Algo' : self.Algo.__class__.__name__, 'f' : self.Algo.f.__name__, 'dim' : self.dim, 'num_t':self.num_t, 'num_page':self.num_page, 'num_seed':self.num_seed, 'num_sbj':self.num_sbj}
        hyp = self.Algo.__dict__
        print("===============================================================================")
        print("===============================================================================")
        print(info, hyp)
        print("===============================================================================")
        print("===============================================================================")
        
        result = self.iter_sbj().data
        return info, hyp, result
    
    @save
    def make_path(self):
        info = {'Algo' : self.Algo.__class__.__name__, 'f' : self.Algo.f.__name__, 'dim' : self.dim, 'num_t':self.num_t}
        hyp = self.Algo.__dict__
        print("-------------------------------------------------------------------------------")
        print("Only 1 path!")
        print(info, hyp)
        print("-------------------------------------------------------------------------------")
        
        result = self.do_test().data
        return info, hyp, result
    