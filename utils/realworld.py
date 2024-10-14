from .base import *
import itertools


# Sampling Importance Sampling
def SIR(Belief) :
    true_pdf = lambda theta: np.exp(np.sum([Belief.LL_t[L_idx](theta) for L_idx in range(len(Belief.LL_t))], axis=0))
    proposal_pdf = lambda theta: ((2*np.pi) ** (-Belief.dim/2)) * (np.abs(np.linalg.det(Belief.nHess))**0.5) * np.exp(-(1/2) * np.sum((theta - Belief.MAP) @ (Belief.nHess) * (theta - Belief.MAP), axis=1))

    samples = Belief.MAP + (sqrtm(np.linalg.inv(Belief.nHess)) @ Belief.sample_points.T).T
    samples = np.real(samples)
    
    N = len(samples)

    iterations = 10 # this is for SIR iterations numbers

    for _ in range(iterations) :
        weights = np.clip(np.abs(true_pdf(samples) / np.clip(proposal_pdf(samples),1e-6,None)), 1e-6,None)
        weights = weights / np.sum(weights)
        indices = np.random.choice(N, N, p = weights)
        samples = samples[indices]

    return samples


def bin_en(p): return np.nan_to_num( - p*safe_log(p) - (1-p)*safe_log(1-p) )

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
        samples = SIR(Belief)
        
        # f(samples, queries) once calculate
        f_samples_queries = self.f(samples, queries)
        M = len(samples)
        H1_pos = (np.sum(f_samples_queries, axis=0) / M) * safe_log(np.sum(f_samples_queries, axis=0)/M) 
        H1_neg = (np.sum(1 - f_samples_queries, axis=0) / M) * safe_log(np.sum(1 - f_samples_queries, axis=0)/M)
        H1 = H1_pos + H1_neg
        
        H2 = np.sum(f_samples_queries * safe_log(f_samples_queries) + (1 - f_samples_queries) * safe_log(1 - f_samples_queries), axis=0) / M
        
        gain = H1 - H2
        idx = np.argmax(gain)
        return idx

class MaxEnt():
    def __init__(self, model):
        self.f = model
        
    def query(self, Belief, queries):
        samples = SIR(Belief)
        
        # f(samples, queries) once calculate
        f_samples_queries = self.f(samples, queries)

        prob = np.mean(f_samples_queries,axis = 0)
        idx = np.argmax(bin_en(prob))   
        return idx

class VarRatio() :
    def __init__(self, model):
        self.f = model
        
    def query(self, Belief, queries):
        samples = SIR(Belief)
        
        # f(samples, queries) once calculate
        f_samples_queries = self.f(samples, queries)
        
        prob = np.mean(f_samples_queries,axis = 0)
        max_p = np.max([prob, 1-prob], axis = 0)
        idx = np.argmax(1 - max_p)
        return idx

class MeanStd() :
    def __init__(self, model):
        self.f = model

    def query(self, Belief, queries):
        samples = SIR(Belief)
        
        # f(samples, queries) once calculate
        f_samples_queries = self.f(samples, queries)

        pos = np.sqrt(np.mean(f_samples_queries**2,axis = 0) - np.mean(f_samples_queries,axis = 0)**2)
        neg = np.sqrt(np.mean((1-f_samples_queries)**2,axis = 0) - np.mean(1 - f_samples_queries,axis = 0)**2)
        mean_std = pos + neg
        idx = np.argmax(mean_std)
        return idx

# Experiments
class Experiment_Realworld():
    def __init__(self, Algo, dim, f, num_t, MAP_0, nHess_0, R, P, Q, PQI, prior_set = 'Gaussian', gmm_K = 3, gmm_mu = np.zeros([3, 4]), gmm_cov_inv = np.ones([3, 4, 4]), gmm_pi = (1/3) * np.ones(3)):
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
        
        self.prior_set = prior_set
        self.gmm_K = gmm_K
        self.gmm_mu = gmm_mu
        self.gmm_cov_inv = gmm_cov_inv
        self.gmm_pi = gmm_pi     
    
    
    def do(self, i) :   
        self.Belief = Belief(self.f, self.dim, self.MAP_0, self.nHess_0, self.prior_set, 
                             gmm_K = self.gmm_K, gmm_mu = self.gmm_mu, gmm_cov_inv= self.gmm_cov_inv, gmm_pi = self.gmm_pi)
   
        if self.num_t < len(self.PQI[i]) :
                self.true_theta = self.P[i]
                
                self.Belief.initialize(self.MAP_0, self.nHess_0)
                while self.Belief.t < self.num_t:
                    if self.Belief.t == 0 :
                        print("step " + str(self.Belief.t))
                        print("L2 norm ratio : ")
                        print(np.sum((self.true_theta - self.Belief.MAP)**2) / np.sum((self.true_theta - self.Belief.MAP_0)**2))
                    start_time = time.time()
                    idx = self.Algo.query(self, self.Belief, self.Q[self.PQI[i]])
                    end_time = time.time()
                    self.Belief.Record.data['time'].append(end_time - start_time)

                    self.Belief.Record.data['true_f'].append(true_f := self.Belief.f(self.true_theta, self.Q[self.PQI[i][idx]]))
                    self.Belief.Record.data['f'].append(self.Belief.f(self.Belief.MAP, self.Q[self.PQI[i][idx]]))
                    
                    update_time = self.Belief.bayesian_update(self.Q[self.PQI[i][idx]], self.R[i][self.PQI[i][idx]])
                    self.Belief.Record.data['update_time'].append(update_time)
                    self.PQI[i].pop(idx)

                    # Assumption : MAP 가 바뀌는 순간과, 맞 / 틀 은 연관이 있을 것이다
                    if self.Belief.t%10 == 1 :
                        print("step " + str(self.Belief.t))
                        print("L2 norm ratio : ")
                        print(np.sum((self.true_theta - self.Belief.MAP)**2) / np.sum((self.true_theta - self.Belief.MAP_0)**2))
        Data = np.array(self.Belief.Record.data)
        return Data