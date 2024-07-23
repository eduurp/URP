from .base import *
import itertools


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
    
    def do(self, i) :   
        self.Belief = Belief(self.f, self.dim, self.MAP_0, self.nHess_0)
            
        if self.num_t < len(self.PQI[i]) :
                self.true_theta = self.P[i]
                
                self.Belief.initialize(self.MAP_0, self.nHess_0)
                while self.Belief.t < self.num_t:
                    start_time = time.time()
                    
                    # Query suggestion by Algorithm
                    idx = self.Algo.query(self, self.Belief, self.Q[self.PQI[i]])
                    self.Belief.Record.data['true_f'].append(true_f := self.Belief.f(self.true_theta, self.Q[self.PQI[i][idx]]))
                    self.Belief.Record.data['f'].append(self.Belief.f(self.Belief.MAP, self.Q[self.PQI[i][idx]]))

                    self.Belief.bayesian_update(self.Q[self.PQI[i][idx]], self.R[i][self.PQI[i][idx]])
                    self.PQI[i].pop(idx)
                    
                    end_time = time.time()
                    self.Belief.Record.data['time'].append(end_time - start_time)

        Data = np.array(self.Belief.Record)
        return Data
