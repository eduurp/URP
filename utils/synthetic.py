from .base import *
import itertools

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
    def __init__(self):
        pass
    
    def query(self, Belief):
        ### We regarding Only for diagonal component cases ###
        component_idx = np.argmin(Belief.nHess @ np.ones(len(Belief.nHess)).T)
        x = 1e7 * np.ones(len(Belief.nHess))
        x[component_idx] = - Belief.MAP[component_idx]
        return x

# Experiments
class Experiment_Synthetic():
    def __init__(self, Algo, dim, f, num_t, true_theta,  MAP_0, nHess_0):
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
    exp = Experiment_Synthetic(Algo, dim, f, num_t, true_theta, MAP_0, nHess_0)
    for iter in tqdm(range(iterations)) :
        data = exp.iter_t()
        Data.append(data)
        print(iter)
    Data = np.array(Data)
    np.save("Synthetic_" + model_name + "_" + str(dim) + ".npy",Data)