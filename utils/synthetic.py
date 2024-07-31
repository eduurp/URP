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
        x = 1e3 * np.ones(len(Belief.nHess))
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
            
            '''
            if self.Belief.t%20 == 0 :
                print("step " + str(self.Belief.t))
                print("L2 norm : ")
                print(np.sum((self.true_theta - self.Belief.MAP)**2))
                print("Log det based differential entropy : ")
                print(self.Belief.dim/2 * np.log(2*np.pi*np.e) - 0.5*np.log(np.linalg.det(self.Belief.nHess)))
                print("Approximated differential Entropy : ")
                print(self.Belief.ent)
            #'''

        return self.Belief.Record.data