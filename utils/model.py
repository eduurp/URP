from .base import *


# f
def LN(theta, x):
    return sigmoid(theta.T@x[:-1] + x[-1])

def MIRT(theta, x):
    return np.prod([sigmoid(theta[i]+x[i]) for i in range(len(x))])


# Algo
def FIX(Belief):
    return None


class PCM_LN(Algo):
    def __init__(self, alpha):
        self.alpha = alpha

    def query(self, Belief):
        val, vec = np.linalg.eig(Belief.nHess)
        return np.append(self.alpha * vec[:, np.argmin(val)], -self.alpha * vec[:, np.argmin(val)].T@Belief.MAP)


class PCM_MIRT(Algo):
    def __init__(self):
        pass

    def query(self, Belief):
        val, vec = np.linalg.eig(Belief.nHess)
        vec_max = vec[:, np.argmax(val)]

        if np.min(vec_max)*np.max(vec_max) < 0:
            x1, obj1 = self.sub_query(Belief, vec_max)
            x2, obj2 = self.sub_query(Belief, -vec_max)
            return x1 if obj1>obj2 else x2
        else:
            x, obj = self.sub_query(Belief, np.sign(np.max(vec_max))*vec_max)
            return x
    
    def sub_query(self, Belief, vec_max):
        sorted_i = np.argsort(vec_max)[::-1]

        n_J = 0
        for k in range(1, Belief.dim + 1):
            J = sorted_i[:k]
            if np.min(vec_max[J]) / np.sum(vec_max[J]) >= 1 / (k + 1): n_J = k
            else:   break
        
        x = np.zeros(Belief.dim)
        J = sorted_i[:n_J]
        sum_v = np.sum(vec_max[J])
        for i in range(Belief.dim):
            if i in J:
                x[i] = sigmoid_inv( ((sum_v / vec_max[i]) * (1/(n_J + 1))) ) - (Belief.MAP[i])
            else:
                x[i] = 999 - (Belief.MAP[i])

        obj = (1/np.prod(vec_max[J])) * ((np.sum(vec_max[J]))/(n_J+1))**(n_J)
        return x, obj

