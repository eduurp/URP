from .base import *
import itertools

# f
def LN(theta, x):
    return sigmoid(theta.T@x[:-1] + x[-1])

def MIRT(theta, x):
    return np.prod([sigmoid(theta[i]+x[i]) for i in range(len(x))])


# Algo
def FIX(Belief):
    return None


class PCM_LN(Algo):
    f = staticmethod(LN)

    def __init__(self, alpha):
        self.alpha = alpha

    def query(self, Belief):
        val, vec = np.linalg.eig(Belief.nHess)
        return np.append(self.alpha * vec[:, np.argmin(val)], -self.alpha * vec[:, np.argmin(val)].T@Belief.MAP)

import numdifftools as nd

class PCM_MIRT(Algo):
    f = staticmethod(MIRT)

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
        
        J = sorted_i[:n_J]
        return self.query_J(Belief, vec_max, J)


    def query_J(self, Belief, vec_max, J):
        x = np.zeros(Belief.dim)
        n_J = len(J)
        sum_v = np.sum(vec_max[J])
        for i in range(Belief.dim):
            if i in J:
                x[i] = sigmoid_inv( ((sum_v / vec_max[i]) * (1/(n_J + 1))) ) - (Belief.MAP[i])
            else:
                x[i] = 999 - (Belief.MAP[i])

        obj = (1/np.prod(vec_max[J])) * ((np.sum(vec_max[J]))/(n_J+1))**(n_J+1)
        '''
        func = lambda theta : Belief.f(theta, x)
        jcb = nd.Gradient(func)
        obj = np.array(jcb(Belief.MAP)).T@vec_max
        '''
        return x, obj


class PCM_MIRT_EXP(PCM_MIRT):
    def sub_query(self, Belief, vec_max):
        sorted_i = np.argsort(vec_max)[::-1]

        n_J = 0
        for k in range(1, Belief.dim + 1):
            J = sorted_i[:k]
            if np.min(vec_max[J]) / np.sum(vec_max[J]) >= 1 / (k + 1): n_J = k
            else:   break
        
        opt_J = sorted_i[:n_J]

        objs = {}
        for d in range(1, Belief.dim+1):
            for J in itertools.combinations(list(range(Belief.dim)), d):
                if np.min(vec_max[list(J)]) / np.sum(vec_max[list(J)]) >= 1 / (len(list(J)) + 1):
                    objs[f'opt {list(J)}' if set(list(J)) == set(opt_J) else str(list(J))] = self.query_J(Belief, vec_max, list(J))
        
        print('-----')
        for k, v in dict(sorted(objs.items(), key=lambda item: item[1][1])).items():
            print(f'{k} : {v[1]}')
        print('-----')

        '''
        with open('opt.txt', 'a') as file:
            file.write(f"{max(objs, key=lambda k: objs[k][1])}\n")

        with open('vec.txt', 'a') as file:
            file.write(f"{vec_max[0]} {vec_max[1]}\n")
        '''

        return self.query_J(Belief, vec_max, opt_J)