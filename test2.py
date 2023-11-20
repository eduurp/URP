import os
import numpy as np
import matplotlib.pyplot as plt

def each(func):
    def each_func(A_seed_page_t):
        return np.mean([ [ [ func(A) for A in A_t ] for A_t in A_page_t] for A_page_t in A_seed_page_t ], axis=(0,1))
    return each_func

@each
def det(A): return np.linalg.det(A)

@each
def logdet(A): return np.log(np.linalg.det(A))

base_MIRT = "[231118 05-15-00] MIRT 4 100 10 10 PCM_MIRT"
nHess_MIRT = np.load(os.path.join(base_MIRT, "nHess.npy"))

base_LN = "[231119 15-34-39] LN 4 100 10 10 PCM_LN alpha_0.5"
nHess_LN = np.load(os.path.join(base_LN, "nHess.npy"))

A = np.load(os.path.join(base_MIRT, "nHess.npy"))
B = np.load(os.path.join(base_LN, "nHess.npy"))
print(np.linalg.det(A[0][0][0]))
print(np.linalg.det(B[0][0][0]))

Hess_nlogdet_MIRT = -logdet(nHess_MIRT)
plt.plot(Hess_nlogdet_MIRT, label='MIRT')

Hess_nlogdet_LN = -logdet(nHess_LN)
plt.plot(Hess_nlogdet_LN, label='LN')
plt.legend()
plt.show()
'''
f = np.load(os.path.join(base, "f.npy"))

print(f)
plt.plot(f[1][0].astype(float))
plt.show()



#'''
