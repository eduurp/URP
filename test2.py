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
base_MIRT = "[240426 22-42-23] PCM_MIRT_EXP MIRT 4 1 1 3 100"
nHess_MIRT = np.load(os.path.join(base_MIRT, "nHess.npy"))

# nHess_0_MIRT = np.load(os.path.join(base_MIRT, "nHess_0.npy"))

'''
true_f = np.load(os.path.join(base_MIRT, "true_f.npy"))
print(true_f[:,:,-1])
'''

base_LN = "[231119 15-34-39] LN 4 100 10 10 PCM_LN alpha_0.5"
nHess_LN = np.load(os.path.join(base_LN, "nHess.npy"))

# nHess_0_LN = np.load(os.path.join(base_MIRT, "nHess_0.npy"))

Hess_det_MIRT = det(nHess_MIRT)
Hess_nlogdet_MIRT = -logdet(nHess_MIRT)
d=4

for i in range(100):
    print( np.linalg.eig(nHess_MIRT[0][0][i])[0] )

plt.plot([ (H**(1/d))/(t+1) for t, H in enumerate(Hess_det_MIRT) ], label='MIRT')
plt.plot([ t/4 for t, H in enumerate(Hess_det_MIRT) ], label='MIRT')

Hess_det_LN = det(nHess_LN)
Hess_nlogdet_LN = -logdet(nHess_LN)
# plt.plot(Hess_nlogdet_LN, label='LN')
plt.legend()
# plt.show()

'''
f = np.load(os.path.join(base, "f.npy"))

print(f)
plt.plot(f[1][0].astype(float))
plt.show()



#'''
