import numpy as np

A = np.load("answers.npy")

print(A.shape)

'''
from scipy.stats import ortho_group
import numpy as np

t = 10
d = 3

X = ortho_group.rvs(dim=t)
a = np.random.random(t)
A = np.diag(a)
B = X@A@X.T
B_1 = B[:d,:d]
B_2 = X[:d]@A@X[:d].T
Y = np.random.random((d, t))

print(np.max(np.linalg.eig(B_1)[0]))
print(np.max(np.linalg.eig(B)[0]))
'''
