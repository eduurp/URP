import os
import numpy as np
import matplotlib.pyplot as plt

num_t = 100
num_seed = 10
num1 = 1
num2 = 2
about = 'FIX'

file_name = f"exam_{num_t}_{num_seed}_{num1}_{num2}_{about}"
file_path = os.path.join('.', file_name)

for file in os.listdir(file_path):
    if file == 'x_seed_t.npy':
        x = np.load(os.path.join(file_path, file))
    elif file == 'y_seed_t.npy':
        y = np.load(os.path.join(file_path, file))
    elif file == 'neg_hessian_seed_t.npy':
        neg_hessian = np.load(os.path.join(file_path, file))
    elif file == 'true_theta.npy':
        true_theta = np.load(os.path.join(file_path, file))
    elif file == 'theta_hat_seed_t.npy':
        theta_hat = np.load(os.path.join(file_path, file))
    elif file == 'correct_rate_seed_t.npy':
        correct_rate = np.load(os.path.join(file_path, file))

#'''
cov_determinant = []
for neg_hess in neg_hessian:
    cov_det = []
    for neg_h in neg_hess:
        cov_det.append(1/np.linalg.det(neg_h))
    cov_determinant.append(cov_det)

avg_cov_determinant = np.mean(cov_determinant, axis=0)
for seed in range(10):  plt.plot(cov_determinant[seed])
#'''

'''
print(np.linalg.norm([0.25, 0.25, 0.25, 0.25]))
print(x[:,:,:])
'''
print(true_theta)

bound = 1/2
dim = 4
bound_cov_determinant = np.array([1/(((bound**2/(4*dim))*t + 1)**dim) for t in range(100)])

plt.plot(bound_cov_determinant, label='bound')
plt.legend()
plt.show()

# TODO : theta_map 대신 theta_true를 써서 실험 해보자