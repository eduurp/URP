import os
import numpy as np
import matplotlib.pyplot as plt

num_t = 100
num_seed = 10
num1 = 1
num2 = 7
about = 'GB3'

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

cov_inv_trace = []
cov_inv_determinant = []
cov_inv_mineigvalue = []
for neg_hess in neg_hessian:
    cov_inv_tr, cov_inv_det, cov_inv_mineigval = [], [], []
    for t, neg_h in enumerate(neg_hess):
        cov_inv_tr.append(np.trace(neg_h)/4)
        cov_inv_det.append(np.linalg.det(neg_h)**(1/4))
        cov_inv_mineigval.append(np.min(np.linalg.eig(neg_h)[0]))
    cov_inv_trace.append(cov_inv_tr)
    cov_inv_determinant.append(cov_inv_det)
    cov_inv_mineigvalue.append(cov_inv_mineigval)

avg_cov_inv_trace = np.mean(cov_inv_trace, axis=0)
avg_cov_inv_determinant = np.mean(cov_inv_determinant, axis=0)
avg_cov_inv_mineigvalue = np.mean(cov_inv_mineigvalue, axis=0)

for seed in range(10):  plt.plot(cov_inv_trace[seed], c='b', alpha=0.5)
for seed in range(10):  plt.plot(cov_inv_determinant[seed], c='r', alpha=0.5)
for seed in range(10):  plt.plot(cov_inv_mineigvalue[seed], c='g', alpha=0.5)
plt.plot(avg_cov_inv_trace , c='b', label='trace of Cov_inv')
plt.plot(avg_cov_inv_determinant , c='r', label='det of Cov_inv')
plt.plot(avg_cov_inv_mineigvalue , c='g', label='min eig val of Cov_inv')

plt.xlabel('step t')
plt.ylabel('trace - det - min eig val of Cov inv')
plt.title(f'True theta = {np.round(true_theta, 2)}, Method = GB')
plt.legend()
plt.savefig('fig_6_0')
plt.show()

'''
cov_determinant = []
coefficient = []
for neg_hess in neg_hessian:
    cov_det, coef = [], []
    for t, neg_h in enumerate(neg_hess):
        cov_det.append(1/np.linalg.det(neg_h))
        coef.append((np.linalg.det(neg_h)**(1/4)))
    cov_determinant.append(cov_det)
    coefficient.append(coef)

avg_cov_determinant = np.mean(cov_determinant, axis=0)
for seed in range(10):  plt.plot(np.log(cov_determinant[seed]), c='b')

bound = 1/2
dim = 4
bound_cov_determinant = np.array([1/(((bound**2/(4*dim))*t + (1/avg_cov_determinant[0])**(1/dim))**dim) for t in range(100)])

plt.plot(np.log(bound_cov_determinant), c='r', label='upper bound')
plt.xlabel('step t')
plt.ylabel('log det of Cov')
plt.title(f'True theta = {np.round(true_theta, 2)}, Method = GB')
plt.legend()
plt.savefig('fig_4_3')
plt.show()

avg_coefficient = np.mean(coefficient, axis=0)
for seed in range(10):  plt.plot(coefficient[seed], c='b')

bound_coefficient = np.array([bound**2/(4*dim)*t + avg_coefficient[0] for t in range(100)])

plt.plot(bound_coefficient, c='r', label='lower bound')
plt.xlabel('step t')
plt.ylabel('coef')
plt.title(f'True theta = {np.round(true_theta, 2)}, Method = GB')
plt.legend()
# plt.savefig('fig_7_1')
plt.show()
#'''