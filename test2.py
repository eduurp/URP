import os
import numpy as np

num_t = 100
num_seed = 10
num1 = 1
num2 = 3
about = 'GB'

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

seed = 0

for t in range(99):
    # print(f'{theta_hat[seed][t+1]} : {y[seed][t]} : {correct_rate[seed][t]}')
    print(f'{neg_hessian[seed][t+1]}')
    print(f'val : {np.linalg.eig(neg_hessian[seed][t+1])[0]}')
    print(f'vec : {np.linalg.eig(neg_hessian[seed][t+1])[1]}')
print(true_theta)