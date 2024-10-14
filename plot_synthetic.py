
# Synthetic Experiment Drawing Code
'''
This code is for drawing Synthetic Experiments
'''

import numpy as np
import matplotlib.pyplot as plt
import os


def safe_log(x):
    return np.log(np.clip(x, 1e-10, None))  

def list_files_in_directory(directory_path):
    files = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files


def get_result(directory_path, true_theta_path) :
    files = list_files_in_directory(directory_path)
    true_theta = np.load(true_theta_path)
    Datas = []
    for file in files :
        Data = np.load(file,allow_pickle=True)
        dim = len(Data.item().get('nHess')[0])
        logdet = safe_log(np.linalg.det(Data.item().get('nHess')))
        L2_norm = np.sqrt(np.sum(np.array(Data.item().get('MAP') - true_theta) ** 2,axis=1))
        L2_normalized = L2_norm / L2_norm[0]
        
        Datas.append([logdet, L2_norm, L2_normalized])
    
    Datas = np.array(Datas)
    
    log_det_mean, log_det_std = np.mean(Datas[:, 0, :],axis=0), np.std(Datas[:, 0, :],axis=0)
    L2_norm_mean, L2_norm_std = np.mean(Datas[:, 1, :],axis=0), np.std(Datas[:, 1, :],axis=0)
    L2_normalized_mean, L2_normalized_std = np.mean(Datas[:, 2, :],axis=0), np.std(Datas[:, 2, :],axis=0)
    
    return log_det_mean, log_det_std, L2_norm_mean, L2_norm_std, L2_normalized_mean, L2_normalized_std

def plot_result(model_name, directory_path, true_theta_path, tagging, dim = 4, alpha_LN = 10) :

    log_det_mean, log_det_std, L2_norm_mean, L2_norm_std, L2_normalized_mean, L2_normalized_std = get_result(directory_path, true_theta_path)
    
    plt.figure(figsize= (12,4)) 
    plt.subplot(1,2,1)
    plt.plot(log_det_mean, label=r"$\log\det \Lambda_t$")
    plt.fill_between([t for t in range(len(log_det_mean))],log_det_mean - log_det_std, log_det_mean + log_det_std, alpha = 0.4)  

    if model_name == 'LN' :
        plt.plot(np.array([t for t in range(1, len(log_det_mean))]), 
             dim * np.log(np.array([t for t in range(1, len(log_det_mean))])) + dim*np.log(alpha_LN**2 / (4*dim)), linestyle = '--', color='black',
             label = r'$d\log t + d\log\frac{\alpha^2}{4d}$')
        #plt.xlim([0,200])
    else :
        plt.plot(np.array([t for t in range(1, len(log_det_mean))]), 
             dim * np.log(np.array([t for t in range(1, len(log_det_mean))])) + dim*np.log(1 / (4*dim)), linestyle = '--', color='black',
             label = r'$d\log t + d\log\frac{1}{4d}$')
        #plt.xlim([0,200])
    
    plt.xlabel("epoches")
    plt.ylabel(r"$\log\det \Lambda_t$")
    plt.title("log det of negative hessian")
    plt.grid()
    plt.legend()

    
    plt.subplot(1,2,2)
    L2_norm_mean = L2_normalized_mean
    L2_norm_std = L2_normalized_std
    plt.plot(L2_norm_mean, label = r"$\frac{\Vert \hat{\theta}_t - \theta^* \Vert_2}{\Vert \hat{\theta}_0 - \theta^* \Vert_2}$")
    plt.fill_between([t for t in range(len(L2_norm_mean))],L2_norm_mean - L2_norm_std, L2_norm_mean + L2_norm_std, alpha = 0.4)  

    if model_name == 'LN' :
        plt.plot(np.array([t for t in range(1, len(L2_norm_mean))]), 
             np.sqrt((4 * ((dim/alpha_LN) **2)) * (1 / np.array([t for t in range(1, len(log_det_mean))]))), linestyle = '--', color='black',
             label = r'$\frac{2d}{\alpha}\frac{1}{\sqrt{t}}$')
        plt.xlim([0,200])
    else :
        plt.plot(np.array([t for t in range(5, len(L2_norm_mean))]), 
             np.sqrt((4 * (dim **2)) * (1 / np.array([t for t in range(5, len(log_det_mean))]))), linestyle = '--', color='black',
             label = r'$2d\frac{1}{\sqrt{t}}$')
        plt.xlim([0,200])
        

    plt.xlabel("epoches")
    plt.title("L2 norm of estimator")
    plt.ylabel(r"$\frac{\Vert \hat{\theta}_t - \theta^* \Vert_2}{\Vert \hat{\theta}_0 - \theta^* \Vert_2}$")
    plt.grid()
    plt.legend()
    plt.show()

    plt.savefig(dirs.split('/')[0] + '/' + model_name + '_' + tagging + '.pdf')

model = 'LN'
figure_num = 9
tagging = 'StudentT'

dirs = './Figures/Figure_' + str(figure_num) + '/' + model + '_synthetic_' + tagging
param_dirs = dirs + '/params_synthetic/' + 'true_theta_4dim_' + model + '.npy'
plot_result(model, dirs, param_dirs, tagging)
