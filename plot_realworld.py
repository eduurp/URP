# Synthetic Experiment Drawing Code
'''
This code is for drawing real world dataset experiments
'''


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def sigmoid(x): return 1 / (1 + np.exp(-x))

def MIRT(ps, qs):
    ps_, qs_ = ps, qs
    if np.array(qs).ndim==1:  qs_ = [qs]
    if np.array(ps).ndim==1:  ps_ = [ps]
    r = np.prod(sigmoid(np.array([qs_] * len(ps_)) + np.array([ps_] * len(qs_)).swapaxes(0,1)), axis=2)
    return r

def LN(ps, qs):
    ps_, qs_ = ps, qs
    if np.array(qs).ndim==1:  qs_ = qs.reshape(1,-1)
    if np.array(ps).ndim==1:  ps_ = ps.reshape(1,-1)
    r = sigmoid(ps_@qs_.T[:-1] + qs_.T[-1])
    return r

def threshold_check(P,Q,R,model_name) :
    if model_name == 'LN' :
        model = LN
    else :
        model = MIRT

    R_hat = model(P,Q)

    # not NaN value get
    indices = np.argwhere(~np.isnan(R))
    values_true = np.array([R[row, col] for row, col in indices])
    values_f = np.array([R_hat[row, col] for row, col in indices])

    # finding optimal threashold value
    best_threshold = 0
    best_accuracy = 0

    for threshold in np.linspace(0, 1, 1001):
        # Binarization
        values_binary = (values_f >= threshold).astype(int)
        # Calculate Accuracy
        accuracy = np.mean(values_binary == values_true)
        # Optimal Threshold update
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

def calculate_accuarcy(MAP,Q,R,model_name, idx, best_threshold) : 
    if model_name == 'LN' :
        model = LN
    else :
        model = MIRT

    R_hat = model(MAP,Q).reshape(-1)
    R_true = R[idx]
    non_nan_indices = ~np.isnan(R_true)

    values_binary = (R_hat >= best_threshold).astype(int)
    accuracy = np.mean(values_binary[non_nan_indices] == R_true[non_nan_indices])
    return accuracy

def list_files_in_directory(directory_path):
    files = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files

def get_result_std_based(directory_path, P, Q, R, model_name, best_threshold) :
    files = list_files_in_directory(directory_path)
    Datas = []
    Accs = []
    for file in files :
        Data = np.load(file,allow_pickle=True)
        dim = len(Data.item().get('nHess')[0])

        logdet = np.log(np.linalg.det(Data.item().get('nHess')))
        idx = int(file.split('\\')[-1].split('_')[0])
        #idx = int(file.split('/')[-1].split('_')[0])
        L2_norm = np.sqrt(np.sum((Data.item().get('MAP') - P[idx])**2,axis=1))
        L2_normalized = L2_norm / L2_norm[0]

        time = np.append(np.array(Data.item().get('time')), 0)
        update_time = np.append(np.array(Data.item().get('update_time')), 0)

        MAP = np.array(Data.item().get('MAP'))[-1]
        accuracy = calculate_accuarcy(MAP,Q,R,model_name, idx, best_threshold)

        Datas.append([logdet, L2_norm, L2_normalized, time, update_time])
        Accs.append(accuracy)

    Datas = np.array(Datas)
    Accs = np.array(Accs)

    log_det_mean, log_det_std = np.mean(Datas[:, 0, :],axis=0), np.std(Datas[:, 0, :],axis=0)
    L2_norm_mean, L2_norm_std = np.mean(Datas[:, 1, :],axis=0), np.std(Datas[:, 1, :],axis=0)
    L2_normalized_mean, L2_normalized_std = np.mean(Datas[:, 2, :],axis=0), np.std(Datas[:, 2, :],axis=0)
    accuracy_mean, accuracy_std = np.mean(Accs), np.std(Accs)
    return log_det_mean, log_det_std, L2_norm_mean, L2_norm_std, L2_normalized_mean, L2_normalized_std, Datas[:, 3, :], Datas[:, 4, :], accuracy_mean, accuracy_std

def plot_result(dirs, model_name, Algorithm_names, grade) :
    plt.figure(figsize= (10,4)) 
    lists = []
    
    P = np.load(dirs + 'params_realworld/'+ 'Theta_'+ model_name + '_4.npy')
    Q = np.load(dirs + 'params_realworld/'+ 'Query_'+ model_name + '_4.npy')
    R = np.load(dirs + 'params_realworld/' + model_name + '_Dataset.npy')
    best_threshold, best_accuracy = threshold_check(P,Q,R,model_name)
    
    print("Origin Accuracy is : ")
    print(round(best_accuracy,3))
    print("Threshold is : ")
    print(round(best_threshold,3))
    print("")

    for Algorithm_name in Algorithm_names :
        directory_path = dirs +  'realworld_' + model_name + '_' + Algorithm_name
        log_det_mean, log_det_s, L2_norm_mean, L2_norm_s, L2_normalized_mean, L2_normalized_std, times, update_times, accuracy_mean, accuracy_s = get_result_std_based(directory_path, P,Q,R, model_name, best_threshold)

        plt.subplot(1,2,1)
        plt.plot(log_det_mean, label=Algorithm_name)
        plt.fill_between([t for t in range(len(log_det_mean))],log_det_mean - 0.2*log_det_s, log_det_mean + 0.2*log_det_s, alpha = 0.2)          
        plt.xlabel(r"Time $t$",fontsize=15)
        plt.ylabel("log determinant of hessian" + "\n" + r"$\log\det\Lambda_t$",fontsize = 12)
        plt.xlim([0,len(L2_norm_mean)-1])
        
        plt.grid()
        plt.legend(fontsize=13)
        plt.tight_layout()

        plt.subplot(1,2,2)
        plt.plot(L2_norm_mean, label = Algorithm_name)
        plt.fill_between([t for t in range(len(L2_norm_mean))],L2_norm_mean - 0.2*L2_norm_s, L2_norm_mean + 0.2*L2_norm_s, alpha = 0.2)  
        plt.xlabel(r"Time $t$",fontsize=15)
        plt.ylabel("Estimate error"+ '\n'+r"$\Vert \hat{\theta}_t - \theta^* \Vert_2$", fontsize = 12)
        plt.xlim([0,len(L2_norm_mean)-1])
        plt.grid()
        plt.legend(fontsize=13)
        plt.tight_layout()

        if Algorithm_name == 'DURM' :
            Time = update_times + times
        else :
            Time = times

        Tot_time = np.sum(Time,axis = 1)
        lists.append([Algorithm_name , round(log_det_mean[-1],1), round(log_det_s[-1],1), round(L2_norm_mean[-1],2), round(L2_norm_s[-1],2), 
                      round(np.mean(Tot_time),1), round(np.std(Tot_time),1), round(accuracy_mean,3), round(accuracy_s,3)])
    plt.tight_layout()
    columns = ["algorithm name", "log det mean", "log det time", "Estimation error mean", "Estimation error std", "time mean", "time std", "acc mean", "acc std"]
    df = pd.DataFrame(lists, columns=columns)
    print(df)
    plt.show()
    plt.savefig(dirs.split('/')[0] + '/'  + dirs.split('/')[1] + '_' + model_name + "_figure" + '.pdf')

Fig_number = 8
grade = 7
model_name = 'LN'

dirs = './Figures/Figure_' + str(Fig_number) + '/Grade' + str(grade) + '/'

print(str(grade) + " grade data " + model_name + " \n")
plot_result(dirs, model_name,['DURM','BALD','MaxEnt','VarRatio','MeanStd'], grade)