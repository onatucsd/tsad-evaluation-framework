import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from scipy import signal
from utils.tools import *

def calculate_tr_val_errors(model, device, laoder, isCase2=False):
        
    train_errors = [] 
    for _, (batch_tr_x, _) in enumerate(laoder):  
        batch_tr_x = batch_tr_x.float().to(device) 
        outputs_tr = model(batch_tr_x, None, None, None)
        if(isCase2):
            outputs_tr = torch.zeros_like(outputs_tr)  
        err_tr = batch_tr_x - outputs_tr
        err_tr = err_tr.detach().cpu().numpy() 
        train_errors.append(err_tr)
        
    return train_errors

def calculate_ml_val_errors(model, device, laoder, isCase2=False, isCase3=False):
    
    if(isCase3):
        model.apply(init_weights)
        model.eval()
        
    train_errors = [] 
    for _, (batch_tr_x, _) in enumerate(laoder):  
        batch_tr_x = batch_tr_x.float().to(device) 
        outputs_tr = model(batch_tr_x, None, None, None)
        if(isCase2):
            outputs_tr = torch.zeros_like(outputs_tr)  
        err_tr = batch_tr_x - outputs_tr
        err_tr = err_tr.detach().cpu().numpy() 
        train_errors.append(err_tr)
    
    train_errors = np.concatenate(train_errors, axis=0)
    train_errors = train_errors.reshape(train_errors.shape[0]*train_errors.shape[1], train_errors.shape[2])
    return train_errors

def calculate_vali_scores(model, device, vali_loader, isCase2=False, isCase3=False):
    
    if(isCase3):
        model.apply(init_weights)
        model.eval()  
    
    vali_scores = []
    anomaly_criterion = nn.MSELoss(reduce=False)   

    for _, (batch_x, batch_y) in enumerate(vali_loader):
        batch_x = batch_x.float().to(device) # shape: [batch=128, window=100, features=51]
        outputs = model(batch_x, None, None, None) 
        if(isCase2):
            outputs = torch.zeros_like(outputs) 
        vali_score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
        vali_score = vali_score.detach().cpu().numpy() 
        vali_scores.append(vali_score)

    vali_scores = np.concatenate(vali_scores, axis=0).reshape(-1)

    return vali_scores

def calculate_test_scores(model, device, test_loader, isCase2=False, isCase3=False):
    # Default scoring function in TSLib (mean over channels)  
    if(isCase3):
        model.apply(init_weights)
        model.eval()
      
    test_scores = []
    test_labels = []
    for _, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.float().to(device) # shape: [batch=128, window=100, features=51]
        outputs = model(batch_x, None, None, None)   
        if(isCase2):
            outputs = torch.zeros_like(outputs) 
        anomaly_criterion = nn.MSELoss(reduce=False)    
        test_score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
        test_score = test_score.detach().cpu().numpy() 
        test_scores.append(test_score)
        test_labels.append(batch_y)

    test_scores = np.concatenate(test_scores, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    # window collapsing
    # test_scores = np.concatenate(test_scores, axis=0)
    # test_scores = np.mean(test_scores, axis=1)
    # test_labels = np.concatenate(test_labels, axis=0)
    # test_labels = np.max(test_labels, axis=1).reshape(-1)

    return test_scores, test_labels

def calculate_normalized_scores(model, device, vali_loader, test_loader, isCase2=False, isCase3=False, isMax=False):
    
    if(isCase3):
        model.apply(init_weights)
        model.eval()
        vali_errors = calculate_tr_val_errors(model, device, vali_loader)
    elif(isCase2):
        vali_errors = calculate_tr_val_errors(model, device, vali_loader, True) 
    else:
        vali_errors = calculate_tr_val_errors(model, device, vali_loader)
    
    vali_errors = np.concatenate(vali_errors, axis=0)
    score_vali = np.mean(vali_errors, axis=0)
    
    test_scores = []
    test_labels = []
    for _, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.float().to(device) 
        outputs_test = model(batch_x, None, None, None)
        if(isCase2):
            outputs_test = torch.zeros_like(outputs_test)  
        err_test = batch_x - outputs_test
        err_test = err_test.detach().cpu().numpy() 
        score_test = err_test - score_vali 
        if(isMax):
            score_test = np.max(score_test, axis=2) # max score across channels
        else:
            score_test = np.sqrt(np.mean((score_test ** 2), axis=2)) # root-mean-square across channels
        test_scores.append(score_test)
        test_labels.append(batch_y)

    test_scores = np.concatenate(test_scores, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    
    return test_scores, test_labels 

def calculate_gauss_s_scores(model, device, vali_loader, test_loader, isCase2=False, isCase3=False):
    
    if(isCase3):
        model.apply(init_weights)
        model.eval()
        vali_errors = calculate_tr_val_errors(model, device, vali_loader)
    elif(isCase2):
        vali_errors = calculate_tr_val_errors(model, device, vali_loader, True)
    else:
        vali_errors = calculate_tr_val_errors(model, device, vali_loader)
    
    vali_errors = np.concatenate(vali_errors, axis=0)
    mean_vali = np.mean(vali_errors, axis=0)
    std_vali = np.std(vali_errors, axis=0)
    constant_std = 0.000001
    std_vali[std_vali == 0] = constant_std

    test_scores = []
    test_labels = []
    distribution = norm(0, 1)
    for _, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.float().to(device) 
        outputs_test = model(batch_x, None, None, None)
        if(isCase2):
            outputs_test = torch.zeros_like(outputs_test)  
        err_test = batch_x - outputs_test
        err_test = err_test.detach().cpu().numpy() 
        score_test = -distribution.logsf((err_test - mean_vali) / std_vali)
        score_test = np.sum(score_test, axis=2)
        test_scores.append(score_test)
        test_labels.append(batch_y)
        
    test_scores = np.concatenate(test_scores, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    
    return test_scores, test_labels

def calculate_gauss_d_scores(model, device, test_loader, isCase2=False, isCase3=False):
    
    if(isCase3):
        model.apply(init_weights)
        model.eval()
    # Window-based dynamic gaussian function using the test data            
    test_scores = []
    test_labels = []
    constant_std = 0.000001
    distribution = norm(0, 1)
    for _, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.float().to(device) # (batch_size, window_size, number_of_channels)
        outputs_test = model(batch_x, None, None, None)
        if(isCase2):
            outputs_test = torch.zeros_like(outputs_test)  
        err_test = batch_x - outputs_test
        err_test = err_test.detach().cpu().numpy() 
        mean_test = np.mean(err_test, axis=1)
        std_test = np.std(err_test, axis=1)
        std_test[std_test == 0] = constant_std
        err_test_reshaped = err_test.reshape(err_test.shape[1], err_test.shape[0], err_test.shape[2])
        score_test = -distribution.logsf((err_test_reshaped - mean_test) / std_test)
        score_test = score_test.reshape(err_test.shape[0], err_test.shape[1], err_test.shape[2])
        score_test = np.sum(score_test, axis=2)
        test_scores.append(score_test)
        test_labels.append(batch_y)
        
    test_scores = np.concatenate(test_scores, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

    return test_scores, test_labels

def calculate_gauss_d_k_scores(model, device, test_loader, isCase2=False, isCase3=False):
    
    if(isCase3):
        model.apply(init_weights)
        model.eval()
    # Gauss-D + convolution over channels              
    test_scores = []
    test_labels = []
    constant_std = 0.000001
    distribution = norm(0, 1)
    for _, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.float().to(device) 
        outputs_test = model(batch_x, None, None, None)
        if(isCase2):
            outputs_test = torch.zeros_like(outputs_test) 
        err_test = batch_x - outputs_test
        err_test = err_test.detach().cpu().numpy() 
        mean_test = np.mean(err_test, axis=1)
        std_test = np.std(err_test, axis=1)
        std_test[std_test == 0] = constant_std
        err_test_reshaped = err_test.reshape(err_test.shape[1], err_test.shape[0], err_test.shape[2])
        gauss_d_score = -distribution.logsf((err_test_reshaped - mean_test) / std_test)
        gauss_d_score = gauss_d_score.reshape(err_test.shape[0], err_test.shape[1], err_test.shape[2])
        test_scores.append(gauss_d_score)
        test_labels.append(batch_y)
    
    test_scores = np.concatenate(test_scores, axis=0) # shape: (449820, 100, 51)
    
    # convolution operation and update test scores
    kernel_sigma = 10
    gaussian_kernel = signal.gaussian(kernel_sigma * 8, std=kernel_sigma)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    conv_test_scores = []
    divider = 64260 # here select a value that can divide total # of test samples
    conv_range = int(test_scores.shape[0]/divider)
    for i in range(conv_range): 
        # print("Convolution iteration:", i)
        conv_score = signal.convolve(test_scores[i*divider:(i+1)*divider,:,:], 
                                     gaussian_kernel[np.newaxis, np.newaxis, :], 
                                     mode='same', method='direct')
        conv_score = np.sum(conv_score, axis=2)
        conv_test_scores.append(conv_score)
    
    test_scores = np.concatenate(conv_test_scores, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

    return test_scores, test_labels

def dynamic_thresholding(model, device, test_loader, channel_id=40, isCase1=False, isCase2=False, isCase3=False):

    # Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding
    # https://arxiv.org/abs/1802.04431

    if(isCase3):
        model.apply(init_weights)
        model.eval()
    
    test_labels = []
    errors = []

    for _, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.float().to(device) # shape: [batch=128, window=100, features=51]
        outputs = model(batch_x, None, None, None)
        if(isCase2):
            outputs = torch.zeros_like(outputs)    
        batch_x = batch_x.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy() 
        err_test = np.mean(batch_x, axis=2) - np.mean(outputs, axis=2) # use average channel values    
        #err_test = batch_x[:,:,channel_id] - outputs[:,:,channel_id] # channel-based error
        err_test = np.abs(err_test)
        errors.append(err_test)
        test_labels.append(batch_y)
    
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    errors = np.concatenate(errors, axis=0) # (n_of_samples, win_size) 

    if(isCase1):
        case1_errors = errors.reshape(-1)
        case1_errors = np.random.uniform(0, np.max(case1_errors), len(test_labels)) 
        errors = case1_errors.reshape(errors.shape[0], errors.shape[1])  
    
    preds = []
    for sample_id in range(errors.shape[0]):
        e = errors[sample_id, :]
        e_s = ewma(e, 0.01) # smoothed errors using EWMA
        
        # Threshold calculation
        mean_e_s = np.mean(e_s)
        std_e_s = np.std(e_s)
        z_values = range(2, 11)
        res_max = -1000
        th_opt = -1
        
        for z_value in z_values: 
            threshold = mean_e_s + z_value * std_e_s
            mean_delta = mean_e_s - np.mean(e_s[e_s < threshold])
            std_delta = std_e_s - np.std(e_s[e_s < threshold])
            e_a = e_s[e_s > threshold]
            e_seq = find_continuous_sequences(e_s > threshold)
            
            denom = len(e_seq)**2 + len(e_a)
            if (denom == 0): # invalid case
                continue
            
            else:
                res = (mean_delta/mean_e_s + std_delta/std_e_s) / denom
                if (res > res_max):
                    res_max = res
                    th_opt = threshold
        
        if(th_opt == -1):
            th_opt = mean_e_s + 2*std_e_s
            
        print("Optimal threshold:", th_opt)
        pred = np.where(e_s > th_opt, 1, 0)
        
        '''
        # False positive mitigation
        # First method: Pruning anomalies
        e_seq = find_continuous_sequences(e_s > th_opt)
        e_max = []
        if(len(e_seq) != 0): 
            #anomaly_scores = []
            for seq in e_seq:
                b_index = seq[0]
                e_index = seq[1]
                seq_max = np.max(e_s[b_index:e_index])
                e_max.append(seq_max)
                #anomaly_score = (seq_max - th_opt)/(mean_e_s + std_e_s)
                #anomaly_scores.append(anomaly_score)
        
            #print("Anomaly scores:", anomaly_scores)
        
        e_seq_non_anom = find_continuous_sequences(e_s < th_opt)
        if(len(e_seq_non_anom) != 0): 
            e_non_anom = []
            for seq in e_seq_non_anom:
                b_index = seq[0]
                e_index = seq[1]
                seq_max = np.max(e_s[b_index:e_index])
                e_non_anom.append(seq_max)
            e_max.append(np.max(e_non_anom))
            
        e_max_sorted = np.sort(e_max)[::-1]
        #print("E_max sorted:", e_max_sorted)
        #d_values = []
        p_value = 0.1
        for i in range(1, len(e_max_sorted)):
            #print("Minimum decrease calculation")
            d_value = (e_max_sorted[i-1] - e_max_sorted[i]) / e_max_sorted[i-1]
            #d_values.append(d_value)
            if(p_value < d_value):
                print("Reclassify as nominal")
                pred = 0
            #else: 
            #    print("Remain as anomalies")
        # Second method: Learning from history
        e_seq = find_continuous_sequences(e_s > th_opt)
        e_max = []
        if(len(e_seq) != 0): 
            anomaly_scores = []
            for seq in e_seq:
                b_index = seq[0]
                e_index = seq[1]
                seq_max = np.max(e_s[b_index:e_index])
                e_max.append(seq_max)
                anomaly_score = (seq_max - th_opt)/(mean_e_s + std_e_s)
                anomaly_scores.append(anomaly_score)
        
            print("Anomaly scores:", anomaly_scores)
            percentile = 50 # can determine this value based on the desired precision-recall balance
            s_min = np.percentile(np.array(anomaly_scores), percentile)
            
            for anom_score in anomaly_scores:
                if(anom_score < s_min):
                    print("Reclassify as nominal")
                    pred = 0
        '''
        preds.append(pred)
    
    preds = np.concatenate(preds, axis=0).reshape(-1)    
    return test_labels, preds

def ewma(data, alpha):
    
    n = len(data)
    ewma_values = np.zeros(n)
    ewma_values[0] = data[0]
    for i in range(1, n):
        ewma_values[i] = alpha * data[i] + (1 - alpha) * ewma_values[i - 1]
    return ewma_values

def find_continuous_sequences(arr):
    
    sequences = []
    start_idx = None

    for i, value in enumerate(arr):
        if value:  
            if start_idx is None:
                start_idx = i
        elif start_idx is not None:
            # Check if the sequence contains at least two True values
            if i - start_idx >= 2:
                sequences.append((start_idx, i - 1))
            start_idx = None

    # Check for the last sequence
    if start_idx is not None and len(arr) - start_idx >= 2:
        sequences.append((start_idx, len(arr) - 1))

    return sequences
