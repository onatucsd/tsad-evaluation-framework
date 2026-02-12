import numpy as np
from sklearn.metrics import *
from utils.tools import pak, get_f_score
from exp.metrics import RocAUC
from math import log
from utils.grimshaw import grimshaw

def top_k(scores, anomaly_ratio):
    """
    :param scores: list or np.array or tensor, test anomaly scores
    """
    return np.percentile(scores, 100 - anomaly_ratio)

def trivial_percentile_val(scores_val, p_val=99):
    """
    :param scores: list or np.array or tensor, test anomaly scores
    :param p_val: 95 is two stds, 99.7 is 3 stds 
    """
    return np.percentile(scores_val, p_val)

def best_f_score(scores, targets, ispak=False):
    """
    :param scores: list or np.array or tensor, test anomaly scores
    :param targets: list or np.array or tensor, test target labels
    :param ispak: True/False
    :return: max threshold
    """
    prec, rec, thresholds = precision_recall_curve(targets, scores)
    
    if(not ispak):
        fscores = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
        opt_num = np.squeeze(np.argmax(fscores))
        opt_thres = thresholds[opt_num]
    
    else: 
        fscores = []
        for threshold in thresholds:
            print("Threshold:", threshold)
            preds = pak(scores, targets, threshold, k=20)
            preds = np.array(preds)
            fscore = f1_score(targets, preds)
            fscores.append(fscore)
        opt_thres = thresholds[np.argmax(fscores)]
            
    return opt_thres  

def best_f_efficient(scores, targets, ispak=False):
    """
    :param scores: list or np.array or tensor, test anomaly scores
    :param targets: list or np.array or tensor, test target labels
    :param ispak: True/False
    :return: max threshold
    """
    NUMBER_OF_SAMPLES = 20
    f1_max = 0
    max_th = 0
    thresholds = np.random.uniform(low=np.percentile(scores, 50), high=np.max(scores), size=NUMBER_OF_SAMPLES)
    #thresholds = np.random.uniform(low=np.percentile(scores, 50), high=np.percentile(scores, 95), size=NUMBER_OF_SAMPLES)
    
    if(not ispak):
        for th in thresholds:
            print("Threshold:", th)
            preds = np.where(scores > th, 1, 0)
            pred_f1 = f1_score(targets, preds)
            if(pred_f1 > f1_max):
                f1_max = pred_f1
                max_th = th
        
    else:
        for th in thresholds:
            print("Threshold:", th)
            pa_scores = pak(scores, targets, th, k=20)
            pa_scores = np.array(pa_scores)
            targets = np.array(targets)
            pred_f1 = f1_score(targets, pa_scores) 
            if(pred_f1 > f1_max):
                f1_max = pred_f1
                max_th = th
    
    return max_th

def tail_p(scores, targets, dim=51):
    """
    :param scores: list or np.array or tensor, test anomaly scores
    :param targets: list or np.array or tensor, test target labels
    :param dim: number of channels
    :return: max threshold
    """
    # th(tail-p) = −mlog10(e) where log10(e):{1, 2, 3, 4, 5} and m: number of channels
    tails = [1, 2, 3, 4, 5]
    f1_max = 0
    threshold_max = 0
    for tail in tails:
        threshold = tail * dim
        preds = np.where(scores > threshold, 1, 0)
        pred_f1 = f1_score(targets, preds)
        if(pred_f1 > f1_max):
            f1_max = pred_f1
            threshold_max = threshold
    print("Maximum F1:", f1_max)
    print("Maximum threshold:", threshold_max)
    return threshold_max

def best_f_score_percentile(scores, targets, vali_scores, percentile_range, pa=True, k=0):
    """
    :param scores: list or np.array or tensor, test anomaly scores
    :param targets: list or np.array or tensor, test target labels
    :param vali_scores: list or np.array or tensor, vali anomaly scores
    :param percentile_range: threshold search interval
    :param pa: True/False
    :param k: PA%K threshold
    :return: max threshold
    """
    f1_max_pak = 0
    for percentile in percentile_range:
        print("Percentile: ", percentile)
        threshold = np.percentile(vali_scores, percentile)
        preds = pak(scores, targets, threshold, k=20) # use default k=20
        preds = np.array(preds)
        f1_score_pak = f1_score(targets, preds)
        if(f1_score_pak > f1_max_pak):
            f1_max_pak = f1_score_pak
            threshold_max = threshold
            percentile_max = percentile
    
    print("Percentile max: ", percentile_max)
    print("Threshold max: ", threshold_max)
    return threshold_max

def greedy_search(th_scores, scores, targets, k):
    """
    greedy threshold calculation based on combined vali and test scores:
    1. sample n values within the score range
    2. calculate pak values for the selected values
    3. measure the performance under modified scores
    4. pick the threshold that gives the maximum F1 score (or any metric)
    """
    NUMBER_OF_SAMPLES = 100
    max_roc_score = 0
    max_th = 0
    thresholds = np.random.uniform(low=np.percentile(th_scores, 50), high=np.max(th_scores), size=NUMBER_OF_SAMPLES)
    
    for th in thresholds:
        print("Threshold value: ", th)
        pa_scores = pak(scores, targets, th, k)
        pa_scores = np.array(pa_scores)
        targets = np.array(targets)
        th_roc_value = RocAUC().score(targets, pa_scores) 
        
        if(th_roc_value > max_roc_score):
            max_roc_score = th_roc_value
            max_th = th
    
    return max_th

def pot(data:np.array, risk:float=1e-4, init_level:float=0.98, num_candidates:int=10, epsilon:float=1e-8) -> float:
    ''' 
    Peak-over-Threshold Alogrithm

    Args:
        data: data to process
        risk: detection level
        init_level: probability associated with the initial threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform
    
    Returns:
        z: threshold searching by pot
        t: init threshold 
    '''
    # Set init threshold
    t = np.sort(data)[int(init_level * data.size)]
    peaks = data[data > t] - t

    # Grimshaw
    gamma, sigma = grimshaw(peaks=peaks, 
                            threshold=t, 
                            num_candidates=num_candidates, 
                            epsilon=epsilon
                            )

    # Calculate Threshold
    r = data.size * risk / peaks.size
    if gamma != 0:
        z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
    else: 
        z = t - sigma * log(r)

    return z, t

def spot(data:np.array, num_init:int, risk:float):
    ''' 
    Streaming Peak over Threshold

    Args:
        data: data to process
        num_init: number of data point selected to init threshold
        risk: detection level

    Returns:
        logs: 't' threshold with dataset length; 'a' anomaly datapoint index
    '''
    logs = {'t': [], 'a': []}

    init_data = data[:num_init]
    rest_data = data[num_init:]

    z, t = pot(init_data)
    k = num_init
    peaks = init_data[init_data > t] - t
    logs['t'] = [z] * num_init

    for index, x in enumerate(rest_data):
        if x > z:
            logs['a'].append(index + num_init)
        elif x > t:
            peaks = np.append(peaks, x - t)
            gamma, sigma = grimshaw(peaks=peaks, threshold=t)
            k = k + 1
            r = k * risk / peaks.size
            z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            k = k + 1

        logs['t'].append(z)
    
    return logs

def dspot(data:np.array, num_init:int, depth:int, risk:float):
    ''' 
    Streaming Peak over Threshold with Drift

    Args:
        data: data to process
        num_init: number of data point selected to init threshold
        depth: number of data point selected to detect drift
        risk: detection level

    Returns: 
        logs: 't' threshold with dataset length; 'a' anomaly datapoint index
    '''
    logs = {'t': [], 'a': []}

    base_data = data[:depth]
    init_data = data[depth:depth + num_init]
    rest_data = data[depth + num_init:]

    for i in range(num_init):
        temp = init_data[i]
        init_data[i] -= base_data.mean()
        np.delete(base_data, 0)
        np.append(base_data, temp)

    z, t = pot(init_data)
    k = num_init
    peaks = init_data[init_data > t] - t
    logs['t'] = [z] * (depth + num_init)

    for index, x in enumerate(rest_data):
        temp = x
        x -= base_data.mean()
        if x > z:
            logs['a'].append(index + num_init + depth)
        elif x > t:
            peaks = np.append(peaks, x - t)
            gamma, sigma = grimshaw(peaks=peaks, threshold=t)
            k = k + 1
            r = k * risk / peaks.size
            z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
            np.delete(base_data, 0)
            np.append(base_data, temp)
        else:
            k = k + 1
            np.delete(base_data, 0)
            np.append(base_data, temp)

        logs['t'].append(z)
    
    return logs