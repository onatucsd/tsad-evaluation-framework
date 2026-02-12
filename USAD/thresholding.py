import numpy as np
from sklearn.metrics import *
from tools import pak, get_f_score
from metrics import RocAUC

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

