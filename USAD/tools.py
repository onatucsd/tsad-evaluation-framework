import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from tqdm import tqdm
from metrics import RocAUC, PrAUC
from thresholding import *

plt.switch_backend('agg')

def obtain_data(dataset): 
    return dataset.train, dataset.val, dataset.test, dataset.test_labels

def plot_anomaly_labels(scores, labels):
    opt_th = best_f_score(scores, labels)
    model_preds = np.where(scores > opt_th, 1, 0)
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(np.arange(len(predicted_anomalies)), model_preds, marker='o', markersize=5, linestyle='', color='blue', label='Predicted Anomalies')
    ax.plot(np.arange(len(predicted_anomalies)), scores, color='magenta', linestyle='-', label='Anomaly Scores')
    ax.plot(np.arange(len(true_anomalies)), labels, marker='x', markersize=5, linestyle='', color='red', label='True Anomalies')
    ax.set_xlabel('Time')
    ax.set_ylabel('Anomaly Detection')
    ax.set_title('Comparison of Predicted and True Anomalies')
    ax.legend()
    file_name = 'anomaly_figure.png'
    plt.savefig(file_name)
    plt.show()

def plot_loss(loss_values):
    epochs = list(range(1, len(loss_values) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_values, marker='x', linestyle='-', color='r')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    file_name = 'loss_graph.png'
    plt.savefig(file_name)
    plt.show()

def plot_factors(model, data, scores, test_labels, contam_value, isTest=False):
    normal_indices = np.where(test_labels == 0)[0]
    anomaly_indices = np.where(test_labels == 1)[0]
    plt.figure(figsize=(50, 30))
    plt.plot(normal_indices, scores[normal_indices], marker='o', linestyle='None', color='b', label='Normal')
    plt.plot(anomaly_indices, scores[anomaly_indices], marker='x', linestyle='None', color='r', label='Anomaly')
    plt.xlabel('Sample ID', fontsize=24)
    plt.ylabel('Isolation Forest Score', fontsize=24)
    plt.xticks(fontsize=20, fontweight='normal', color='black') 
    plt.yticks(fontsize=20, fontweight='normal', color='black')
    plt.legend(fontsize=24)
    plt.grid(True)
    file_name = model + '_' + data + '_' + str(contam_value) + '_IF_scores.png'
    if(isTest):
        file_name = model + '_' + data + '_' + str(contam_value) + '_test_included_IF.png'
    plt.savefig(file_name)
    plt.show()

def sliding_time_window(data, window_size=100):
    num_timestamps, num_features = data.shape
    num_windows = num_timestamps - window_size + 1
    windows = np.empty((num_windows, window_size, num_features))
    
    for i in range(num_windows):
        windows[i] = data[i:i + window_size]
    
    return windows

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def pak_updated(predicts, actuals, k=20):
    
    """
    :param scores: anomaly scores
    :param targets: target labels
    :param thres: anomaly threshold
    :param k: PA%K ratio, 0 equals to conventional point adjust and 100 equals to original predictions
    :return: point_adjusted predictions
    """

    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(predicts))

    for i in range(len(one_start_idx)):
        if predicts[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            predicts[one_start_idx[i]:zero_start_idx[i]] = 1

    return predicts

def get_f_score(prec, rec):
    if prec == 0 and rec == 0:
        f_score = 0
    else:
        f_score = 2 * (prec * rec) / (prec + rec)
    return f_score  

#def calculate_threshold_updated(test_scores, test_targets, vali_scores, percentile_range, pa=True, k=0):
     
    # method 1: calculate the threshold based on nominal anomaly scores
    #roc_max = 0
    #for percentile in percentile_range:
    #    print("Percentile: ", percentile)
    #    threshold = np.percentile(vali_scores, percentile)
        #print("Threshold: ", threshold)
    #    preds = (test_scores > threshold).astype(int)
    #    roc_score = metrics.roc_auc_score(test_targets, preds)
    #    if(roc_score > roc_max):
    #        roc_max = roc_score
    #        threshold_max = threshold
    #        percentile_max = percentile
    
    #print("Percentile max: ", percentile_max)
    #print("Threshold max: ", threshold_max)
    #return threshold_max

    # method 2: calculate the threshold based on pak scores
    #roc_max_pak = 0
    #for percentile in percentile_range:
    #    print("Percentile: ", percentile)
    #    threshold = np.percentile(vali_scores, percentile)
    #    preds = pak(test_scores, test_targets, threshold, k=20) # use default k=20
    #    preds = np.array(preds)
    #    roc_score_pak = metrics.roc_auc_score(test_targets, preds)
    #    if(roc_score_pak > roc_max_pak):
    #        roc_max_pak = roc_score_pak
    #        threshold_max = threshold
    #        percentile_max = percentile
    
    #print("Percentile max: ", percentile_max)
    #print("Threshold max: ", threshold_max)
    #return threshold_max


def calculate_threshold_greedy(th_scores, scores, targets, k):
    '''
    greedy threshold calculation based on combined vali and test scores:
    1. sample n values within the score range
    2. calculate pak values for the selected values
    3. measure the performance under modified scores
    4. pick the threshold that gives the maximum F1 score (or any metric)
    '''
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

'''
def calculate_threshold(scores, targets, pa=True, interval=10, k=20):
    """
    :param scores: list or np.array or tensor, anomaly score
    :param targets: list or np.array or tensor, target labels
    :param pa: True/False
    :param interval: threshold search interval
    :param k: PA%K threshold
    :return: results dictionary
    """
    assert len(scores) == len(targets)
    results = {}

    precision, recall, threshold = metrics.precision_recall_curve(targets, scores)
    f1_score = 2 * precision * recall / (precision + recall + 1e-12)
    print("Threshold length: ", len(threshold))
    # print("Threshold: ", threshold)
    print("F1 Score length: ", len(f1_score))

    #results['best_f1_wo_pa'] = np.max(f1_score)
    #results['best_precision_wo_pa'] = precision[np.argmax(f1_score)]
    #results['best_recall_wo_pa'] = recall[np.argmax(f1_score)]
    #results['prauc_wo_pa'] = metrics.average_precision_score(targets, scores)
    #results['auc_wo_pa'] = metrics.roc_auc_score(targets, scores)
    
    #opt_threshold = threshold[np.argmax(f1_score)]
    #print("Optimum threshold: ", opt_threshold)
    #return opt_threshold

    #return results

    if pa:
        # find F1 score with optimal threshold of best_f1_wo_pa
        pa_scores = pak(scores, targets, threshold[np.argmax(f1_score)], k)
        results['raw_f1_w_pa'] = metrics.f1_score(targets, pa_scores)
        results['raw_precision_w_pa'] = metrics.precision_score(targets, pa_scores)
        results['raw_recall_w_pa'] = metrics.recall_score(targets, pa_scores)
        
        # find best F1 score with varying thresholds
        if len(scores) // interval < 1:
            ths = threshold
        else:
            ths = [threshold[interval*i] for i in range(len(threshold)//interval)]
        
        print("Threshold list: ", ths)
        pa_f1_scores = [metrics.f1_score(targets, pak(scores, targets, th, k)) for th in tqdm(ths)]
        pa_f1_scores = np.asarray(pa_f1_scores)
        results['best_f1_w_pa'] = np.max(pa_f1_scores)
        results['best_f1_th_w_pa'] = ths[np.argmax(pa_f1_scores)]

        pa_scores = pak(scores, targets, ths[np.argmax(pa_f1_scores)], k)
        results['best_precision_w_pa'] = metrics.precision_score(targets, pa_scores)
        results['best_recall_w_pa'] = metrics.recall_score(targets, pa_scores)
        results['pa_f1_scores'] = pa_f1_scores
    
    return results
'''

def pak(scores, actuals, thres, k):
    
    """
    :param scores: anomaly scores
    :param targets: target labels
    :param thres: anomaly threshold
    :param k: PA%K ratio, 0 equals to conventional point adjust and 100 equals to original predictions
    :return: point_adjusted predictions
    """
    predicts = (scores > thres).astype(int)
    actuals = actuals.astype(int)
    # predicts = scores > thres
    # actuals = targets > 0.01

    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(predicts))

    for i in range(len(one_start_idx)):
        if predicts[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            predicts[one_start_idx[i]:zero_start_idx[i]] = 1

    return predicts

def calculate_performance(pred, real, threshold, k_value):
    pred = (pred > threshold).astype(int)
    gt = real.astype(int)
    pred = pak_updated(pred, gt, k=k_value)
    pred = np.array(pred)
    accuracy = accuracy_score(gt, pred)
    # roc_score = roc_auc_score(gt, pred)
    auroc = RocAUC().score(gt, pred) 
    aupr = PrAUC().score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, ROC-AUC-score : {:0.4f}, PR-AUC-score : {:0.4f}".format(
            accuracy, precision, recall, f_score, auroc, aupr))

def calculate_limited_vali_test(vali_scores, test_scores, test_labels, ratio):
    anom_indexes = np.where(test_labels == 1)[0]
    np.random.shuffle(anom_indexes)
    print("Total number of anomalies:", len(anom_indexes))
    num_of_anom = int(len(anom_indexes)*(ratio/100))
    print("Limited number of anomalies:", num_of_anom)
    scores = np.hstack((test_scores[anom_indexes[0:num_of_anom]], vali_scores))
    labels = np.hstack((np.ones(num_of_anom), np.zeros(vali_scores.shape[0])))
    return scores, labels

def calculate_limited_test(test_scores, test_labels, ratio, isAllClean=False, isNoAnomaly=False):
    anom_indexes = np.where(test_labels == 1)[0]
    #np.random.shuffle(anom_indexes)
    clean_indexes = np.where(test_labels == 0)[0]
    #np.random.shuffle(clean_indexes)
    num_of_anom = int(len(anom_indexes)*(ratio/100))
    num_of_clean = int(len(clean_indexes)*(ratio/100))
    if(isAllClean):
        num_of_clean = len(clean_indexes)
    if(isNoAnomaly):
        num_of_anom = 0
    scores = np.hstack((test_scores[anom_indexes[0:num_of_anom]], 
                                        test_scores[clean_indexes[0:num_of_clean]]))
    labels = np.hstack((np.ones(num_of_anom), np.zeros(num_of_clean)))
    return scores,labels

def obtain_test_clean(test_scores, test_labels, ratio):
    clean_indexes = np.where(test_labels == 0)[0]
    np.random.shuffle(clean_indexes)
    num_of_clean = int(len(clean_indexes)*(ratio/100))
    scores = test_scores[clean_indexes[0:num_of_clean]]
    return scores

def obtain_test_mixed(test_scores, test_labels, ratio):
    anom_indexes = np.where(test_labels == 1)[0]
    clean_indexes = np.where(test_labels == 0)[0]
    np.random.shuffle(anom_indexes)
    np.random.shuffle(clean_indexes)
    num_of_clean = int(len(clean_indexes)*(ratio/100))
    num_of_anom = int(len(anom_indexes)*(ratio/100))
    scores = np.vstack((test_scores[anom_indexes[0:num_of_anom], :], 
                                        test_scores[clean_indexes[0:num_of_clean], :]))
    return scores

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        #m.bias.data.fill_(0.01)

def save_val_encoded(model, data, encoded, isTest=False):
    directory = './encoded/' + data 
    file_name = model + '_vali.npy'
    if(isTest):
        file_name = model + '_test.npy'
    file_path = os.path.join(directory, file_name)
    np.save(file_path, encoded)
    
def load_val_encoded(model, data, isTest=False):
    pre_calc_errors = np.empty((0,)) 
    directory = './encoded/' + data 
    file_name = model + '_vali.npy'
    if(isTest):
        file_name = model + '_test.npy'
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        pre_calc_errors = np.load(file_path)
    else:
        print("Please first calculate the errors!")
    
    return pre_calc_errors

def save_val_errors(model, data, errors, isTest=False, isCase2=False, isCase3=False):
    directory = './errors/' + data 
    file_name = model + '_vali.npy'
    if(isTest):
        file_name = model + '_test.npy'    
    if(isCase2):
        file_name = model + '_vali_Case2.npy' 
        if(isTest):
            file_name = model + '_test_Case2.npy'
    elif(isCase3):
        file_name = model + '_vali_Case3.npy' 
        if(isTest):
            file_name = model + '_test_Case3.npy' 
    file_path = os.path.join(directory, file_name)
    np.save(file_path, errors)
    #np.savez(file_path, errors)

def load_val_errors(model, data, isTest=False, isCase2=False, isCase3=False):
    pre_calc_errors = np.empty((0,)) 
    directory = './errors/' + data 
    file_name = model + '_vali.npy'
    if(isTest):
        file_name = model + '_test.npy'
    if(isCase2):
        file_name = model + '_vali_Case2.npy' 
        if(isTest):
            file_name = model + '_test_Case2.npy' 
    elif(isCase3):
        file_name = model + '_vali_Case3.npy' 
        if(isTest):
            file_name = model + '_test_Case3.npy' 
    
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        pre_calc_errors = np.load(file_path)
    else:
        print("Please first calculate the errors!")
    
    return pre_calc_errors
    
def save_one_class_scores(model, scores, classifier, contam_value, isTest=False):
    directory = './traditional_scores'
    file_name = model + '_' + classifier + '_' + str(contam_value) + '.npy'
    if(isTest):
        file_name = model + '_' + classifier + '_' + str(contam_value) + '_test_included.npy'
    file_path = os.path.join(directory, file_name)
    np.save(file_path, scores)
    
def load_one_class_scores(model, classifier, contam_value, isTest=False):
    directory = './traditional_scores'
    file_name = model + '_' + classifier + '_' + str(contam_value) + '.npy'
    if(isTest):
        file_name = model + '_' + classifier + '_' + str(contam_value) + '_test_included.npy'
    file_path = os.path.join(directory, file_name)
    one_class_scores = np.load(file_path)
    return one_class_scores

def save_labels(data, labels):
    directory = './labels'
    file_name = data + '_test_labels.npy'
    file_path = os.path.join(directory, file_name)
    np.save(file_path, labels)
    
def load_labels(data):
    pre_calc_labels = np.empty((0,)) 
    directory = './labels'
    file_name = data + '_test_labels.npy'
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        pre_calc_labels = np.load(file_path)
    else:
        print("Please first calculate the labels!")
    
    return pre_calc_labels

def save_vali_scores(model, data, scoring_function, scores, isCase2=False, isCase3=False, isMax=False):
    directory = './scores/' + data 
    file_name = model + '_' + scoring_function + '_vali.npy'
    if(isMax):
        file_name = model + '_' + scoring_function + 'max_vali.npy' 
    elif(isCase2):
        file_name = model + '_' + scoring_function + '_vali_Case2.npy' 
    elif(isCase3):
        file_name = model + '_' + scoring_function + '_vali_Case3.npy' 
    file_path = os.path.join(directory, file_name)
    np.save(file_path, scores)
    
def load_vali_scores(model, data, scoring_function, isCase2=False, isCase3=False, isMax=False):
    pre_calc_scores = np.empty((0,)) 
    directory = './scores/' + data 
    file_name = model + '_' + scoring_function + '_vali.npy'
    if(isMax):
        file_name = model + '_' + scoring_function + 'max_vali.npy' 
    elif(isCase2):
        file_name = model + '_' + scoring_function + '_vali_Case2.npy' 
    elif(isCase3):
        file_name = model + '_' + scoring_function + '_vali_Case3.npy' 
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        pre_calc_scores = np.load(file_path)
    else:
        print("Please first calculate the vali scores!")
    
    return pre_calc_scores

def save_scores(model, data, scoring_function, scores, isCase2=False, isCase3=False, isMax=False):
    directory = './scores/' + data 
    file_name = model + '_' + scoring_function + '.npy'
    if(isMax):
        file_name = model + '_' + scoring_function + '_max.npy'
    if(isCase2):
        file_name = model + '_' + scoring_function + '_Case2.npy' 
    elif(isCase3):
        file_name = model + '_' + scoring_function + '_Case3.npy' 
    file_path = os.path.join(directory, file_name)
    np.save(file_path, scores)

def load_scores(model, data, scoring_function, isCase2=False, isCase3=False, isMax=False):
    pre_calc_scores = np.empty((0,)) 
    directory = './scores/' + data 
    file_name = model + '_' + scoring_function + '.npy'
    if(isMax):
        file_name = model + '_' + scoring_function + '_max.npy'
    if(isCase2):
        file_name = model + '_' + scoring_function + '_Case2.npy'
    elif(isCase3):
        file_name = model + '_' + scoring_function + '_Case3.npy'
    
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        pre_calc_scores = np.load(file_path)
    else:
        print("Please first calculate the scores!")
    
    return pre_calc_scores

def plot_losses(train_losses, val_losses, test_losses, epochs, model, data):
    
    # Plot losses with different colors
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o', color='orange')
    #plt.plot(epochs, test_losses, label='Test Loss', marker='o', color='green')
    
    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.title(model + '-' + data + '-Training, Validation, and Test Losses')
    plt.title(model + '-' + data + '-Training-Validation Losses')
    
    # Add legend
    plt.legend()
    
    # Show plot
    plt.grid(True)
    directory = './figures/' + data
    file_name = model + '_loss_plot.png'
    file_path = os.path.join(directory, file_name)
    plt.savefig(file_path)
    plt.show()


def plot_reconstruction_errors(model, data, test_errors, test_labels):
    # Plot reconstruction error for each channel
    channels = []
    anom_errors = []
    non_anom_errors = []
    for channel_id in range(test_errors.shape[1]):
        channels.append(str(channel_id))
        ch_errors = test_errors[:, channel_id]
        anom_er = np.mean(ch_errors[test_labels == 1])
        anom_errors.append(anom_er)
        nonanon_er = np.mean(ch_errors[test_labels == 0])
        non_anom_errors.append(nonanon_er)

    bar_width = 0.35
    bar_positions1 = np.arange(len(channels))  
    bar_positions2 = bar_positions1 + bar_width
    plt.figure(figsize=(30, 18))
    plt.bar(bar_positions1, anom_errors, color='red', edgecolor='black', width=bar_width, label='Anomaly')
    plt.bar(bar_positions2, non_anom_errors, color='green', edgecolor='black', width=bar_width, label='Clean')

    plt.xlabel('Channel', fontsize=20)
    plt.ylabel('Reconstruction Error', fontsize=20)
    plt.xticks(bar_positions1 + bar_width / 2, channels) 
    plt.xticks(fontsize=16, fontweight='normal', color='black') 
    plt.yticks(fontsize=16, fontweight='normal', color='black')
    plt.legend(fontsize=20)
    file_name = model + '_' + data + '_reconstruction-bar.png'
    plt.savefig(file_name)
    plt.show()
    
def plot_factors(model, data, scores, test_labels, contam_value, isTest=False):
    normal_indices = np.where(test_labels == 0)[0]
    anomaly_indices = np.where(test_labels == 1)[0]
    plt.figure(figsize=(50, 30))
    plt.plot(normal_indices, scores[normal_indices], marker='o', linestyle='None', color='b', label='Normal')
    plt.plot(anomaly_indices, scores[anomaly_indices], marker='x', linestyle='None', color='r', label='Anomaly')
    plt.xlabel('Sample ID', fontsize=24)
    plt.ylabel('Isolation Forest Score', fontsize=24)
    plt.xticks(fontsize=20, fontweight='normal', color='black') 
    plt.yticks(fontsize=20, fontweight='normal', color='black')
    plt.legend(fontsize=24)
    plt.grid(True)
    file_name = model + '_' + data + '_' + str(contam_value) + '_IF_scores.png'
    if(isTest):
        file_name = model + '_' + data + '_' + str(contam_value) + '_test_included_IF.png'
    plt.savefig(file_name)
    plt.show()