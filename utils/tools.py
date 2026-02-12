import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.utils import column_or_1d, assert_all_finite, check_consistent_length
from tqdm import tqdm
from exp.metrics import RocAUC, PrAUC

plt.switch_backend('agg')


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
    num_of_anom = int(len(anom_indexes)*(ratio/100))
    scores = np.hstack((test_scores[anom_indexes[0:num_of_anom]], vali_scores))
    labels = np.hstack((np.ones(num_of_anom), np.zeros(vali_scores.shape[0])))
    return scores, labels

def calculate_limited_test(test_scores, test_labels, ratio, isAllClean=False, isNoAnomaly=False):
    anom_indexes = np.where(test_labels == 1)[0]
    np.random.shuffle(anom_indexes)
    clean_indexes = np.where(test_labels == 0)[0]
    np.random.shuffle(clean_indexes)
    num_of_anom = int(len(anom_indexes)*(ratio/100))
    num_of_clean = int(len(clean_indexes)*(ratio/100))
    if(isAllClean):
        num_of_clean = len(clean_indexes)
    if(isNoAnomaly):
        num_of_anom = 0
    scores = np.hstack((test_scores[anom_indexes[0:num_of_anom]], test_scores[clean_indexes[0:num_of_clean]]))
    labels = np.hstack((np.ones(num_of_anom), np.zeros(num_of_clean)))
    return scores, labels

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        #m.bias.data.fill_(0.01)

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
    
