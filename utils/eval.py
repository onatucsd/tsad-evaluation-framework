import numpy as np
from sklearn.metrics import *
from exp.metrics import RocAUC, PrAUC
# import time

def calculate_roc_auc(test_labels, test_scores):
    # measure prediction performance before threshold calculation
    auroc = RocAUC().score(test_labels, test_scores) 
    aupr = PrAUC().score(test_labels, test_scores)
    print("ROC-AUC : {:0.4f}, PR-AUC : {:0.4f} ".format(auroc, aupr))

def dyn_th_calculate_metrics(labels, preds):
    # measure prediction performance for pre-calculated preds
    print("ROC-AUC Score:", roc_auc_score(labels, preds))
    print("PR-AUC Score:", average_precision_score(labels, preds))
    print("F-Score:", f1_score(labels, preds))
    print("Composite F-score:", get_composite_fscore(preds, get_events(labels), labels))
    print("F-PA Score:", f1_score(labels, pak_wo_th(preds, labels, k=0)))
    print("F-PAK Score:", f1_score(labels, pak_wo_th(preds, labels, k=20)))
    print("F-K-AUC Score:", calculate_f_k_auc_wo_th(labels, preds, max_k=100))

def calculate_metrics(labels, scores, th):
    # measure prediction performance given optimal threshold
    preds = np.where(scores > th, 1, 0)
    #print("Accuracy:", accuracy_score(labels, preds))
    #print("Precision:", precision_score(labels, preds))
    #print("Recall:", recall_score(labels, preds))
    #print("ROC-AUC Score:", roc_auc_score(labels, preds))
    #print("PR-AUC Score:", average_precision_score(labels, preds))
    print("F-Score:", f1_score(labels, preds))
    print("Composite F-score:", get_composite_fscore(preds, get_events(labels), labels))
    print("F-PA Score:", f1_score(labels, pak(scores, labels, th, k=0)))
    print("F-PAK Score:", f1_score(labels, pak(scores, labels, th, k=20)))
    print("F-K-AUC Score:", calculate_f_k_auc(labels, scores, th, max_k=100))
    print("ROC-K-AUC Score:", calculate_roc_k_auc(labels, scores))

def get_composite_fscore(pred_labels, true_events, y_test, return_prec_rec=False):
    # composite F-score: use time-wise precision (PR[t]) and event-based recall (R[e]) to calculate the F1-score 
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp/(tp + fn)
    prec_t = precision_score(y_test, pred_labels)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c

def get_events(y_test, outlier=1, normal=0, breaks=[]):
    # returns a dict about event start and end
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
            elif tim in breaks:
                # A break point was hit, end current event and start new one
                event_end = tim - 1
                events[event] = (event_start, event_end)
                event += 1
                event_start = tim
        else:
            # event_by_time_true[tim] = 0
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    
    return events

def pak(scores, actuals, thres, k):
    # PA%K evaluation protocol: https://ojs.aaai.org/index.php/AAAI/article/view/20680
    preds = (scores > thres).astype(int)
    actuals = actuals.astype(int)

    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(preds))

    for i in range(len(one_start_idx)):
        if preds[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            preds[one_start_idx[i]:zero_start_idx[i]] = 1

    return preds

def pak_wo_th(preds, actuals, k):
    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(preds))

    for i in range(len(one_start_idx)):
        if preds[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            preds[one_start_idx[i]:zero_start_idx[i]] = 1

    return preds

def calculate_f_k_auc(labels, scores, threshold, max_k=100):
    # measure the F1-score at different levels of K and compute an area under this curve
    ks = [k/100 for k in range(0, max_k+1, 10)]
    f1s = []
    for k in range(0, max_k+1, 10):
        adjusted_preds = pak(scores, labels, threshold, k=k)
        f1 = f1_score(labels, adjusted_preds)
        f1s.append(f1)
    
    area_under_f1 = auc(ks, f1s)
    return area_under_f1

def calculate_f_k_auc_wo_th(labels, preds, max_k=100):
    ks = [k/100 for k in range(0, max_k+1, 10)]
    f1s = []
    for k in range(0, max_k+1, 10):
        adjusted_preds = pak_wo_th(preds, labels, k=k)
        f1 = f1_score(labels, adjusted_preds)
        f1s.append(f1)
    
    area_under_f1 = auc(ks, f1s)
    return area_under_f1

def get_fp_tp_rate(preds, actuals):
    tn, fp, fn, tp = confusion_matrix(actuals, preds, labels=[0, 1]).ravel()
    true_positive_rate = tp/(tp+fn)
    false_positive_rate = fp/(fp+tn)
    return false_positive_rate, true_positive_rate

def pak_protocol_only_roc(scores, labels, threshold, max_k=100):
    fprs = []
    tprs = []
    for k in range(0, max_k+1, 10):
        #print("K value:", k)
        adjusted_preds = pak(scores, labels, threshold, k=k)
        fpr, tpr = get_fp_tp_rate(adjusted_preds, labels)
        fprs.append(fpr)
        tprs.append(tpr)

    return fprs, tprs

def calculate_roc_k_auc(label, score): 
    # measure the true positive rates and false positive rates across thresholds and K values
    false_pos_rates = []
    true_pos_rates = []    
    thresholds = np.arange(0, score.max(), score.max()/10)

    for thresh in thresholds:
        #print("Threshold value:", thresh)
        fprs, tprs = pak_protocol_only_roc(score, label, thresh)
        false_pos_rates.append(fprs)
        true_pos_rates.append(tprs)

    false_pos_rates = np.array(false_pos_rates).flatten()
    true_pos_rates = np.array(true_pos_rates).flatten()
    sorted_indexes = np.argsort(false_pos_rates) 
    false_pos_rates = false_pos_rates[sorted_indexes]
    true_pos_rates = true_pos_rates[sorted_indexes]
    roc_score = auc(false_pos_rates, true_pos_rates)
    return roc_score
