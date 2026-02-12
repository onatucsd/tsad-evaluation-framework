import numpy as np
from sklearn.metrics import *
from metrics import RocAUC, PrAUC

def calculate_roc_auc(test_labels, test_scores):
    auroc = RocAUC().score(test_labels, test_scores)
    return auroc
    #print("ROC-AUC : {:0.4f} ".format(auroc))

def calculate_pr_auc(test_labels, test_scores):
    aupr = PrAUC().score(test_labels, test_scores)
    return aupr
    #print("PR-AUC : {:0.4f} ".format(aupr))

def dyn_th_calculate_metrics(labels, preds):
    # measure prediction performance for pre-calculated preds
    #print("Accuracy:", accuracy_score(labels, preds))
    #print("Precision:", precision_score(labels, preds))
    #print("Recall:", recall_score(labels, preds))
    #print("ROC-AUC Score:", roc_auc_score(labels, preds))
    #print("PR-AUC Score:", average_precision_score(labels, preds))
    print("F-Score:", f1_score(labels, preds))
    #print("Composite F-score:", get_composite_fscore(preds, get_events(labels), labels))
    print("F-PA Score:", f1_score(labels, pak_wo_th(preds, labels, k=0)))
    print("F-PAK Score:", f1_score(labels, pak_wo_th(preds, labels, k=20)))
    print("F-K-AUC Score:", calculate_f_k_auc_wo_th(labels, preds, max_k=100))
    
def calculate_metrics(labels, scores, th):
    # measure prediction performance after threshold calculation
    preds = np.where(scores > th, 1, 0)
    #print("Accuracy:", accuracy_score(labels, preds))
    #print("Precision:", precision_score(labels, preds))
    #print("Recall:", recall_score(labels, preds))
    print("F-Score:", f1_score(labels, preds))
    print("F-PA Score:", f1_score(labels, pak(scores, labels, th, k=0)))
    print("F-PAK Score:", f1_score(labels, pak(scores, labels, th, k=20)))
    print("F-K-AUC Score:", calculate_f_k_auc(labels, scores, th, max_k=100))
    print("Composite F-score:", get_composite_fscore(preds, get_events(labels), labels))
    #print("ROC-AUC Score:", roc_auc_score(labels, preds))
    #print("PR-AUC Score:", average_precision_score(labels, preds))
    #print("Accuracy:", accuracy_score(labels, preds))
    #print("Precision:", precision_score(labels, preds))
    #print("Recall:", recall_score(labels, preds))
    #print("ROC-AUC Score:", roc_auc_score(labels, preds))
    #print("PR-AUC Score:", average_precision_score(labels, preds))
    #print("ROC-K-AUC Score:", calculate_roc_k_auc(labels, scores))

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
        adjusted_preds = pak(scores, labels, threshold, k=k)
        fpr, tpr = get_fp_tp_rate(adjusted_preds, labels)
        fprs.append(fpr)
        tprs.append(tpr)

    return fprs, tprs

def calculate_roc_k_auc(label, score): 
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

def pak_protocol(scores, labels, threshold, max_k=100):
    ks = [k/100 for k in range(0, max_k+1, 10)]
    f1s = []
    fprs = []
    tprs = []
    preds = []
    
    for k in range(0, max_k+1, 10):
        #print("k value:", k)
        adjusted_preds = pak(scores, labels, threshold, k=k)
        f1 = f1_score(labels, adjusted_preds)
        f1s.append(f1)
        fpr, tpr = get_fp_tp_rate(adjusted_preds, labels)
        fprs.append(fpr)
        tprs.append(tpr)
        preds.append(adjusted_preds)
    
    area_under_f1 = auc(ks, f1s)
    max_f1_k = max(f1s)
    k_max = f1s.index(max_f1_k)
    preds_for_max = preds[f1s.index(max_f1_k)]
    
    return area_under_f1, max_f1_k, k_max, preds_for_max, fprs, tprs

def evaluate(score, label, validation_thresh=None):
    false_pos_rates = []
    true_pos_rates = []
    f1s = []
    max_f1s_k = []
    preds = []
    thresholds = np.arange(0, score.max(), score.max()/50)

    max_ks = []
    pairs = []

    for thresh in thresholds:
        print("Threshold value:", thresh)
        f1, max_f1_k, k_max, best_preds, fprs, tprs = pak_protocol(score, label, thresh)
        max_f1s_k.append(max_f1_k)
        max_ks.append(k_max)
        preds.append(best_preds)
        false_pos_rates.append(fprs)
        true_pos_rates.append(tprs)
        f1s.append(f1)
        pairs.extend([(thresh, i) for i in range(101)])
    
    if validation_thresh:
        f1, max_f1_k, max_k, best_preds, _, _ = pak_protocol(score, label, validation_thresh)
    
    else:    
        f1 = max(f1s)
        max_possible_f1 = max(max_f1s_k)
        max_idx = max_f1s_k.index(max_possible_f1)
        max_k = max_ks[max_idx]
        thresh_max_f1 = thresholds[max_idx]
        best_preds = preds[max_idx]
        best_thresh = thresholds[f1s.index(f1)]
    
    roc_max = auc(np.transpose(false_pos_rates)[max_k], np.transpose(true_pos_rates)[max_k])
    
    false_pos_rates = np.array(false_pos_rates).flatten()
    true_pos_rates = np.array(true_pos_rates).flatten()

    sorted_indexes = np.argsort(false_pos_rates) 
    false_pos_rates = false_pos_rates[sorted_indexes]
    true_pos_rates = true_pos_rates[sorted_indexes]
    pairs = np.array(pairs)[sorted_indexes]
    roc_score = auc(false_pos_rates, true_pos_rates)

    #preds = predictions[f1s.index(f1)]
    
    if validation_thresh:
        return {
            'f1': f1,   # f1_k(area under f1) for validation threshold
            'ROC/AUC': roc_score, # for all ks and all thresholds obtained on test scores
            'f1_max': max_f1_k, # best f1 across k values
            'preds': best_preds, # corresponding to best k 
            'k': max_k, # the k value correlated with the best f1 across k=1,100
            'thresh_max': validation_thresh,
            'roc_max': roc_score,
        }
    
    else:
        return {
            'f1': f1,
            'ROC/AUC': roc_score,
            'threshold': best_thresh,
            'f1_max': max_possible_f1, 
            'roc_max': roc_max,
            'thresh_max': thresh_max_f1, 
            'preds': best_preds,
            'k': max_k,
        }, false_pos_rates, true_pos_rates
