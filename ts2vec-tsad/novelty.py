from sklearn.ensemble import IsolationForest
import numpy as np

def fit_isolation_forest(vali_data, test_data):
    isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
    isolation_forest.fit(vali_data)
    test_scores = isolation_forest.decision_function(test_data) 
    test_scores = -test_scores
    min_score = np.min(test_scores)
    if(min_score < 0):
        test_scores = test_scores - min_score
    else:
        test_scores = test_scores + min_score
    
    return test_scores