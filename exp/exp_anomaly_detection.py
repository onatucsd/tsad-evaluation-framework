from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import *
from utils.eval import *
from utils.scoring import *
from utils.thresholding import *
from exp.metrics import RocAUC, PrAUC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
            super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        #if self.args.use_multi_gpu and self.args.use_gpu:
        #    model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')
        
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # check TAD baselines first
        isCase1, isCase2, isCase3 = False, False, False  
        if(self.args.baseline == 1):
            isCase1 = True
        elif(self.args.baseline == 2):
            isCase2 = True
        else:
            isCase3 = True

        # dynamic thresholding
        if(self.args.th_function == "Dyn-Th"):
            print("************Dynamic Thresholding************")
            dyn_labels, dyn_preds = dynamic_thresholding(self.model, self.device, test_loader, isCase1, isCase2, isCase3)
            dyn_th_calculate_metrics(dyn_labels, dyn_preds)
            return
        
        # scoring_functions = {ML, MoC, NE, GS, GD}
        scoring_function = self.args.sc_function
        if(scoring_function == "ML"):
            print("************ML-based Scores************") 
            val_errors = calculate_ml_val_errors(self.model, self.device, vali_loader, isCase2, isCase3)
            test_errors = calculate_ml_val_errors(self.model, self.device, test_loader, isCase2, isCase3)
            test_labels = load_labels(self.args.data)
            
            print("******************Local Outlier Factor (LOF)******************")
            contam_value = 0.1
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contam_value, n_jobs=-1)
            lof.fit(val_errors)
            test_scores = lof.decision_function(test_errors) 
            print("******************Isolation forest******************")
            contam_value = 0.1
            isolation_forest = IsolationForest(n_estimators=100, contamination=contam_value, n_jobs=-1)
            isolation_forest.fit(val_errors)
            test_scores = isolation_forest.decision_function(test_errors) 
            print("******************SGD One-class SVM (SGD-OCSVM)******************")
            nu_val = 0.1
            svm_model = SGDOneClassSVM(nu=nu_val)
            svm_model.fit(val_errors)
            test_scores = svm_model.score_samples(test_errors)
            
            test_scores = -test_scores
            
        if(scoring_function == "MoC"):
            print("************Mean over Channels (MoC) Scores************")  
            vali_scores = calculate_vali_scores(self.model, self.device, vali_loader, isCase2, isCase3)
            test_scores, test_labels = calculate_test_scores(self.model, self.device, test_loader, isCase2, isCase3)
        elif(scoring_function == "NE"): 
            print("************Normalized Scores************")
            test_scores, test_labels = calculate_normalized_scores(self.model, self.device, vali_loader, test_loader, isCase2, isCase3)
        elif(scoring_function == "GS"): 
            print("************Gauss-S Scores************")
            test_scores, test_labels = calculate_gauss_s_scores(self.model, self.device, vali_loader, test_loader, isCase2, isCase3)
        else:
            print("************Gauss-D Scores************")
            test_scores, test_labels = calculate_gauss_d_scores(self.model, self.device, test_loader, isCase2, isCase3)

        if(isCase1):    
            test_scores = np.random.uniform(0, np.max(test_scores), len(test_labels))  
        
        if(self.args.th_idp): # threshold independent evaluation
            calculate_roc_auc(test_labels, test_scores)
            
        else:
            # th_functions = {Top-k, Best-F, Best-F-Eff, Tail-p, Dynamic}
            thresholding_function = self.args.th_function
            
            # PA analysis to calculate scores over multiple ratios
            ratios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            for ratio in ratios:
                print("Ratio:", ratio)
                print("*********************Equal anomaly and clean*********************")
                lim_scores, lim_labels = calculate_limited_test(test_scores, test_labels, ratio, False, False)
                opt_th = best_f_score(lim_scores, lim_labels)
                calculate_metrics(test_labels, test_scores, opt_th)
                print("*********************Whole clean*********************")
                lim_scores, lim_labels = calculate_limited_test(test_scores, test_labels, ratio, True, False)
                opt_th = best_f_score(lim_scores, lim_labels)
                calculate_metrics(test_labels, test_scores, opt_th)
                print("*********************No anomaly*********************")
                lim_scores, lim_labels = calculate_limited_test(test_scores, test_labels, ratio, False, True)
                opt_th = best_f_score(lim_scores, lim_labels)
                calculate_metrics(test_labels, test_scores, opt_th)
            return
            
            lim_scores, lim_labels = calculate_limited_test(test_scores, test_labels, self.args.ratio)
            #lim_scores, lim_labels = calculate_limited_test(vali_scores, test_scores, test_labels, self.args.ratio)
            if(thresholding_function == "Top-k"):
                print("************Top-K Thresholding************")
                anomaly_ratio = (np.sum(test_labels == 1) / len(test_labels))*100
                opt_th = top_k(test_scores, anomaly_ratio) 
            elif(thresholding_function == "Best-F"): 
                print("************Best-F Thresholding************")
                if(self.args.ratio == 100):
                    opt_th = best_f_score(test_scores, test_labels)
                else:
                    opt_th = best_f_score(lim_scores, lim_labels)
            elif(thresholding_function == "Best-F-Eff"): 
                print("************Best-F Efficient Thresholding************")
                if(self.args.ratio == 100):
                    opt_th = best_f_efficient(test_scores, test_labels)
                else:
                    opt_th = best_f_efficient(lim_scores, lim_labels)
            elif(thresholding_function == "Tail-p" and (scoring_function == "GS" or scoring_function == "GD")): 
                print("************Tail-p Thresholding************")
                if(self.args.ratio == 100):
                    opt_th = tail_p(test_scores, test_labels)
                else:
                    opt_th = tail_p(lim_scores, lim_labels)
            elif(thresholding_function == "SPOT"): 
                print("************SPOT************")
                log = spot(test_scores, int(test_scores.shape[0]*0.1), 0.1)
                opt_th = np.array(log['t'])
            elif(thresholding_function == "DSPOT"): 
                print("************DSPOT************")
                log = dspot(test_scores, int(test_scores.shape[0]*0.1), 100, 0.1)
                opt_th = np.array(log['t'])
        
            print("Opt threshold:", opt_th)
        
            # evaluation metrics = {Accuracy, Pr, Rec, ROC-AUC, PR-AUC, F-1, F-C}
            calculate_metrics(test_labels, test_scores, opt_th)
