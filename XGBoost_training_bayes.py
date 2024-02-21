
from logging.config import valid_ident
import os 
import pickle
import xgboost as xgb
import numpy as np
import scipy
import sklearn
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.feature_selection import RFECV, RFE
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold
from functools import partial
import json
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_validate
print("scipy version :", scipy.__version__)
def train_bayes_model(events_sig, events_bkg , input_features):
    # if input_features list don't contain sigmarv, add it
    if 'sigmarv' not in input_features:
        df_sig = events_sig.arrays(input_features+['sigmarv'],library='pd')
    else:
        df_sig = events_sig.arrays(input_features,library='pd')
    df_pfff = events_bkg['DataDriven_QCD'].arrays(input_features,library='pd')
    df_pp = events_bkg['pp'].arrays(input_features,library='pd')
    df_sig_weight = events_sig.arrays(['weight','weight_absRatio'],library='pd')
    df_bkg_pfff_weight = events_bkg['DataDriven_QCD'].arrays(['weight',"Norm_SFs"],library='pd')
    df_bkg_pp_weight = events_bkg['pp'].arrays(['weight',"Norm_SFs"],library='pd')
    
    ms_weight = (df_sig['vtxprob']*1./df_sig['sigmarv']+(1-df_sig['vtxprob'])*1./df_sig['sigmawv'])
    sig_total_weight = df_sig_weight['weight'] * df_sig_weight['weight_absRatio'] * ms_weight
    sig_total_weight = sig_total_weight.clip(lower=0)
    bkg_pfff_total_weight = df_bkg_pfff_weight['weight'] * df_bkg_pfff_weight['Norm_SFs']

    # small tune to decrease the dd weigth so that the dd weigth would not have the overfitting problem and have the better data/mc agreement plot
    bkg_pfff_total_weight = bkg_pfff_total_weight 

    bkg_pfff_total_weight = bkg_pfff_total_weight.clip(lower=0)
    bkg_pp_total_weight = df_bkg_pp_weight['weight'] * df_bkg_pp_weight['Norm_SFs']
    bkg_pp_total_weight = bkg_pp_total_weight.clip(lower=0)
    reweight_sig_bkg = (sum(bkg_pfff_total_weight) + sum(bkg_pp_total_weight)) / sum(sig_total_weight)
    sig_total_weight = reweight_sig_bkg*sig_total_weight
    #define the dataframe
    sig_dataframe = pd.DataFrame()
    bkg_pp_dataframe = pd.DataFrame()
    bkg_pfff_dataframe = pd.DataFrame()

    sig_dataframe = df_sig[input_features]
    sig_dataframe['target'] = 1
    sig_dataframe['key'] = 'sig'
    sig_dataframe['total_weight'] = sig_total_weight

    bkg_pp_dataframe = df_pp
    bkg_pp_dataframe['target'] = 0
    bkg_pp_dataframe['key'] = 'pp'
    bkg_pp_dataframe['total_weight'] = bkg_pp_total_weight 

    bkg_pfff_dataframe = df_pfff
    bkg_pfff_dataframe['target'] = 0
    bkg_pfff_dataframe['key'] = "pfff"
    bkg_pfff_dataframe['total_weight'] = bkg_pfff_total_weight

    Hgg_dataframe = pd.concat([sig_dataframe,bkg_pp_dataframe,bkg_pfff_dataframe])
    # Hgg_dataframe = pd.concat([sig_dataframe,bkg_pp_dataframe])


    Hgg_dataframe['total_weight'] = Hgg_dataframe['total_weight'].clip(lower= 0)

    # os.chdir("/eos/user/z/zhenxuan/BDT/XGboost/python")
    # from load_data import *
    rng = np.random.RandomState(31337)

    ##############
    ## options ##
    ##############
    FeatureSelection = False
    RFESelection = False
    n_features = 10
    GridSearch = False

    variables = input_features
    print("variables:" ,variables)
    data = Hgg_dataframe
    traindataset, valdataset  = train_test_split(data, test_size=0.3, random_state=rng)
    nS = len(traindataset.iloc[(traindataset.target.values == 1)])
    nB = len(traindataset.iloc[(traindataset.target.values == 0)])
    print("nB/nS:",nB/nS)

    rng = np.random.RandomState(31337)

    pbounds = {
        'learning_rate': (0.001, 0.1),
        'n_estimators': (1000, 10000),
        'max_depth': (3,8),
        'subsample': (0.8, 1.0),  # Change for big datasets
        'colsample_bytree': (0.8, 1.0),  # Change for datasets with lots of features
        'reg_alpha': (0, 5),  
        'reg_lambda': (0, 5), 
        'gamma': (0, 5),
        'min_child_weight':(0,5)}
    def xgboost_hyper_param(learning_rate,
                            n_estimators,
                            max_depth,
                            subsample,
                            colsample_bytree,
                            reg_alpha,
                            reg_lambda,
                            gamma,
                            min_child_weight
                            ):

        max_depth = int(max_depth)
        n_estimators = int(n_estimators)

        clf = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            gamma=gamma,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            scale_pos_weight=nB/nS,
            tree_method='gpu_hist'
            )

        clf.fit(
            traindataset[input_features].values,
            traindataset.target.values,
            sample_weight=(traindataset["total_weight"].values)
            )       
        probaT = clf.predict_proba(valdataset[variables].values.astype(np.float64))
        fprt, tprt, thresholds = roc_curve(valdataset["target"].astype(np.float64), probaT[:,1],sample_weight = valdataset['total_weight'])
        test_auct = auc(np.sort(fprt), np.sort(tprt))
        proba = clf.predict_proba(traindataset[variables].values)
        fpr, tpr, thresholds = roc_curve(traindataset["target"].values,proba[:,1],sample_weight = traindataset['total_weight'].values)
        train_auc = auc(np.sort(fpr), np.sort(tpr))
        modified_test_auct = 1/(abs(train_auc - test_auct) + 1/ test_auct)
        print("ZZ: abs(train_auc - test_auct) + 1/test_auct = ", 1/ (abs(train_auc - test_auct)) + test_auct)
        print("ZZ: test_auct = ",test_auct)
        print("ZZ : reg_alpha = ",reg_alpha)
        return modified_test_auct



        #FIemmi
        #Define a version of cross_val_score that takes into account sample weights when computing the score
        # def cross_val_score_weighted(model, X, y, weights, cv=3, metric=sklearn.metrics.roc_auc_score): #the metric is set to be the roc_auc_score, but it can be changed if needed
        #     kf = KFold(n_splits=cv)
        #     kf.get_n_splits(X)
        #     scores = []
        #     for train_index, test_index in kf.split(X): #perform k-fold splitting by hand
        #         model_clone = sklearn.base.clone(model)
        #         X_train, X_test = X[train_index], X[test_index]
        #         y_train, y_test = y[train_index], y[test_index]
        #         weights_train, weights_test = weights[train_index], weights[test_index]
        #         model_clone.fit(X_train,y_train,sample_weight=weights_train) #fit a the cloned model using the sample weights
        #         probaT = model_clone.predict_proba(X_test)
        #         test_auct = metric(y_test, probaT[:,1],sample_weight = weights_test) #compute the scorer in the same fashion of L165, i.e., using the sample weights correctly
        #         proba = model_clone.predict_proba(X_train)
        #         train_auc = metric(y_train, proba[:,1],sample_weight = weights_train)
        #         scores.append(abs(train_auc - test_auct) + 1/test_auct)
        #     return scores
        
        # #FIemmi
        # #Instead of using the ordinary cross_val_score method, use the custom cross_val_score_weighted
        # #cv_vals = cross_val_score(clf, traindataset[input_features].values,traindataset.target.values, fit_params={'sample_weight':traindataset["total_weight"].values }, cv=3, scoring='roc_auc')
        # cv_vals = cross_val_score_weighted(clf, traindataset[input_features].values,traindataset.target.values, traindataset["total_weight"].values, cv=3)
        # print("cv_vals:", cv_vals)
        # print("roc_auc: {0} +/- {1}".format(np.array(cv_vals).mean(), np.array(cv_vals).std()))
        # print("ZZ: cv_vals mean scores:", -np.mean(cv_vals))
        # return - np.mean(cv_vals)
    """
        probaT = clf.predict_proba(valdataset[variables].values.astype(np.float64))
        score = log_loss(valdataset["target"].values, probaT[:,1],sample_weight = valdataset['total_weight'])
        fprt, tprt, thresholds = roc_curve(valdataset["target"].astype(np.float64), probaT[:,1],sample_weight = valdataset['total_weight'])
        test_auct = auc(np.sort(fprt), np.sort(tprt))
        print("test_auc",test_auct)
        proba = clf.predict_proba(traindataset[variables].values)
        fpr, tpr, thresholds = roc_curve(traindataset["target"].values,proba[:,1],sample_weight = traindataset['total_weight'].values)
        train_auc = auc(np.sort(fpr), np.sort(tpr))
        print("test auc - train auc:", -abs(test_auct - train_auc))
        print("reg_alpha:",reg_alpha)
        # more diagnosis, in case
        # cross_val = cross_validate(clf,traindataset[input_features].values,traindataset.target.values,cv=3,return_train_score=True,fit_params={'sample_weight':traindataset["total_weight"].values } )
        # print('Training sample score: ',cross_val['train_score'])
        # print("test score",cross_val['test_score'].mean())
        # return cross_val['test_score'].mean()
        # print("log func:", score)
        # if (type(score) == type(None)):
        #     score = 10000
        return -abs(test_auct - train_auc)
        # return test_auct
        # return np.mean(cross_val_score(clf, traindataset[input_features].values,traindataset.target.values, cv=3, scoring='loss_function'))
        # return test_auct
        # print("roc_auc:",np.mean(cross_val_score(clf, traindataset[input_features].values,traindataset.target.values,fit_params={'sample_weight':traindataset["total_weight"].values }, cv=3, scoring='roc_auc')))
        # print("roc_auc wo sample weight:",np.mean(cross_val_score(clf, traindataset[input_features].values,traindataset.target.values, cv=3, scoring='roc_auc')))
        # return np.mean(cross_val_score(clf, traindataset[input_features].values,traindataset.target.values,fit_params={'sample_weight':traindataset["total_weight"].values }, cv=3, scoring='roc_auc'))
    """
    optimizer = BayesianOptimization(
        f=xgboost_hyper_param,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(init_points=8,n_iter=300)

    #Extracting the best parameters
    params = optimizer.max['params']
    print(params)

def train_bayes_model_combined(events_sig_16postVFP, events_sig_16preVFP , events_sig_17, events_sig_18,  events_bkg_16postVFP, events_bkg_16preVFP , events_bkg_17, events_bkg_18 ):

    input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv"]
     # ------------------------------- UL16 postVFP ------------------------------- #
    df_sig_16postVFP = events_sig_16postVFP.arrays(input_features,library='pd')
    df_pfff_16postVFP = events_bkg_16postVFP['DataDriven_QCD'].arrays(input_features,library='pd')
    df_pp_16postVFP = events_bkg_16postVFP['pp'].arrays(input_features,library='pd')
    df_sig_weight_16postVFP = events_sig_16postVFP.arrays(['weight'],library='pd')
    df_bkg_pfff_weight_16postVFP = events_bkg_16postVFP['DataDriven_QCD'].arrays(['weight',"Norm_SFs"],library='pd')
    df_bkg_pp_weight_16postVFP = events_bkg_16postVFP['pp'].arrays(['weight',"Norm_SFs"],library='pd')

    sig_total_weight_16postVFP = df_sig_weight_16postVFP['weight']*(df_sig_16postVFP['vtxprob']*1./df_sig_16postVFP['sigmarv']+(1-df_sig_16postVFP['vtxprob'])*1./df_sig_16postVFP['sigmawv'])
    sig_total_weight_16postVFP = sig_total_weight_16postVFP.clip(lower=0)
    bkg_pfff_total_weight_16postVFP = df_bkg_pfff_weight_16postVFP['weight'] * df_bkg_pfff_weight_16postVFP['Norm_SFs']
    bkg_pfff_total_weight_16postVFP = bkg_pfff_total_weight_16postVFP.clip(lower=0)
    bkg_pp_total_weight_16postVFP = df_bkg_pp_weight_16postVFP['weight'] * df_bkg_pp_weight_16postVFP['Norm_SFs']
    bkg_pp_total_weight_16postVFP = bkg_pp_total_weight_16postVFP.clip(lower=0)

    reweight_sig_bkg = (sum(bkg_pfff_total_weight_16postVFP) + sum(bkg_pp_total_weight_16postVFP)) / sum(sig_total_weight_16postVFP)
    sig_total_weight_16postVFP = reweight_sig_bkg*sig_total_weight_16postVFP
    sig_total_weight_16postVFP = sig_total_weight_16postVFP # pay more attention on the signal
    print("ZZ: check signal 16postVFP weight:",sum(sig_total_weight_16postVFP))
    print("ZZ: check bkgPP 16postVFP weight:",sum(bkg_pp_total_weight_16postVFP))
    print("ZZ: check bkgDD 16postVFP weight:",sum(bkg_pfff_total_weight_16postVFP))
    # -------------------------------- UL16 preVFP ------------------------------- #
    df_sig_16preVFP = events_sig_16preVFP.arrays(input_features,library='pd')
    df_pfff_16preVFP = events_bkg_16preVFP['DataDriven_QCD'].arrays(input_features,library='pd')
    df_pp_16preVFP = events_bkg_16preVFP['pp'].arrays(input_features,library='pd')
    df_sig_weight_16preVFP = events_sig_16preVFP.arrays(['weight'],library='pd')
    df_bkg_pfff_weight_16preVFP = events_bkg_16preVFP['DataDriven_QCD'].arrays(['weight',"Norm_SFs"],library='pd')
    df_bkg_pp_weight_16preVFP = events_bkg_16preVFP['pp'].arrays(['weight',"Norm_SFs"],library='pd')

    sig_total_weight_16preVFP = df_sig_weight_16preVFP['weight']*(df_sig_16preVFP['vtxprob']*1./df_sig_16preVFP['sigmarv']+(1-df_sig_16preVFP['vtxprob'])*1./df_sig_16preVFP['sigmawv'])
    sig_total_weight_16preVFP = sig_total_weight_16preVFP.clip(lower=0)
    bkg_pfff_total_weight_16preVFP = df_bkg_pfff_weight_16preVFP['weight'] * df_bkg_pfff_weight_16preVFP['Norm_SFs']
    bkg_pfff_total_weight_16preVFP = bkg_pfff_total_weight_16preVFP.clip(lower=0)
    bkg_pp_total_weight_16preVFP = df_bkg_pp_weight_16preVFP['weight'] * df_bkg_pp_weight_16preVFP['Norm_SFs']
    bkg_pp_total_weight_16preVFP = bkg_pp_total_weight_16preVFP.clip(lower=0)

    reweight_sig_bkg = (sum(bkg_pfff_total_weight_16preVFP) + sum(bkg_pp_total_weight_16preVFP)) / sum(sig_total_weight_16preVFP)
    sig_total_weight_16preVFP = reweight_sig_bkg*sig_total_weight_16preVFP
    sig_total_weight_16preVFP = sig_total_weight_16preVFP # pay more attention on the signal
    print("ZZ: check signal 16preVFP weight:",sum(sig_total_weight_16preVFP))
    print("ZZ: check bkgPP 16preVFP weight:",sum(bkg_pp_total_weight_16preVFP))
    print("ZZ: check bkgDD 16preVFP weight:",sum(bkg_pfff_total_weight_16preVFP))
    # -------------------------------- UL17 ------------------------------- #
    df_sig_17 = events_sig_17.arrays(input_features,library='pd')
    df_pfff_17 = events_bkg_17['DataDriven_QCD'].arrays(input_features,library='pd')
    df_pp_17 = events_bkg_17['pp'].arrays(input_features,library='pd')
    df_sig_weight_17 = events_sig_17.arrays(['weight'],library='pd')
    df_bkg_pfff_weight_17 = events_bkg_17['DataDriven_QCD'].arrays(['weight',"Norm_SFs"],library='pd')
    df_bkg_pp_weight_17 = events_bkg_17['pp'].arrays(['weight',"Norm_SFs"],library='pd')

    sig_total_weight_17 = df_sig_weight_17['weight']*(df_sig_17['vtxprob']*1./df_sig_17['sigmarv']+(1-df_sig_17['vtxprob'])*1./df_sig_17['sigmawv'])
    sig_total_weight_17 = sig_total_weight_17.clip(lower=0)
    bkg_pfff_total_weight_17 = df_bkg_pfff_weight_17['weight'] * df_bkg_pfff_weight_17['Norm_SFs']
    bkg_pfff_total_weight_17 = bkg_pfff_total_weight_17.clip(lower=0)
    bkg_pp_total_weight_17 = df_bkg_pp_weight_17['weight'] * df_bkg_pp_weight_17['Norm_SFs']
    bkg_pp_total_weight_17 = bkg_pp_total_weight_17.clip(lower=0)

    reweight_sig_bkg = (sum(bkg_pfff_total_weight_17) + sum(bkg_pp_total_weight_17)) / sum(sig_total_weight_17)
    sig_total_weight_17 = reweight_sig_bkg*sig_total_weight_17
    sig_total_weight_17 = sig_total_weight_17 # pay more attention on the signal
    print("ZZ: check signal 17 weight:",sum(sig_total_weight_17))
    print("ZZ: check bkgPP 17 weight:",sum(bkg_pp_total_weight_17))
    print("ZZ: check bkgDD 17 weight:",sum(bkg_pfff_total_weight_17))
    # -------------------------------- UL18 ------------------------------- #
    df_sig_18 = events_sig_18.arrays(input_features,library='pd')
    df_pfff_18 = events_bkg_18['DataDriven_QCD'].arrays(input_features,library='pd')
    df_pp_18 = events_bkg_18['pp'].arrays(input_features,library='pd')
    df_sig_weight_18 = events_sig_18.arrays(['weight'],library='pd')
    df_bkg_pfff_weight_18 = events_bkg_18['DataDriven_QCD'].arrays(['weight',"Norm_SFs"],library='pd')
    df_bkg_pp_weight_18 = events_bkg_18['pp'].arrays(['weight',"Norm_SFs"],library='pd')

    sig_total_weight_18 = df_sig_weight_18['weight']*(df_sig_18['vtxprob']*1./df_sig_18['sigmarv']+(1-df_sig_18['vtxprob'])*1./df_sig_18['sigmawv'])
    sig_total_weight_18 = sig_total_weight_18.clip(lower=0)
    bkg_pfff_total_weight_18 = df_bkg_pfff_weight_18['weight'] * df_bkg_pfff_weight_18['Norm_SFs']
    bkg_pfff_total_weight_18 = bkg_pfff_total_weight_18.clip(lower=0)
    bkg_pp_total_weight_18 = df_bkg_pp_weight_18['weight'] * df_bkg_pp_weight_18['Norm_SFs']
    bkg_pp_total_weight_18 = bkg_pp_total_weight_18.clip(lower=0)

    reweight_sig_bkg = (sum(bkg_pfff_total_weight_18) + sum(bkg_pp_total_weight_18)) / sum(sig_total_weight_18)
    sig_total_weight_18 = reweight_sig_bkg*sig_total_weight_18
    sig_total_weight_18 = sig_total_weight_18 # pay more attention on the signal
    print("ZZ: check signal 18 weight:",sum(sig_total_weight_18))
    print("ZZ: check bkgPP 18 weight:",sum(bkg_pp_total_weight_18))
    print("ZZ: check bkgDD 18 weight:",sum(bkg_pfff_total_weight_18))

    #define the dataframe
    sig_dataframe = pd.DataFrame()
    bkg_pp_dataframe = pd.DataFrame()
    bkg_pfff_dataframe = pd.DataFrame()

    sig_dataframe = pd.concat([df_sig_16postVFP, df_sig_16preVFP, df_sig_17, df_sig_18])
    sig_dataframe['target'] = 1
    sig_dataframe['key'] = 'sig'
    sig_dataframe['total_weight'] = pd.concat([sig_total_weight_16postVFP, sig_total_weight_16preVFP, sig_total_weight_17, sig_total_weight_18])

    bkg_pp_dataframe = pd.concat([df_pp_16postVFP, df_pp_16preVFP, df_pp_17, df_pp_18])
    bkg_pp_dataframe['target'] = 0
    bkg_pp_dataframe['key'] = 'pp'
    bkg_pp_dataframe['total_weight'] = pd.concat([bkg_pp_total_weight_16postVFP,bkg_pp_total_weight_16preVFP,bkg_pp_total_weight_17,bkg_pp_total_weight_18])

    bkg_pfff_dataframe = pd.concat([df_pfff_16postVFP,df_pfff_16preVFP,df_pfff_17,df_pfff_18])
    bkg_pfff_dataframe['target'] = 0
    bkg_pfff_dataframe['key'] = "pfff"
    bkg_pfff_dataframe['total_weight'] = pd.concat([bkg_pfff_total_weight_16postVFP, bkg_pfff_total_weight_16preVFP, bkg_pfff_total_weight_17, bkg_pfff_total_weight_18])

    Hgg_dataframe = pd.concat([sig_dataframe,bkg_pp_dataframe,bkg_pfff_dataframe])
    # Hgg_dataframe = pd.concat([sig_dataframe,bkg_pp_dataframe])


    Hgg_dataframe['total_weight'] = Hgg_dataframe['total_weight'].clip(lower= 0)


    # os.chdir("/eos/user/z/zhenxuan/BDT/XGboost/python")
    # from load_data import *
    rng = np.random.RandomState(31337)

    ##############
    ## options ##
    ##############
    FeatureSelection = False
    RFESelection = False
    n_features = 10
    GridSearch = False

    variables = input_features
    print("variables:" ,variables)
    data = Hgg_dataframe
    traindataset, valdataset  = train_test_split(data, test_size=0.3, random_state=rng)
    nS = len(traindataset.iloc[(traindataset.target.values == 1)])
    nB = len(traindataset.iloc[(traindataset.target.values == 0)])
    print("nB/nS:",nB/nS)

    rng = np.random.RandomState(31337)

    input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv"]
    pbounds = {
        'learning_rate': (0.001, 0.1),
        'n_estimators': (1000, 10000),
        'max_depth': (3,7),
        'subsample': (0.8, 1.0),  # Change for big datasets
        'colsample_bytree': (0.8, 1.0),  # Change for datasets with lots of features
        'reg_alpha': (0, 10),  
        'reg_lambda': (0, 10), 
        'gamma': (0, 10),
        'min_child_weight':(0,10)}
    def xgboost_hyper_param(learning_rate,
                            n_estimators,
                            max_depth,
                            subsample,
                            colsample_bytree,
                            reg_alpha,
                            reg_lambda,
                            gamma,
                            min_child_weight
                            ):

        max_depth = int(max_depth)
        n_estimators = int(n_estimators)

        clf = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            gamma=gamma,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            scale_pos_weight=nB/nS,
            tree_method='gpu_hist'
            )

        clf.fit(
            traindataset[input_features].values,
            traindataset.target.values,
            sample_weight=(traindataset["total_weight"].values)
            )       
        probaT = clf.predict_proba(valdataset[variables].values.astype(np.float64))
        fprt, tprt, thresholds = roc_curve(valdataset["target"].astype(np.float64), probaT[:,1],sample_weight = valdataset['total_weight'])
        test_auct = auc(np.sort(fprt), np.sort(tprt))
        proba = clf.predict_proba(traindataset[variables].values)
        fpr, tpr, thresholds = roc_curve(traindataset["target"].values,proba[:,1],sample_weight = traindataset['total_weight'].values)
        train_auc = auc(np.sort(fpr), np.sort(tpr))
        modified_test_auct = abs(train_auc - test_auct) + 1/test_auct
        print("ZZ: abs(train_auc - test_auct) + 1/test_auct = ",abs(train_auc - test_auct) + 1/test_auct)
        print("ZZ: test_auct = ",test_auct)
        print("ZZ : reg_alpha = ",reg_alpha)
        return modified_test_auct



        #FIemmi
        #Define a version of cross_val_score that takes into account sample weights when computing the score
        # def cross_val_score_weighted(model, X, y, weights, cv=3, metric=sklearn.metrics.roc_auc_score): #the metric is set to be the roc_auc_score, but it can be changed if needed
        #     kf = KFold(n_splits=cv)
        #     kf.get_n_splits(X)
        #     scores = []
        #     for train_index, test_index in kf.split(X): #perform k-fold splitting by hand
        #         model_clone = sklearn.base.clone(model)
        #         X_train, X_test = X[train_index], X[test_index]
        #         y_train, y_test = y[train_index], y[test_index]
        #         weights_train, weights_test = weights[train_index], weights[test_index]
        #         model_clone.fit(X_train,y_train,sample_weight=weights_train) #fit a the cloned model using the sample weights
        #         probaT = model_clone.predict_proba(X_test)
        #         test_auct = metric(y_test, probaT[:,1],sample_weight = weights_test) #compute the scorer in the same fashion of L165, i.e., using the sample weights correctly
        #         proba = model_clone.predict_proba(X_train)
        #         train_auc = metric(y_train, proba[:,1],sample_weight = weights_train)
        #         scores.append(abs(train_auc - test_auct) + 1/test_auct)
        #     return scores
        
        # #FIemmi
        # #Instead of using the ordinary cross_val_score method, use the custom cross_val_score_weighted
        # #cv_vals = cross_val_score(clf, traindataset[input_features].values,traindataset.target.values, fit_params={'sample_weight':traindataset["total_weight"].values }, cv=3, scoring='roc_auc')
        # cv_vals = cross_val_score_weighted(clf, traindataset[input_features].values,traindataset.target.values, traindataset["total_weight"].values, cv=3)
        # print("cv_vals:", cv_vals)
        # print("roc_auc: {0} +/- {1}".format(np.array(cv_vals).mean(), np.array(cv_vals).std()))
        # print("ZZ: cv_vals mean scores:", -np.mean(cv_vals))
        # return - np.mean(cv_vals)
    """
        probaT = clf.predict_proba(valdataset[variables].values.astype(np.float64))
        score = log_loss(valdataset["target"].values, probaT[:,1],sample_weight = valdataset['total_weight'])
        fprt, tprt, thresholds = roc_curve(valdataset["target"].astype(np.float64), probaT[:,1],sample_weight = valdataset['total_weight'])
        test_auct = auc(np.sort(fprt), np.sort(tprt))
        print("test_auc",test_auct)
        proba = clf.predict_proba(traindataset[variables].values)
        fpr, tpr, thresholds = roc_curve(traindataset["target"].values,proba[:,1],sample_weight = traindataset['total_weight'].values)
        train_auc = auc(np.sort(fpr), np.sort(tpr))
        print("test auc - train auc:", -abs(test_auct - train_auc))
        print("reg_alpha:",reg_alpha)
        # more diagnosis, in case
        # cross_val = cross_validate(clf,traindataset[input_features].values,traindataset.target.values,cv=3,return_train_score=True,fit_params={'sample_weight':traindataset["total_weight"].values } )
        # print('Training sample score: ',cross_val['train_score'])
        # print("test score",cross_val['test_score'].mean())
        # return cross_val['test_score'].mean()
        # print("log func:", score)
        # if (type(score) == type(None)):
        #     score = 10000
        return -abs(test_auct - train_auc)
        # return test_auct
        # return np.mean(cross_val_score(clf, traindataset[input_features].values,traindataset.target.values, cv=3, scoring='loss_function'))
        # return test_auct
        # print("roc_auc:",np.mean(cross_val_score(clf, traindataset[input_features].values,traindataset.target.values,fit_params={'sample_weight':traindataset["total_weight"].values }, cv=3, scoring='roc_auc')))
        # print("roc_auc wo sample weight:",np.mean(cross_val_score(clf, traindataset[input_features].values,traindataset.target.values, cv=3, scoring='roc_auc')))
        # return np.mean(cross_val_score(clf, traindataset[input_features].values,traindataset.target.values,fit_params={'sample_weight':traindataset["total_weight"].values }, cv=3, scoring='roc_auc'))
    """
    optimizer = BayesianOptimization(
        f=xgboost_hyper_param,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(init_points=3,n_iter=10)

    #Extracting the best parameters
    params = optimizer.max['params']
    print(params)
def train_bayes_model_combined_16(events_sig_16postVFP, events_sig_16preVFP ,  events_bkg_16postVFP, events_bkg_16preVFP ):
    input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv"]
     # ------------------------------- UL16 postVFP ------------------------------- #
    df_sig_16postVFP = events_sig_16postVFP.arrays(input_features,library='pd')
    df_pfff_16postVFP = events_bkg_16postVFP['DataDriven_QCD'].arrays(input_features,library='pd')
    df_pp_16postVFP = events_bkg_16postVFP['pp'].arrays(input_features,library='pd')
    df_sig_weight_16postVFP = events_sig_16postVFP.arrays(['weight'],library='pd')
    df_bkg_pfff_weight_16postVFP = events_bkg_16postVFP['DataDriven_QCD'].arrays(['weight',"Norm_SFs"],library='pd')
    df_bkg_pp_weight_16postVFP = events_bkg_16postVFP['pp'].arrays(['weight',"Norm_SFs"],library='pd')

    sig_total_weight_16postVFP = df_sig_weight_16postVFP['weight']*(df_sig_16postVFP['vtxprob']*1./df_sig_16postVFP['sigmarv']+(1-df_sig_16postVFP['vtxprob'])*1./df_sig_16postVFP['sigmawv'])
    sig_total_weight_16postVFP = sig_total_weight_16postVFP.clip(lower=0)
    bkg_pfff_total_weight_16postVFP = df_bkg_pfff_weight_16postVFP['weight'] * df_bkg_pfff_weight_16postVFP['Norm_SFs']
    bkg_pfff_total_weight_16postVFP = bkg_pfff_total_weight_16postVFP.clip(lower=0)
    bkg_pp_total_weight_16postVFP = df_bkg_pp_weight_16postVFP['weight'] * df_bkg_pp_weight_16postVFP['Norm_SFs']
    bkg_pp_total_weight_16postVFP = bkg_pp_total_weight_16postVFP.clip(lower=0)

    reweight_sig_bkg = (sum(bkg_pfff_total_weight_16postVFP) + sum(bkg_pp_total_weight_16postVFP)) / sum(sig_total_weight_16postVFP)
    sig_total_weight_16postVFP = reweight_sig_bkg*sig_total_weight_16postVFP
    sig_total_weight_16postVFP = sig_total_weight_16postVFP # pay more attention on the signal
    print("ZZ: check signal 16postVFP weight:",sum(sig_total_weight_16postVFP))
    print("ZZ: check bkgPP 16postVFP weight:",sum(bkg_pp_total_weight_16postVFP))
    print("ZZ: check bkgDD 16postVFP weight:",sum(bkg_pfff_total_weight_16postVFP))
    # -------------------------------- UL16 preVFP ------------------------------- #
    df_sig_16preVFP = events_sig_16preVFP.arrays(input_features,library='pd')
    df_pfff_16preVFP = events_bkg_16preVFP['DataDriven_QCD'].arrays(input_features,library='pd')
    df_pp_16preVFP = events_bkg_16preVFP['pp'].arrays(input_features,library='pd')
    df_sig_weight_16preVFP = events_sig_16preVFP.arrays(['weight'],library='pd')
    df_bkg_pfff_weight_16preVFP = events_bkg_16preVFP['DataDriven_QCD'].arrays(['weight',"Norm_SFs"],library='pd')
    df_bkg_pp_weight_16preVFP = events_bkg_16preVFP['pp'].arrays(['weight',"Norm_SFs"],library='pd')

    sig_total_weight_16preVFP = df_sig_weight_16preVFP['weight']*(df_sig_16preVFP['vtxprob']*1./df_sig_16preVFP['sigmarv']+(1-df_sig_16preVFP['vtxprob'])*1./df_sig_16preVFP['sigmawv'])
    sig_total_weight_16preVFP = sig_total_weight_16preVFP.clip(lower=0)
    bkg_pfff_total_weight_16preVFP = df_bkg_pfff_weight_16preVFP['weight'] * df_bkg_pfff_weight_16preVFP['Norm_SFs']
    bkg_pfff_total_weight_16preVFP = bkg_pfff_total_weight_16preVFP.clip(lower=0)
    bkg_pp_total_weight_16preVFP = df_bkg_pp_weight_16preVFP['weight'] * df_bkg_pp_weight_16preVFP['Norm_SFs']
    bkg_pp_total_weight_16preVFP = bkg_pp_total_weight_16preVFP.clip(lower=0)

    reweight_sig_bkg = (sum(bkg_pfff_total_weight_16preVFP) + sum(bkg_pp_total_weight_16preVFP)) / sum(sig_total_weight_16preVFP)
    sig_total_weight_16preVFP = reweight_sig_bkg*sig_total_weight_16preVFP
    sig_total_weight_16preVFP = sig_total_weight_16preVFP # pay more attention on the signal
    print("ZZ: check signal 16preVFP weight:",sum(sig_total_weight_16preVFP))
    print("ZZ: check bkgPP 16preVFP weight:",sum(bkg_pp_total_weight_16preVFP))
    print("ZZ: check bkgDD 16preVFP weight:",sum(bkg_pfff_total_weight_16preVFP))
  

    #define the dataframe
    sig_dataframe = pd.DataFrame()
    bkg_pp_dataframe = pd.DataFrame()
    bkg_pfff_dataframe = pd.DataFrame()

    sig_dataframe = pd.concat([df_sig_16postVFP, df_sig_16preVFP])
    sig_dataframe['target'] = 1
    sig_dataframe['key'] = 'sig'
    sig_dataframe['total_weight'] = pd.concat([sig_total_weight_16postVFP, sig_total_weight_16preVFP])

    bkg_pp_dataframe = pd.concat([df_pp_16postVFP, df_pp_16preVFP])
    bkg_pp_dataframe['target'] = 0
    bkg_pp_dataframe['key'] = 'pp'
    bkg_pp_dataframe['total_weight'] = pd.concat([bkg_pp_total_weight_16postVFP,bkg_pp_total_weight_16preVFP])

    bkg_pfff_dataframe = pd.concat([df_pfff_16postVFP,df_pfff_16preVFP])
    bkg_pfff_dataframe['target'] = 0
    bkg_pfff_dataframe['key'] = "pfff"
    bkg_pfff_dataframe['total_weight'] = pd.concat([bkg_pfff_total_weight_16postVFP, bkg_pfff_total_weight_16preVFP])

    Hgg_dataframe = pd.concat([sig_dataframe,bkg_pp_dataframe,bkg_pfff_dataframe])
    # Hgg_dataframe = pd.concat([sig_dataframe,bkg_pp_dataframe])


    Hgg_dataframe['total_weight'] = Hgg_dataframe['total_weight'].clip(lower= 0)


    # os.chdir("/eos/user/z/zhenxuan/BDT/XGboost/python")
    # from load_data import *
    rng = np.random.RandomState(31337)

    ##############
    ## options ##
    ##############
    FeatureSelection = False
    RFESelection = False
    n_features = 10
    GridSearch = False

    variables = input_features
    print("variables:" ,variables)
    data = Hgg_dataframe
    traindataset, valdataset  = train_test_split(data, test_size=0.3, random_state=rng)
    nS = len(traindataset.iloc[(traindataset.target.values == 1)])
    nB = len(traindataset.iloc[(traindataset.target.values == 0)])
    print("nB/nS:",nB/nS)

    rng = np.random.RandomState(31337)

    input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv"]
    pbounds = {
        'learning_rate': (0.001, 0.1),
        'n_estimators': (500, 10000),
        'max_depth': (3,7),
        'subsample': (0.8, 1.0),  # Change for big datasets
        'colsample_bytree': (0.8, 1.0),  # Change for datasets with lots of features
        'reg_alpha': (0, 10),  
        'reg_lambda': (0, 10), 
        'gamma': (0, 10),
        'min_child_weight':(0,10)}
    def xgboost_hyper_param(learning_rate,
                            n_estimators,
                            max_depth,
                            subsample,
                            colsample_bytree,
                            reg_alpha,
                            reg_lambda,
                            gamma,
                            min_child_weight
                            ):

        max_depth = int(max_depth)
        n_estimators = int(n_estimators)

        clf = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            gamma=gamma,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            scale_pos_weight=nB/nS,
            tree_method='gpu_hist'
            )

        clf.fit(
            traindataset[input_features].values,
            traindataset.target.values,
            sample_weight=(traindataset["total_weight"].values)
            )       
        probaT = clf.predict_proba(valdataset[variables].values.astype(np.float64))
        fprt, tprt, thresholds = roc_curve(valdataset["target"].astype(np.float64), probaT[:,1],sample_weight = valdataset['total_weight'])
        test_auct = auc(np.sort(fprt), np.sort(tprt))
        proba = clf.predict_proba(traindataset[variables].values)
        fpr, tpr, thresholds = roc_curve(traindataset["target"].values,proba[:,1],sample_weight = traindataset['total_weight'].values)
        train_auc = auc(np.sort(fpr), np.sort(tpr))
        modified_test_auct = abs(train_auc - test_auct) + 1/test_auct
        print("ZZ: abs(train_auc - test_auct) + 1/test_auct = ",abs(train_auc - test_auct) + 1/test_auct)
        print("ZZ: test_auct = ",test_auct)
        print("ZZ : reg_alpha = ",reg_alpha)
        return test_auct
    optimizer = BayesianOptimization(
    f=xgboost_hyper_param,
    pbounds=pbounds,
    random_state=1,
    )
    optimizer.maximize(init_points=3,n_iter=200)

    #Extracting the best parameters
    params = optimizer.max['params']
    print(params)
samples = "UL18 Summer20 10 inputfeatures"
print("sample is:", samples)
# input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmawv"]
input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv"]
events_sig_UL18 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/Sig125_negReweighting_v2.root:Sig125")
events_bkg_UL18 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")

train_bayes_model(events_sig= events_sig_UL18, events_bkg= events_bkg_UL18, input_features=input_features)

# samples = "UL17 wo sigamrv"
# print("sample is:", samples)
# events_sig_17 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/Summer20/Sig125_v2_negReweighting.root:Sig125")
# input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv"]
# events_bkg_17 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/Summer20/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmawv"]
# train_bayes_model(events_sig= events_sig_17, events_bkg= events_bkg_17, input_features=input_features)
# events_sig_combined = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/combined/combined_sig.root:Sig125")
# events_bkg_combined = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/combined/combined_bkg.root")

# train_bayes_model(events_sig= events_sig_combined, events_bkg= events_bkg_combined)
# samples = "UL16_preVFP"
# print("sample is:", samples)
# events_sig_UL16PreVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/output_sig125.root:Sig125")
# events_bkg_UL16PreVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# train_bayes_model(events_sig= events_sig_UL16PreVFP, events_bkg= events_bkg_UL16PreVFP)
# samples = "UL16_postVFP new"
# print("sample is:", samples)
# events_sig_UL16PostVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/output_sig125_new.root:Sig125")
# events_bkg_UL16PostVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# train_bayes_model(events_sig= events_sig_UL16PostVFP, events_bkg= events_bkg_UL16PostVFP)

# --------------------------------- combined --------------------------------- #
# events_sig_16preVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/output_sig125_new.root:Sig125")
# events_bkg_16preVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# events_sig_16postVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/output_sig125_new.root:Sig125")
# events_bkg_16postVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# events_sig_17 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/output_sig125_ForBDT.root:Sig125")
# events_bkg_17 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# print("bayes UL17")
# train_bayes_model(events_sig= events_sig_17, events_bkg= events_bkg_17)
# events_sig_18 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS/output_sig125_IncludeLumi.root:Sig125")
# events_bkg_18 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# print("combined ")
# train_bayes_model_combined_16(events_sig_16postVFP = events_sig_16postVFP, events_sig_16preVFP  = events_sig_16preVFP ,  events_bkg_16postVFP = events_bkg_16postVFP, events_bkg_16preVFP  = events_bkg_16preVFP)
# train_bayes_model_combined(events_sig_16postVFP = events_sig_16postVFP, events_sig_16preVFP  = events_sig_16preVFP , events_sig_17 = events_sig_17, events_sig_18 = events_sig_18,  events_bkg_16postVFP = events_bkg_16postVFP, events_bkg_16preVFP  = events_bkg_16preVFP , events_bkg_17 = events_bkg_17, events_bkg_18  = events_bkg_18)

