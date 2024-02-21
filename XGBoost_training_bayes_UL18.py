
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
def train_bayes_model(events_sig, events_bkg , input_features, is_mass_window, weight_style):
    # if input_features list don't contain sigmarv, add it
    if 'sigmarv' not in input_features:
        df_sig = events_sig.arrays(input_features+['sigmarv'],library='pd')
    else:
        df_sig = events_sig.arrays(input_features,library='pd')
        
        
        
        
    # mass window    
    events_sig = events_sig.arrays()
    events_bkg_pp = events_bkg['pp'].arrays()
    events_bkg_dd = events_bkg['DataDriven_QCD'].arrays()
    if is_mass_window == True:
        mask_sig = np.logical_and(events_sig['CMS_hgg_mass']>115,events_sig['CMS_hgg_mass']<135)
        mask_dd = np.logical_and(events_bkg_dd['CMS_hgg_mass']>115,events_bkg_dd['CMS_hgg_mass']<135)
        mask_pp = np.logical_and(events_bkg_pp['CMS_hgg_mass']>115,events_bkg_pp['CMS_hgg_mass']<135)
    else:
        mask_sig = np.logical_and(events_sig['CMS_hgg_mass']>-15,events_sig['CMS_hgg_mass']<13335)
        mask_dd = np.logical_and(events_bkg_dd['CMS_hgg_mass']>-15,events_bkg_dd['CMS_hgg_mass']<13335)
        mask_pp = np.logical_and(events_bkg_pp['CMS_hgg_mass']>-15,events_bkg_pp['CMS_hgg_mass']<13335)
    df_sig = pd.DataFrame.from_records(events_sig[input_features+["CMS_hgg_mass", "weight", "weight_absRatio"]].tolist())
    df_pfff = pd.DataFrame.from_records(events_bkg_dd[input_features+["CMS_hgg_mass"]].tolist())
    df_pp = pd.DataFrame.from_records(events_bkg_pp[input_features+["CMS_hgg_mass"]].tolist())
    df_sig_weight = pd.DataFrame.from_records(events_sig[['weight','sigmarvDecorr']].tolist())
    df_bkg_pfff_weight = pd.DataFrame.from_records(events_bkg_dd[['weight','sigmarvDecorr', 'Norm_SFs']].tolist())
    df_bkg_pp_weight = pd.DataFrame.from_records(events_bkg_pp[['weight','sigmarvDecorr', 'Norm_SFs']].tolist())
    # mask_ms_weight = (df_sig['vtxprob']*1./df_sig['sigmarv']+(1-df_sig['vtxprob'])*1./df_sig['sigmawv']) > 200
    # ms_weight[mask_ms_weight] = 200
    if weight_style == 'msweight':
        ms_weight = (events_sig['vtxprob']*1./events_sig['sigmarv'])+(1-events_sig['vtxprob'])*1./events_sig['sigmawv']
        sig_total_weight = df_sig_weight['weight'] * ms_weight
    elif weight_style == "no_msweight":
        sig_total_weight = df_sig_weight['weight'] # attention: cancel the msweight
    elif weight_style == "sigmarvDecorr":
        sigmarvDecorr_weight = 1. / df_sig_weight['sigmarvDecorr']
        sig_total_weight = df_sig_weight['weight'] * sigmarvDecorr_weight
    elif weight_style == "weight_absRatio":
        ms_weight = (events_sig['vtxprob']*1./events_sig['sigmarv'])+(1-events_sig['vtxprob'])*1./events_sig['sigmawv']
        sig_total_weight = events_sig['weight'] * events_sig['weight_absRatio'] * ms_weight
    # set the sig_total_weight < 0 to 0
    sig_total_weight = np.where(sig_total_weight<0, 0, sig_total_weight)

    
    
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
        'learning_rate': (0.0001, 0.01),
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
        modified_test_auct = 1/(abs(train_auc - test_auct) + 1/ test_auct) + test_auct
        print("ZZ: abs(train_auc - test_auct) + 1/test_auct = ", 1/ (abs(train_auc - test_auct)) + test_auct)
        print("ZZ: train_auc = ",train_auc)
        print("ZZ: test_auct = ",test_auct)
        print("ZZ : reg_alpha = ",reg_alpha)
        return modified_test_auct

    optimizer = BayesianOptimization(
        f=xgboost_hyper_param,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(init_points=5,n_iter=300)

    #Extracting the best parameters
    params = optimizer.max['params']
    print(params)


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



samples = "UL18 masswindow no sigmrv"
input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmawv"]
print("sample is:", samples)
# events_sig_UL16PostVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/output_sig125_new.root:Sig125")
events_sig_18 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/summer20/Sig125_negReweighting_v2.root:Sig125")
events_bkg_18 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
train_bayes_model(events_sig= events_sig_18, events_bkg= events_bkg_18, is_mass_window=True, weight_style='weight_absRatio', input_features = input_features)

