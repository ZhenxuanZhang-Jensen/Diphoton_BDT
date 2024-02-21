
import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFECV, RFE
from functools import partial
import json
import os 
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import ZZTime as zz
zzTime = zz.outputTime()
def train_model(events_sig, events_bkg,colsample_bytree_v ,gamma_v, learning_rate_v, max_depth_v, min_child_weight_v, n_estimators_v, reg_alpha_v, reg_lambda_v, subsample_v,samples ):

    samples = samples
    input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv"]
    df_sig = events_sig.arrays(input_features,library='pd')
    df_pfff = events_bkg['DataDriven_QCD'].arrays(input_features,library='pd')
    df_pp = events_bkg['pp'].arrays(input_features,library='pd')
    df_sig_weight = events_sig.arrays(['weight'],library='pd')
    df_bkg_pfff_weight = events_bkg['DataDriven_QCD'].arrays(['weight',"Norm_SFs"],library='pd')
    df_bkg_pp_weight = events_bkg['pp'].arrays(['weight',"Norm_SFs"],library='pd')

    # sig_total_weight = df_sig_weight['weight']*(df_sig['vtxprob']*1./df_sig['sigmarv']+(1-df_sig['vtxprob'])*1./df_sig['sigmawv'])
    sig_total_weight = df_sig_weight['weight']
    sig_total_weight = sig_total_weight.clip(lower=0)
    bkg_pfff_total_weight = df_bkg_pfff_weight['weight'] * df_bkg_pfff_weight['Norm_SFs']
    bkg_pfff_total_weight = bkg_pfff_total_weight.clip(lower=0)
    # bkg_pfff_total_weight = np.zeros_like(bkg_pfff_total_weight) #attention: just use to check pp
    # small tune to decrease the dd weigth so that the dd weigth would not have the overfitting problem and have the better data/mc agreement plot

    bkg_pp_total_weight = df_bkg_pp_weight['weight'] * df_bkg_pp_weight['Norm_SFs']
    bkg_pp_total_weight = bkg_pp_total_weight.clip(lower=0)
    # reweightPPToDD = sum(bkg_pfff_total_weight) / sum(bkg_pp_total_weight)
    # bkg_pp_total_weight = bkg_pp_total_weight * 0.2
    reweight_sig_bkg = (sum(bkg_pfff_total_weight) + sum(bkg_pp_total_weight)) / sum(sig_total_weight)
    sig_total_weight = reweight_sig_bkg*sig_total_weight
    sig_total_weight = sig_total_weight # pay more attention on the signal
    print("ZZ: check signal weight:",sum(sig_total_weight))
    print("ZZ: check bkgPP weight:",sum(bkg_pp_total_weight))
    print("ZZ: check bkgDD weight:",sum(bkg_pfff_total_weight))

    #define the dataframe
    sig_dataframe = pd.DataFrame()
    bkg_pp_dataframe = pd.DataFrame()
    bkg_pfff_dataframe = pd.DataFrame()

    sig_dataframe = df_sig
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
    rng = np.random.RandomState(312)

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
    model = xgb.XGBClassifier(
                            colsample_bytree = colsample_bytree_v, gamma = gamma_v , learning_rate = learning_rate_v , max_depth = max_depth_v , min_child_weight = min_child_weight_v, n_estimators = n_estimators_v , reg_alpha = reg_alpha_v , reg_lambda = reg_lambda_v , subsample = subsample_v ,
                            objective='binary:logistic', 
                            scale_pos_weight=nB/nS,
                            tree_method='gpu_hist')
    print(len(traindataset['total_weight'][traindataset['total_weight']<0]))
    print('model \n', model)
    # fit model
    model.fit(
        traindataset[variables].values,
        traindataset.target.values,
        sample_weight=(traindataset["total_weight"].values),
        # more diagnosis, in case
        eval_set=[(traindataset[variables].values,  traindataset.target.values),
            (valdataset[variables].values ,  valdataset.target.values)],
        sample_weight_eval_set = [(traindataset["total_weight"].values),(valdataset["total_weight"].values)],
            verbose=True,eval_metric="auc"
        )
    # save model
    pickle.dump(model, open("/hpcfs/cms/cmsgpu/zhangzhx/BDT/" + zzTime+ samples + "_DiphotonXGboost_afterTune_new_withAllSigs.pkl", "wb"))
    # Plot ROC curve
    sig_dataframe = df_sig
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

    print("-----plot ROC--------- \n")
    proba = model.predict_proba(traindataset[variables].values)
    fpr, tpr, thresholds = roc_curve(traindataset["target"].values,proba[:,1],sample_weight = traindataset['total_weight'].values)
    train_auc = auc(np.sort(fpr), np.sort(tpr))
    probaT = model.predict_proba(valdataset[variables].values )
    fprt, tprt, thresholds = roc_curve(valdataset["target"].values, probaT[:,1],sample_weight = valdataset['total_weight'].values)
    test_auct = auc(np.sort(fprt), np.sort(tprt))
    fig, ax = plt.subplots(figsize=(6, 6))
    ## ROC curve
    ax.plot(fpr, tpr, lw=1, label='XGB train (area = %0.3f)'%(train_auc))
    ax.plot(fprt, tprt, lw=1, label='XGB test (area = %0.3f)'%(test_auct))
    ax.set_ylim([0.0,1.0])
    ax.set_xlim([0.0,1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()
    fig.savefig("roc_" + samples + str(zzTime) + ".png")
    plt.cla()
    proba_sig = model.predict_proba(traindataset[variables][traindataset['target']==1].values)
    probaT_sig = model.predict_proba(valdataset[variables][valdataset['target']==1].values)
    proba_bkg = model.predict_proba(traindataset[variables][traindataset['target']==0].values)
    probaT_bkg = model.predict_proba(valdataset[variables][valdataset['target']==0].values)
    print("plot weighted binary score plots")
    plt.figure()
    sig_test_bin_counts, sig_test_bin_edges = np.histogram(probaT_sig[:,1],weights = valdataset['total_weight'][valdataset['target']==1].values,density=True,bins=20);
    bkg_test_bin_counts, bkg_test_bin_edges = np.histogram(probaT_bkg[:,1],weights = valdataset['total_weight'][valdataset['target']==0].values,density=True,bins=20);
    sig_test_bin_centres = (sig_test_bin_edges[:-1] + sig_test_bin_edges[1:]) / 2
    bkg_test_bin_centres = (bkg_test_bin_edges[:-1] + bkg_test_bin_edges[1:]) / 2
    sig_test_y_error = np.sqrt(sig_test_bin_counts)
    bkg_test_y_error = np.sqrt(bkg_test_bin_counts)
    plt.errorbar(x=sig_test_bin_centres, y=sig_test_bin_counts,yerr=0, fmt='o', capsize=2,label='Sig(test sample)')
    plt.errorbar(x=bkg_test_bin_centres, y=bkg_test_bin_counts,yerr=0, fmt='o', capsize=2,label='Bkgs(test sample)')
    bin_counts, bin_edges, patches = plt.hist(proba_sig[:,1],weights = traindataset['total_weight'][traindataset['target']==1].values,density=True,alpha=0.5,label='Sig(training sampel)',bins=20);
    bin_counts, bin_edges, patches = plt.hist(proba_bkg[:,1],weights = traindataset['total_weight'][traindataset['target']==0].values,density=True,alpha=0.5,label='Bkgs(training sample)',bins=20);
    plt.legend()
    plt.xlabel("BDT score")
    plt.ylabel(r"(1/N) dN/dx")
    plt.savefig(samples + "binary_score" + str(zzTime) + ".png")
    bin_counts, bin_edges, patches = plt.hist(proba_bkg[:,1],weights = traindataset['total_weight'][traindataset['target']==0].values,density=True,alpha=0.5,label='Bkgs(training sample)',bins=100,range=(0.8,1));
    plt.savefig(samples + "binary_pp_" + str(zzTime) + ".png")
    return None


def train_combined_model(events_sig_16postVFP, events_sig_16preVFP , events_sig_17, events_sig_18,  events_bkg_16postVFP, events_bkg_16preVFP , events_bkg_17, events_bkg_18 ,colsample_bytree_v ,gamma_v, learning_rate_v, max_depth_v, min_child_weight_v, n_estimators_v, reg_alpha_v, reg_lambda_v, subsample_v,samples ):

    samples = samples
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
    rng = np.random.RandomState(312)

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
    model = xgb.XGBClassifier(
                            colsample_bytree = colsample_bytree_v, gamma = gamma_v , learning_rate = learning_rate_v , max_depth = max_depth_v , min_child_weight = min_child_weight_v, n_estimators = n_estimators_v , reg_alpha = reg_alpha_v , reg_lambda = reg_lambda_v , subsample = subsample_v ,
                            objective='binary:logistic', 
                            scale_pos_weight=nB/nS,
                            tree_method='gpu_hist')
    print(len(traindataset['total_weight'][traindataset['total_weight']<0]))
    print('model \n', model)
    # fit model
    model.fit(
        traindataset[variables].values,
        traindataset.target.values,
        sample_weight=(traindataset["total_weight"].values),
        # more diagnosis, in case
        eval_set=[(traindataset[variables].values,  traindataset.target.values),
            (valdataset[variables].values ,  valdataset.target.values)],
        sample_weight_eval_set = [(traindataset["total_weight"].values),(valdataset["total_weight"].values)],
            verbose=True,eval_metric="auc"
        )
    # save model
    pickle.dump(model, open("/hpcfs/cms/cmsgpu/zhangzhx/BDT/" + zzTime+ samples + "_DiphotonXGboost_afterTune_new_withAllSigs.pkl", "wb"))
    # Plot ROC curve
    print("-----plot ROC--------- \n")
    proba = model.predict_proba(traindataset[variables].values)
    fpr, tpr, thresholds = roc_curve(traindataset["target"].values,proba[:,1],sample_weight = traindataset['total_weight'].values)
    train_auc = auc(np.sort(fpr), np.sort(tpr))
    probaT = model.predict_proba(valdataset[variables].values )
    fprt, tprt, thresholds = roc_curve(valdataset["target"].values, probaT[:,1],sample_weight = valdataset['total_weight'].values)
    test_auct = auc(np.sort(fprt), np.sort(tprt))
    fig, ax = plt.subplots(figsize=(6, 6))
    ## ROC curve
    ax.plot(fpr, tpr, lw=1, label='XGB train (area = %0.3f)'%(train_auc))
    ax.plot(fprt, tprt, lw=1, label='XGB test (area = %0.3f)'%(test_auct))
    ax.set_ylim([0.0,1.0])
    ax.set_xlim([0.0,1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()
    fig.savefig("roc_" + samples + str(zzTime) + ".png")
    plt.cla()
    proba_sig = model.predict_proba(traindataset[variables][traindataset['target']==1].values)
    probaT_sig = model.predict_proba(valdataset[variables][valdataset['target']==1].values)
    proba_bkg = model.predict_proba(traindataset[variables][traindataset['target']==0].values)
    probaT_bkg = model.predict_proba(valdataset[variables][valdataset['target']==0].values)
    print("plot weighted binary score plots")
    plt.figure()
    sig_test_bin_counts, sig_test_bin_edges = np.histogram(probaT_sig[:,1],weights = valdataset['total_weight'][valdataset['target']==1].values,density=True,bins=20);
    bkg_test_bin_counts, bkg_test_bin_edges = np.histogram(probaT_bkg[:,1],weights = valdataset['total_weight'][valdataset['target']==0].values,density=True,bins=20);
    sig_test_bin_centres = (sig_test_bin_edges[:-1] + sig_test_bin_edges[1:]) / 2
    bkg_test_bin_centres = (bkg_test_bin_edges[:-1] + bkg_test_bin_edges[1:]) / 2
    sig_test_y_error = np.sqrt(sig_test_bin_counts)
    bkg_test_y_error = np.sqrt(bkg_test_bin_counts)
    plt.errorbar(x=sig_test_bin_centres, y=sig_test_bin_counts,yerr=0, fmt='o', capsize=2,label='Sig(test sample)')
    plt.errorbar(x=bkg_test_bin_centres, y=bkg_test_bin_counts,yerr=0, fmt='o', capsize=2,label='Bkgs(test sample)')
    bin_counts, bin_edges, patches = plt.hist(proba_sig[:,1],weights = traindataset['total_weight'][traindataset['target']==1].values,density=True,alpha=0.5,label='Sig(training sampel)',bins=20);
    bin_counts, bin_edges, patches = plt.hist(proba_bkg[:,1],weights = traindataset['total_weight'][traindataset['target']==0].values,density=True,alpha=0.5,label='Bkgs(training sample)',bins=20);
    plt.legend()
    plt.xlabel("BDT score")
    plt.ylabel(r"(1/N) dN/dx")
    plt.savefig(samples + "binary_score" + str(zzTime) + ".png")
    bin_counts, bin_edges, patches = plt.hist(proba_bkg[:,1],weights = traindataset['total_weight'][traindataset['target']==0].values,density=True,alpha=0.5,label='Bkgs(training sample)',bins=100,range=(0.8,1));
    plt.savefig(samples + "binary_pp_" + str(zzTime) + ".png")
    return None



events_sig_18 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS/output_sig125_IncludeLumi.root:Sig125")
events_bkg_18 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
train_model(events_sig=events_sig_18, events_bkg=events_bkg_18, colsample_bytree_v= 0.8810535004954177, gamma_v= 0.01010807703975658, learning_rate_v= 0.04019226479637405, max_depth_v=4, min_child_weight_v= 1, n_estimators_v= 4155, reg_alpha_v= 3.235, reg_lambda_v= 4.23, subsample_v= 0.8841130927443083, samples="_UL18_")

# events_sig_UL16_preVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/output_sig125.root:Sig125")
# events_sig_16preVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/output_sig125_new.root:Sig125")
# events_bkg_16preVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# print("sample is:", "UL2016_preVFP")
# train_model(events_sig=events_sig_16preVFP, events_bkg=events_bkg_16preVFP, colsample_bytree_v= 0.8834044009405149, gamma_v= 8.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 4.402676724513391, n_estimators_v= 6000, reg_alpha_v= 5.587806341330127, reg_lambda_v= 4.366821811291432, subsample_v= 0.879353494846134, samples="UL16PreVFP")

# events_sig_UL16_postVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/output_sig125.root:Sig125")
# events_sig_16postVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/output_sig125_new.root:Sig125")
# events_bkg_16postVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# print("sample is:", "UL2016_postVFP")
# train_model(events_sig=events_sig_16postVFP, events_bkg=events_bkg_16postVFP, colsample_bytree_v= 0.8810535004954177, gamma_v= 0.01010807703975658, learning_rate_v= 0.008019226479637405, max_depth_v=4, min_child_weight_v= 0, n_estimators_v= 4155, reg_alpha_v= 0, reg_lambda_v= 0, subsample_v= 0.8841130927443083, samples="UL16PostVFP")
# train_model(events_sig=events_sig_UL16_postVFP_new, events_bkg=events_bkg_UL16_postVFP, colsample_bytree_v= 0.9810535004954177, gamma_v= 10.01010807703975658, learning_rate_v= 0.007019226479637405, max_depth_v= 4, min_child_weight_v= 2.11805334912794, n_estimators_v= 4155, reg_alpha_v= 3.440097655655839, reg_lambda_v= 4.963795272297961, subsample_v= 0.8841130927443083, samples="UL16PostVFP")

# events_sig_UL16_combined = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/combined/combined_16_sig.root:Sig125")
# events_bkg_UL16_combined = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/combined/combined_16_bkg.root")
# print("sample is:", "UL2016_combine")
# train_model(events_sig=events_sig_UL16_combined, events_bkg=events_bkg_UL16_combined, colsample_bytree_v= 0.9810535004954177, gamma_v= 10.01010807703975658, learning_rate_v= 0.007019226479637405, max_depth_v= 4, min_child_weight_v= 4.11805334912794, n_estimators_v= 4155, reg_alpha_v= 8.440097655655839, reg_lambda_v= 9.963795272297961, subsample_v= 0.8841130927443083, samples="_UL16_combined_")
# train_model(events_sig=events_sig_UL16_postVFP, events_bkg=events_bkg_UL16_postVFP, colsample_bytree_v= 0.9810535004954177, gamma_v= 10.01010807703975658, learning_rate_v= 0.007019226479637405, max_depth_v= 4, min_child_weight_v= 19.11805334912794, n_estimators_v= 4155, reg_alpha_v= 8.440097655655839, reg_lambda_v= 9.963795272297961, subsample_v= 0.8841130927443083, samples="UL16PostVFP")
# events_sig_17 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/output_sig125_ForBDT.root:Sig125")
# events_bkg_17 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# train_model(events_sig=events_sig_17, events_bkg=events_bkg_17, colsample_bytree_v= 0.8810535004954177, gamma_v= 0.01010807703975658, learning_rate_v= 0.008019226479637405, max_depth_v=4, min_child_weight_v= 0, n_estimators_v= 4155, reg_alpha_v= 0, reg_lambda_v= 0, subsample_v= 0.8841130927443083, samples="_UL17_")
# ------------------------------ combined model ------------------------------ #
# train_combined_model(events_sig_16postVFP = events_sig_16postVFP, events_sig_16preVFP  = events_sig_16preVFP , events_sig_17 = events_sig_17, events_sig_18 = events_sig_18,  events_bkg_16postVFP = events_bkg_16postVFP, events_bkg_16preVFP  = events_bkg_16preVFP , events_bkg_17 = events_bkg_17, events_bkg_18  = events_bkg_18 ,colsample_bytree_v= 0.9810535004954177, gamma_v= 3.01010807703975658, learning_rate_v= 0.01019226479637405, max_depth_v= 7, min_child_weight_v= 4.11805334912794, n_estimators_v= 4155, reg_alpha_v= 8.440097655655839, reg_lambda_v= 9.963795272297961, subsample_v= 0.8841130927443083, samples="_all_combined_" )
