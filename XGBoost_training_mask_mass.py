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
def train_model(events_sig, events_bkg,colsample_bytree_v ,gamma_v, learning_rate_v, max_depth_v, min_child_weight_v, n_estimators_v, reg_alpha_v, reg_lambda_v, subsample_v,samples, is_mass_window, weight_style, input_features ):

    samples = samples
    
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
    events_sig = events_sig[mask_sig]
    events_bkg_pp = events_bkg_pp[mask_pp]
    events_bkg_dd = events_bkg_dd[mask_dd]
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



    # sig_total_weight = sig_total_weight.clip(lower=0)

    bkg_pfff_total_weight = df_bkg_pfff_weight['weight'] * df_bkg_pfff_weight['Norm_SFs']
    # check bkg_pfff_total_weight < 0 number
    print("ZZ: check bkg_pfff_total_weight < 0 number:",len(bkg_pfff_total_weight[bkg_pfff_total_weight<0]))
    # bkg_pfff_total_weight = bkg_pfff_total_weight.apply(lambda x: x if x>=0. else -x)

    bkg_pfff_total_weight = bkg_pfff_total_weight.clip(lower=0)

    bkg_pp_total_weight = df_bkg_pp_weight['weight'] * df_bkg_pp_weight['Norm_SFs']
    # check bkg_pfff_total_weight < 0 number
    print("ZZ: check bkg_pp_total_weight < 0 number:",len(bkg_pp_total_weight[bkg_pp_total_weight<0]))
    # bkg_pp_total_weight = bkg_pp_total_weight.apply(lambda x: x if x>=0. else -x)

    bkg_pp_total_weight = bkg_pp_total_weight.clip(lower=0)
    
    # sig_total_weight = reweight_sig_bkg*sig_total_weight
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


    # Hgg_dataframe['total_weight'] = Hgg_dataframe['total_weight'].clip(lower= 0)

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
    traindataset, valdataset  = train_test_split(data, test_size=0.5, random_state=rng)
    nS = len(traindataset.iloc[(traindataset.target.values == 1)])
    nB = len(traindataset.iloc[(traindataset.target.values == 0)])
    print("nB/nS:",nB/nS)
    reweight_sig_bkg = (sum(bkg_pfff_total_weight) + sum(bkg_pp_total_weight)) / sum(sig_total_weight)
    print("reweight_sig_bkg:",reweight_sig_bkg)
    model = xgb.XGBClassifier(
                            colsample_bytree = colsample_bytree_v, gamma = gamma_v , learning_rate = learning_rate_v , max_depth = max_depth_v , min_child_weight = min_child_weight_v, n_estimators = n_estimators_v , reg_alpha = reg_alpha_v , reg_lambda = reg_lambda_v , subsample = subsample_v ,
                            objective='binary:logistic',
                            tree_method='gpu_hist',
                            scale_pos_weight=reweight_sig_bkg)
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
    pickle.dump(model, open("/hpcfs/cms/cmsgpu/zhangzhx/BDT/data/forcheck_" + zzTime+ samples + "_DiphotonXGboost_afterTune_new_withAllSigs.pkl", "wb"))
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


    # Hgg_dataframe['total_weight'] = Hgg_dataframe['total_weight'].clip(lower= 0)

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
    fig.savefig("data/roc_" + samples + str(zzTime) + ".png")
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
    plt.savefig("data/" + samples + "binary_score" + str(zzTime) + ".png")
    bin_counts, bin_edges, patches = plt.hist(proba_bkg[:,1],weights = traindataset['total_weight'][traindataset['target']==0].values,density=True,alpha=0.5,label='Bkgs(training sample)',bins=100,range=(0.8,1));
    plt.savefig("data/" + samples + "binary_pp_" + str(zzTime) + ".png")

    # plot the mass distribution in training and testing sample
    plt.figure()
    traindataset['score'] = model.predict_proba(traindataset[variables].values)[:,1]
    valdataset['score'] = model.predict_proba(valdataset[variables].values)[:,1]
    bin_counts, bin_edges, patches = plt.hist(traindataset['CMS_hgg_mass'][traindataset['score']>0.8].values,density=True,alpha=0.5,label='Sig(training sample)',bins=100,range=(100,180));
    bin_counts, bin_edges, patches = plt.hist(valdataset['CMS_hgg_mass'][valdataset['score']>0.8].values,density=True,alpha=0.5,label='Sig(test sample)',bins=100,range=(100,180));
    plt.legend()
    plt.xlabel("mass")
    plt.ylabel(r"(1/N) dN/dx")
    plt.savefig(samples + "mass_sig" + str(zzTime) + ".png")
    # save the train_dataset and val_dataset csv
    # traindataset.to_csv(samples + "train_dataset_" + str(zzTime) + ".csv")
    # valdataset.to_csv(samples + "val_dataset_" + str(zzTime) + ".csv")
    return None


def train_combined_model(events_sig_16postVFP, events_sig_16preVFP , events_sig_17, events_sig_18,  events_bkg_16postVFP, events_bkg_16preVFP , events_bkg_17, events_bkg_18 ,colsample_bytree_v ,gamma_v, learning_rate_v, max_depth_v, min_child_weight_v, n_estimators_v, reg_alpha_v, reg_lambda_v, subsample_v,samples, mass_window ):

    samples = samples
    input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv"]
    all_list_sig = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv","weight","CMS_hgg_mass"]
    all_list_bkg = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv","weight","Norm_SFs","CMS_hgg_mass"]
    # ------------------------------- UL16 postVFP ------------------------------- #
    df_sig_16postVFP = events_sig_16postVFP.arrays(all_list_sig,library='pd')
    df_pfff_16postVFP = events_bkg_16postVFP['DataDriven_QCD'].arrays(all_list_bkg,library='pd')
    df_pp_16postVFP = events_bkg_16postVFP['pp'].arrays(all_list_bkg,library='pd')
    if mass_window == True:
        df_sig_16postVFP = df_sig_16postVFP[(df_sig_16postVFP['CMS_hgg_mass'] > 115) & (df_sig_16postVFP['CMS_hgg_mass'] < 135)]
        df_pfff_16postVFP = df_pfff_16postVFP[(df_pfff_16postVFP['CMS_hgg_mass'] > 115) & (df_pfff_16postVFP['CMS_hgg_mass'] < 135)]
        df_pp_16postVFP = df_pp_16postVFP[(df_pp_16postVFP['CMS_hgg_mass'] > 115) & (df_pp_16postVFP['CMS_hgg_mass'] < 135)]
    

    sig_total_weight_16postVFP = df_sig_16postVFP['weight']*(df_sig_16postVFP['vtxprob']*1./df_sig_16postVFP['sigmarv']+(1-df_sig_16postVFP['vtxprob'])*1./df_sig_16postVFP['sigmawv'])
    sig_total_weight_16postVFP = sig_total_weight_16postVFP.clip(lower=0)
    bkg_pfff_total_weight_16postVFP = df_pfff_16postVFP['weight'] * df_pfff_16postVFP['Norm_SFs']
    bkg_pfff_total_weight_16postVFP = bkg_pfff_total_weight_16postVFP.clip(lower=0)
    bkg_pp_total_weight_16postVFP = df_pp_16postVFP['weight'] * df_pp_16postVFP['Norm_SFs']
    bkg_pp_total_weight_16postVFP = bkg_pp_total_weight_16postVFP.clip(lower=0)

    reweight_sig_bkg = (sum(bkg_pfff_total_weight_16postVFP) + sum(bkg_pp_total_weight_16postVFP)) / sum(sig_total_weight_16postVFP)
    sig_total_weight_16postVFP = reweight_sig_bkg*sig_total_weight_16postVFP
    sig_total_weight_16postVFP = sig_total_weight_16postVFP # pay more attention on the signal
    print("ZZ: check signal 16postVFP weight:",sum(sig_total_weight_16postVFP))
    print("ZZ: check bkgPP 16postVFP weight:",sum(bkg_pp_total_weight_16postVFP))
    print("ZZ: check bkgDD 16postVFP weight:",sum(bkg_pfff_total_weight_16postVFP))
    # -------------------------------- UL16 preVFP ------------------------------- #
    df_sig_16preVFP = events_sig_16preVFP.arrays(all_list_sig,library='pd')
    df_pfff_16preVFP = events_bkg_16preVFP['DataDriven_QCD'].arrays(all_list_bkg,library='pd')
    df_pp_16preVFP = events_bkg_16preVFP['pp'].arrays(all_list_bkg,library='pd')
    if mass_window == True:
        df_sig_16preVFP = df_sig_16preVFP[(df_sig_16preVFP['CMS_hgg_mass'] > 115) & (df_sig_16preVFP['CMS_hgg_mass'] < 135)]
        df_pfff_16preVFP = df_pfff_16preVFP[(df_pfff_16preVFP['CMS_hgg_mass'] > 115) & (df_pfff_16preVFP['CMS_hgg_mass'] < 135)]
        df_pp_16preVFP = df_pp_16preVFP[(df_pp_16preVFP['CMS_hgg_mass'] > 115) & (df_pp_16preVFP['CMS_hgg_mass'] < 135)]


    sig_total_weight_16preVFP = df_sig_16preVFP['weight']*(df_sig_16preVFP['vtxprob']*1./df_sig_16preVFP['sigmarv']+(1-df_sig_16preVFP['vtxprob'])*1./df_sig_16preVFP['sigmawv'])
    sig_total_weight_16preVFP = sig_total_weight_16preVFP.clip(lower=0)
    bkg_pfff_total_weight_16preVFP = df_pfff_16preVFP['weight'] * df_pfff_16preVFP['Norm_SFs']
    bkg_pfff_total_weight_16preVFP = bkg_pfff_total_weight_16preVFP.clip(lower=0)
    bkg_pp_total_weight_16preVFP = df_pp_16preVFP['weight'] * df_pp_16preVFP['Norm_SFs']
    bkg_pp_total_weight_16preVFP = bkg_pp_total_weight_16preVFP.clip(lower=0)

    reweight_sig_bkg = (sum(bkg_pfff_total_weight_16preVFP) + sum(bkg_pp_total_weight_16preVFP)) / sum(sig_total_weight_16preVFP)
    sig_total_weight_16preVFP = reweight_sig_bkg*sig_total_weight_16preVFP
    sig_total_weight_16preVFP = sig_total_weight_16preVFP # pay more attention on the signal
    print("ZZ: check signal 16preVFP weight:",sum(sig_total_weight_16preVFP))
    print("ZZ: check bkgPP 16preVFP weight:",sum(bkg_pp_total_weight_16preVFP))
    print("ZZ: check bkgDD 16preVFP weight:",sum(bkg_pfff_total_weight_16preVFP))
    # -------------------------------- UL17 ------------------------------- #
    df_sig_17 = events_sig_17.arrays(all_list_sig,library='pd')
    df_bkg_pfff_17 = events_bkg_17['DataDriven_QCD'].arrays(all_list_bkg,library='pd')
    df_bkg_pp_17 = events_bkg_17['pp'].arrays(all_list_bkg,library='pd')
    if mass_window == True:
        df_sig_17 = df_sig_17[(df_sig_17['CMS_hgg_mass'] > 115) & (df_sig_17['CMS_hgg_mass'] < 135)]
        df_bkg_pfff_17 = df_bkg_pfff_17[(df_bkg_pfff_17['CMS_hgg_mass'] > 115) & (df_bkg_pfff_17['CMS_hgg_mass'] < 135)]
        df_bkg_pp_17 = df_bkg_pp_17[(df_bkg_pp_17['CMS_hgg_mass'] > 115) & (df_bkg_pp_17['CMS_hgg_mass'] < 135)]

    sig_total_weight_17 = df_sig_17['weight']*(df_sig_17['vtxprob']*1./df_sig_17['sigmarv']+(1-df_sig_17['vtxprob'])*1./df_sig_17['sigmawv'])
    sig_total_weight_17 = sig_total_weight_17.clip(lower=0)
    bkg_pfff_total_weight_17 = df_bkg_pfff_17['weight'] * df_bkg_pfff_17['Norm_SFs']
    bkg_pfff_total_weight_17 = bkg_pfff_total_weight_17.clip(lower=0)
    bkg_pp_total_weight_17 = df_bkg_pp_17['weight'] * df_bkg_pp_17['Norm_SFs']
    bkg_pp_total_weight_17 = bkg_pp_total_weight_17.clip(lower=0)

    reweight_sig_bkg = (sum(bkg_pfff_total_weight_17) + sum(bkg_pp_total_weight_17)) / sum(sig_total_weight_17)
    sig_total_weight_17 = reweight_sig_bkg*sig_total_weight_17
    sig_total_weight_17 = sig_total_weight_17 # pay more attention on the signal
    print("ZZ: check signal 17 weight:",sum(sig_total_weight_17))
    print("ZZ: check bkgPP 17 weight:",sum(bkg_pp_total_weight_17))
    print("ZZ: check bkgDD 17 weight:",sum(bkg_pfff_total_weight_17))
    # -------------------------------- UL18 ------------------------------- #
    df_sig_18 = events_sig_18.arrays(all_list_sig,library='pd')
    df_bkg_pfff_18 = events_bkg_18['DataDriven_QCD'].arrays(all_list_bkg,library='pd')
    df_bkg_pp_18 = events_bkg_18['pp'].arrays(all_list_bkg,library='pd')
    if mass_window == True:
        df_sig_18 = df_sig_18[(df_sig_18['CMS_hgg_mass'] > 115) & (df_sig_18['CMS_hgg_mass'] < 135)]
        df_bkg_pfff_18 = df_bkg_pfff_18[(df_bkg_pfff_18['CMS_hgg_mass'] > 115) & (df_bkg_pfff_18['CMS_hgg_mass'] < 135)]
        df_bkg_pp_18 = df_bkg_pp_18[(df_bkg_pp_18['CMS_hgg_mass'] > 115) & (df_bkg_pp_18['CMS_hgg_mass'] < 135)]

    sig_total_weight_18 = df_sig_18['weight']*(df_sig_18['vtxprob']*1./df_sig_18['sigmarv']+(1-df_sig_18['vtxprob'])*1./df_sig_18['sigmawv'])
    sig_total_weight_18 = sig_total_weight_18.clip(lower=0)
    bkg_pfff_total_weight_18 = df_bkg_pfff_18['weight'] * df_bkg_pfff_18['Norm_SFs']
    bkg_pfff_total_weight_18 = bkg_pfff_total_weight_18.clip(lower=0)
    bkg_pp_total_weight_18 = df_bkg_pp_18['weight'] * df_bkg_pp_18['Norm_SFs']
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

    bkg_pp_dataframe = pd.concat([df_pp_16postVFP, df_pp_16preVFP, df_bkg_pp_17, df_bkg_pp_18])
    bkg_pp_dataframe['target'] = 0
    bkg_pp_dataframe['key'] = 'pp'
    bkg_pp_dataframe['total_weight'] = pd.concat([bkg_pp_total_weight_16postVFP,bkg_pp_total_weight_16preVFP,bkg_pp_total_weight_17,bkg_pp_total_weight_18])

    bkg_pfff_dataframe = pd.concat([df_pfff_16postVFP,df_pfff_16preVFP,df_bkg_pfff_17,df_bkg_pfff_18])
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


def train_combined_model_16(events_sig_16postVFP, events_sig_16preVFP ,  events_bkg_16postVFP, events_bkg_16preVFP  ,colsample_bytree_v ,gamma_v, learning_rate_v, max_depth_v, min_child_weight_v, n_estimators_v, reg_alpha_v, reg_lambda_v, subsample_v,samples ):

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
    pickle.dump(model, open("/hpcfs/cms/cmsgpu/zhangzhx/BDT/data/" + zzTime+ samples + "_DiphotonXGboost_afterTune_new_withAllSigs.pkl", "wb"))
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
    fig.savefig("data/roc_" + samples + str(zzTime) + ".png")
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
    plt.savefig("data/"+samples + "binary_score" + str(zzTime) + ".png")
    bin_counts, bin_edges, patches = plt.hist(proba_bkg[:,1],weights = traindataset['total_weight'][traindataset['target']==0].values,density=True,alpha=0.5,label='Bkgs(training sample)',bins=100,range=(0.8,1));
    plt.savefig("data/" + samples + "binary_pp_" + str(zzTime) + ".png")
    return None

# train_model(events_sig=events_sig_UL18, events_bkg=events_bkg_UL18, colsample_bytree_v =  0.8280773877190468, gamma_v =  1.981014890848788, learning_rate_v =  0.08027371229887814, max_depth_v =  7, min_child_weight_v =  3.1342417815924284, n_estimators_v =  7077, reg_alpha_v =  8.763891522960384, reg_lambda_v =  8.946066635038473, subsample_v =  0.8170088422739556, samples="_combined_")
input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv", "sigmawv"]
print("UL16 postVFP")
events_sig_16postVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/Sig125_negReweighting.root:Sig125")
events_bkg_16postVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# train_model(events_sig=events_sig_16postVFP, events_bkg=events_bkg_16postVFP, colsample_bytree_v= 0.8608603609075975, gamma_v= 0.29975671856728225, learning_rate_v= 0.001238242871497659, max_depth_v= 4, min_child_weight_v= 3.956982120607801, n_estimators_v= 10000, reg_alpha_v= 2.166724550549883, reg_lambda_v= 1.952904326597468, subsample_v= 0.9618189046163824, is_mass_window=True, weight_style = 'weight_absRatio', samples="UL16postVFP_masswindow_withmsweight_withoutsigmarv_with_reweight_negweight", input_features=input_features)
print("UL16 preVFP")
events_sig_16preVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/Sig125_negReweighting.root:Sig125")
events_bkg_16preVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# train_model(events_sig=events_sig_16preVFP, events_bkg=events_bkg_16preVFP, colsample_bytree_v= 0.8948760833688892, gamma_v= 0.9645108804971642, learning_rate_v= 0.007874466942704075, max_depth_v= 5, min_child_weight_v= 4.129984900703856, n_estimators_v= 9437, reg_alpha_v= 4.474098756750759, reg_lambda_v= 4.718973066164986, subsample_v= 0.9082379136306589,is_mass_window=True, weight_style = 'weight_absRatio', samples="UL17_masswindow_withmsweight_withoutsigmarv_with_reweight_negweight", input_features=input_features)
print("UL17")
events_sig_17 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/Summer20/Sig125_v2_negReweighting.root:Sig125")
events_bkg_17 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/Summer20/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# train_model(events_sig=events_sig_17, events_bkg=events_bkg_17, colsample_bytree_v= 0.940792319283802, gamma_v= 0.30865834016958027, learning_rate_v= 0.008369487290202371, max_depth_v= 5, min_child_weight_v= 4.280961940134619, n_estimators_v= 7981, reg_alpha_v= 1.4683323877005279, reg_lambda_v= 3.909623496263803, subsample_v= 0.8767693231352818, is_mass_window=True, weight_style = 'weight_absRatio', samples="UL17_masswindow_withmsweight_withoutsigmarv_with_reweight_negweight", input_features=input_features)
# train_model(events_sig=events_sig_17, events_bkg=events_bkg_17, colsample_bytree_v= 0.5, gamma_v= 0.30865834016958027, learning_rate_v= 0.008369487290202371, max_depth_v= 4, min_child_weight_v= 10, n_estimators_v= 7981, reg_alpha_v= 20, reg_lambda_v= 10.909623496263803, subsample_v= 0.5, is_mass_window=False, weight_style = 'weight_absRatio', samples="UL17_nomasswindow_withmsweight_withoutsigmarv_with_reweight_negweight_tuenedHP", input_features=input_features)
train_model(events_sig=events_sig_17, events_bkg=events_bkg_17, colsample_bytree_v= 0.5, gamma_v= 0.30865834016958027, learning_rate_v= 0.008369487290202371, max_depth_v= 4, min_child_weight_v= 10, n_estimators_v= 7981, reg_alpha_v= 20, reg_lambda_v= 10.909623496263803, subsample_v= 0.5, is_mass_window=False, weight_style = 'weight_absRatio', samples="UL17_nomasswindow_withmsweight_withsigmarv_with_reweight_negweight_tuenedHP", input_features=input_features)

print("UL18")
events_sig_18 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/summer20/Sig125_negReweighting_v2.root:Sig125")
events_bkg_18 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
# train_model(events_sig=events_sig_18, events_bkg=events_bkg_18, colsample_bytree_v = 0.8, gamma_v= 0.0, learning_rate_v= 0.01, max_depth_v= 4, min_child_weight_v= 0.18066827977230054, n_estimators_v= 9146, reg_alpha_v= 3.9823847640268735, reg_lambda_v= 0.7353158132427824, subsample_v= 0.9653995804176685, is_mass_window=True, weight_style = 'weight_absRatio', samples="UL17_nomasswindow_withmsweight_withoutsigmarv_with_reweight_negweight", input_features=input_features)



