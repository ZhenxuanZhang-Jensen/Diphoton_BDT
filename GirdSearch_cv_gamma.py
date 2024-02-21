import uproot
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

events_sig = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/BDT/output_sig125_ForBDT.root:Sig125")
events_bkg = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/BDT/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")

input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv"]
df_sig = events_sig.arrays(input_features,library='pd')
df_pfff = events_bkg['DataDriven_QCD'].arrays(input_features,library='pd')
df_pp = events_bkg['pp'].arrays(input_features,library='pd')
df_sig_weight = events_sig.arrays(['weight'],library='pd')
df_bkg_pfff_weight = events_bkg['DataDriven_QCD'].arrays(['weight',"Norm_SFs"],library='pd')
df_bkg_pp_weight = events_bkg['pp'].arrays(['weight',"Norm_SFs"],library='pd')

sig_total_weight = df_sig_weight['weight']*(df_sig['vtxprob']*1./df_sig['sigmarv']+(1-df_sig['vtxprob'])*1./df_sig['sigmawv'])
sig_total_weight.clip(lower=0)
bkg_pfff_total_weight = df_bkg_pfff_weight['weight'] * df_bkg_pfff_weight['Norm_SFs']
bkg_pfff_total_weight.clip(lower=0)
bkg_pp_total_weight = df_bkg_pp_weight['weight'] * df_bkg_pp_weight['Norm_SFs']
bkg_pp_total_weight.clip(lower=0)
reweight_sig_bkg = (sum(bkg_pfff_total_weight) + sum(bkg_pp_total_weight)) / sum(sig_total_weight)
sig_total_weight = reweight_sig_bkg*sig_total_weight
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
data = Hgg_dataframe
variables = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv"]
rng = np.random.RandomState(31337)
traindataset, valdataset  = train_test_split(data, test_size=0.3, random_state=rng)
nS = len(traindataset.iloc[(traindataset.target.values == 1)])
nB = len(traindataset.iloc[(traindataset.target.values == 0)])
print("nB/nS:",nB/nS)
# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
# param_test2b = {
#  'min_child_weight':[6,8,10,12]
# }
# param_test3 = {
#  'gamma':[i/10.0 for i in range(0,5)]
# }
# param_test4 = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }
# param_test5 = {
#  'subsample':[i/100.0 for i in range(50,65,5)],
#  'colsample_bytree':[i/100.0 for i in range(85,100,5)]
# }
# param_test6 = {
#  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
# }
print("min_child_weight!!! \n")
param_test7 = {
#  'max_depth':range(3,20,2)
#  'min_child_weight':range(1,10,1)
#  'gamma':[i/10.0 for i in range(0,10)],
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)],
#  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}

gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=nB/nS, seed=27,tree_method='gpu_hist'), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=4, cv=5)

gsearch1.fit(traindataset[variables],traindataset['target'],traindataset['total_weight'].values)
print(gsearch1.best_params_)
print(gsearch1.best_score_)