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
other_params = {'eta': 0.22, 'n_estimators': 200, 'gamma': 0.78, 'max_depth': 5, 'min_child_weight': 6,
                'colsample_bytree': 1, 'colsample_bylevel': 0.9, 'subsample': 0.8, 'reg_lambda': 90, 'reg_alpha': 7,'scale_pos_weight':nB/nS,
                'seed': 33,'tree_method':'gpu_hist'}
cv_params = {'n_estimators': np.linspace(100, 1000, 10, dtype=int)}
# cv_params = {'max_depth': np.linspace(1, 10, 10, dtype=int)}5
# cv_params = {'min_child_weight': np.linspace(1, 10, 10, dtype=int)}
# cv_params = {'gamma': np.linspace(0, 1, 10)}
# cv_params = {'gamma': np.linspace(0, 0.1, 11)}
# cv_params = {'subsample': np.linspace(0, 1, 11)}
# cv_params = {'subsample': np.linspace(0.9, 1, 11)}
# cv_params = {'colsample_bytree': np.linspace(0, 1, 11)[1:]}
# cv_params = {'reg_lambda': np.linspace(0, 100, 11)}
# cv_params = {'reg_lambda': np.linspace(80, 100, 11)}                
# cv_params = {'reg_alpha': np.linspace(0, 10, 11)}
# cv_params = {'reg_alpha': np.linspace(0, 1, 11)}
# cv_params = {'eta': np.logspace(-2, 0, 10)}
print('cv_params',cv_params)
classifier_model = xgb.XGBClassifier(**other_params)  # 注意这里的两个 * 号！
gs = GridSearchCV(classifier_model, cv_params, verbose=2, refit=True, cv=5, n_jobs=-1)
gs.fit(traindataset[variables],traindataset['target'],traindataset['total_weight'].values)  # X为训练数据的特征值，y为训练数据的label
# 性能测评
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)