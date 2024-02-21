
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
import pandas as pd
df = pd.read_csv("xgboost_investment.csv")
rng = np.random.RandomState(31337)
variables = ['macd_dif','macd','kdj_k','kdj_d','kdj_j','rsi_6','rsi_12','cci']
indicators =['slowk','slowd','fastk','fastd','WILLR']
variables = variables + indicators
data = df
traindataset, valdataset  = train_test_split(data, test_size=0.3, random_state=rng)
nS = len(traindataset.iloc[(traindataset.target.values == 1)])
nB = len(traindataset.iloc[(traindataset.target.values == 0)])
print("nB/nS:",nB/nS)
param_test7 = {
 'max_depth':range(3,20,2),
 'min_child_weight':range(1,100,2),
 'gamma':[i/10.0 for i in range(0,10)],
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)],
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=nB/nS, seed=27,tree_method='gpu_hist'), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=4, cv=5)

gsearch1.fit(traindataset[variables],traindataset['target'])
print(gsearch1.best_params_)
print(gsearch1.best_score_)