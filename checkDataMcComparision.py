import pickle
import xgboost as xgb
import numpy as np
import json
import os 
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd
import sys
import ZZTime as zz
zztime = zz.outputTime()
file = "/hpcfs/cms/cmsgpu/zhangzhx/BDT/DiphotonXGboost_afterTune_withAllSigs_bayes.pkl"
with open(file, 'rb') as f:  
    clf = pickle.loads(f.read(), encoding='bytes')
#load bkg and data
eventsBkg = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
eventsData = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/New_UL2018data.root")
# variables = ['leadptom', 'subleadptom', 'leadeta', 'subleadeta', 'CosPhi', 'vtxprob', 'sigmarv', 'sigmawv'] #attention: remove subleadmva to check data/mc agreement
variables = ['leadmva', 'subleadmva', 'leadptom', 'subleadptom', 'leadeta', 'subleadeta', 'CosPhi', 'vtxprob', 'sigmarv', 'sigmawv']

probaBkgPP = clf.predict_proba(eventsBkg['pp'].arrays(variables,library='pd')[variables].values)
probaBkgDD = clf.predict_proba(eventsBkg['DataDriven_QCD'].arrays(variables,library='pd')[variables].values)
probaData = clf.predict_proba(eventsData['UL2018data'].arrays(variables,library='pd')[variables].values)

plt.subplot(2,1,1)
countsBkgs,binsBkgs,_ = plt.hist((probaBkgDD[:,1],probaBkgPP[:,1]),weights=(eventsBkg['DataDriven_QCD']['weight'].array() * eventsBkg['DataDriven_QCD']['Norm_SFs'].array(),eventsBkg['pp']['weight'].array() * eventsBkg['pp']['Norm_SFs'].array() ),stacked=True,bins=40)
countsData,binsData,_ = plt.hist(probaData[:,1],fill=None,bins=40)
plt.subplot(2,1,2)
bin_center=(binsBkgs[1:]+binsBkgs[:-1])/2
plt.scatter(bin_center,countsData/(countsBkgs[1,:]))
plt.savefig("DataOverMC"+ zztime + ".png")
plt.clf()


