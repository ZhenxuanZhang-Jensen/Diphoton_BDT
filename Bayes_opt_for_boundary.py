import numpy as np
import scipy
from bayes_opt import BayesianOptimization
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os
def get_fwhm(hist_sig, bin_sig):
    peak_index = np.argmax(hist_sig)
    peak_mass = 0.5 * (bin_sig[peak_index] + bin_sig[peak_index + 1])
    half_max_height = 0.5 * hist_sig[peak_index]
    # 找到半最大值左侧位置
    left_index = np.where(hist_sig[:peak_index] <= half_max_height)[0][-1]
    # 找到半最大值右侧位置
    right_index = np.where(hist_sig[peak_index:] <= half_max_height)[0][0] + peak_index
    fwhm = bin_sig[right_index] - bin_sig[left_index]
    return fwhm

def get_score(events,file_name,input_features):
    import pickle
    # file_name = "DiphotonXGboost_afterTune_new_withAllSigs.pkl"
    # load
    xgb_model_loaded = pickle.load(open(file_name, "rb"))

    # xgb_model_loaded.predict(test)[0] == xgb_model.predict(test)[0]
    all_variables = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv",'sigmarvDecorr','CMS_hgg_mass',"diphoMVA",'weight', 'leadSigEOverE', 'subleadSigEOverE','leadSCeta','subleadSCeta']
    # create sigmarv over sigmawv
    try:
        df = events.arrays(all_variables + ['Norm_SFs'],library='pd')
    except:
        df = events.arrays(all_variables ,library='pd')
    df['score'] = xgb_model_loaded.predict_proba(df[input_features].values)[:,1]
    return df

def calculate_FWHM_significance(events_sig, events_bkg_pp, events_bkg_dd, EBEB):
    debug = False
        
    # start opt byes
    def opt_bo(bo1, bo2, bo3, bo4):
        # divide them based on BDT score
        # events1 -> [1-bo1, 1]
        events_sig1 = events_sig[(events_sig['score'] >= 1-bo1) & (events_sig['score'] <=1)]
        events_bkg_pp1 = events_bkg_pp[(events_bkg_pp['score'] >= 1-bo1) & (events_bkg_pp['score'] <=1)]
        events_bkg_dd1 = events_bkg_dd[(events_bkg_dd['score'] >= 1-bo1) & (events_bkg_dd['score'] <=1)]
        # events2 -> [1-bo2-bo1, 1-bo1)
        events_sig2 = events_sig[(events_sig['score'] >= 1-bo2-bo1) & (events_sig['score'] <1-bo1)]
        events_bkg_pp2 = events_bkg_pp[(events_bkg_pp['score'] >= 1-bo2-bo1) & (events_bkg_pp['score'] <1-bo1)]
        events_bkg_dd2 = events_bkg_dd[(events_bkg_dd['score'] >= 1-bo2-bo1) & (events_bkg_dd['score'] <1-bo1)]
        # events3 -> [1-bo3-bo2-bo1, 1-bo2-bo1)
        events_sig3 = events_sig[(events_sig['score'] >= 1-bo3-bo2-bo1) & (events_sig['score'] <1-bo2-bo1)]
        events_bkg_pp3 = events_bkg_pp[(events_bkg_pp['score'] >= 1-bo3-bo2-bo1) & (events_bkg_pp['score'] <1-bo2-bo1)]
        events_bkg_dd3 = events_bkg_dd[(events_bkg_dd['score'] >= 1-bo3-bo2-bo1) & (events_bkg_dd['score'] <1-bo2-bo1)]
        # events4 -> [1-bo4-bo3-bo2-bo1, 1-bo3-bo2-bo1)
        events_sig4 = events_sig[(events_sig['score'] >= 1-bo4-bo3-bo2-bo1) & (events_sig['score'] <1-bo3-bo2-bo1)]
        events_bkg_pp4 = events_bkg_pp[(events_bkg_pp['score'] >= 1-bo4-bo3-bo2-bo1) & (events_bkg_pp['score'] <1-bo3-bo2-bo1)]
        events_bkg_dd4 = events_bkg_dd[(events_bkg_dd['score'] >= 1-bo4-bo3-bo2-bo1) & (events_bkg_dd['score'] <1-bo3-bo2-bo1)]
        if debug:
            print('events_sig1', len(events_sig1))
            print('events_sig2', len(events_sig2))
            print('events_sig3', len(events_sig3))
            print('events_sig4', len(events_sig4))
            print('events_bkg_pp1', len(events_bkg_pp1))
            print('events_bkg_pp2', len(events_bkg_pp2))
            print('events_bkg_pp3', len(events_bkg_pp3))
            print('events_bkg_pp4', len(events_bkg_pp4))
        # if no events in any regions, return 0
        if (np.sum(events_bkg_pp1['weight']) + np.sum(events_bkg_dd1['weight'])) < 20 or (np.sum(events_bkg_pp2['weight']) + np.sum(events_bkg_dd2['weight'])) < 20 or (np.sum(events_bkg_pp3['weight']) + np.sum(events_bkg_dd3['weight'])) < 20 or (np.sum(events_bkg_pp4['weight']) + np.sum(events_bkg_dd4['weight'])) < 20:
            print('not good bo')
            print('bkg weight not >20')
            return 0
        
        # make sure number of events cat1 < cat2 < cat3 < cat4
        if len(events_bkg_pp1) > len(events_bkg_pp2) or len(events_bkg_pp2) > len(events_bkg_pp3) or len(events_bkg_pp3) > len(events_bkg_pp4):
            print('not good bo')
            print('number of events not in order')
            return 0
        
        print("bo1: ", bo1)
        print("bo2: ", bo2)
        print("bo3: ", bo3)
        print("bo4: ", bo4)
        # keep five digits
        bo1 = np.round(bo1, 6)
        bo2 = np.round(bo2, 6)
        bo3 = np.round(bo3, 6)
        bo4 = np.round(bo4, 6)
        # print cat
        print("cat1: ", 1-bo1, " - ", 1)
        print("cat2: ", 1-bo2-bo1, " - ", 1-bo1)
        print("cat3: ", 1-bo3-bo2-bo1, " - ", 1-bo2-bo1)
        print("cat4: ", 1-bo4-bo3-bo2-bo1, " - ", 1-bo3-bo2-bo1)
            
        if debug:
            print("len(events_sig1): ", len(events_sig1))
            print("len(events_sig2): ", len(events_sig2))
            print("len(events_sig3): ", len(events_sig3))
            print("len(events_sig4): ", len(events_sig4))

        
        # get 1,2,3,4 sig and bkg fwhm and significance
        # 1
        # check signal mass distribution
        hist_sig1, bin_sig1 = np.histogram(events_sig1['CMS_hgg_mass'], bins=100, range=(110,150), weights=events_sig1['weight'])
        if debug:
            print("hist_sig1: ", hist_sig1)
            print("bin_sig1: ", bin_sig1)
        # get signal mass FWHM
        fwhm1 = get_fwhm(hist_sig1, bin_sig1)
        # get significance s/sqrt(s+b)
        s = np.sum(events_sig1['weight'][(events_sig1['CMS_hgg_mass'] > 115) & (events_sig1['CMS_hgg_mass'] < 135)]) * 41.5
        pp1_weight = (events_bkg_pp1['weight']*events_bkg_pp1['Norm_SFs'])[(events_bkg_pp1['CMS_hgg_mass'] > 115) & (events_bkg_pp1['CMS_hgg_mass'] < 135)]
        dd1_weight = (events_bkg_dd1['weight']*events_bkg_dd1['Norm_SFs'])[(events_bkg_dd1['CMS_hgg_mass'] > 115) & (events_bkg_dd1['CMS_hgg_mass'] < 135)]
        b = np.sum(pp1_weight) + np.sum(dd1_weight)
        significance1 = s / np.sqrt(s + b)
        # 2
        # check signal mass distribution
        hist_sig2, bin_sig2 = np.histogram(events_sig2['CMS_hgg_mass'], bins=100, range=(110,150), weights=events_sig2['weight'])
        # check signal mass FWHM
        fwhm2 = get_fwhm(hist_sig2, bin_sig2)
        # get significance s/sqrt(s+b)
        s = np.sum(events_sig2['weight'][(events_sig2['CMS_hgg_mass'] > 115) & (events_sig2['CMS_hgg_mass'] < 135)]) * 41.5
        pp2_weight = (events_bkg_pp2['weight']*events_bkg_pp2['Norm_SFs'])[(events_bkg_pp2['CMS_hgg_mass'] > 115) & (events_bkg_pp2['CMS_hgg_mass'] < 135)]
        dd2_weight = (events_bkg_dd2['weight']*events_bkg_dd2['Norm_SFs'])[(events_bkg_dd2['CMS_hgg_mass'] > 115) & (events_bkg_dd2['CMS_hgg_mass'] < 135)]
        b = np.sum(pp2_weight) + np.sum(dd2_weight)
        significance2 = s / np.sqrt(s + b)
        # 3
        # check signal mass distribution
        hist_sig3, bin_sig3 = np.histogram(events_sig3['CMS_hgg_mass'], bins=100, range=(110,150), weights=events_sig3['weight'])
        # check signal mass FWHM
        fwhm3 = get_fwhm(hist_sig3, bin_sig3)
        # get significance s/sqrt(s+b)
        s = np.sum(events_sig3['weight'][(events_sig3['CMS_hgg_mass'] > 115) & (events_sig3['CMS_hgg_mass'] < 135)]) * 41.5
        pp3_weight = (events_bkg_pp3['weight']*events_bkg_pp3['Norm_SFs'])[(events_bkg_pp3['CMS_hgg_mass'] > 115) & (events_bkg_pp3['CMS_hgg_mass'] < 135)]
        dd3_weight = (events_bkg_dd3['weight']*events_bkg_dd3['Norm_SFs'])[(events_bkg_dd3['CMS_hgg_mass'] > 115) & (events_bkg_dd3['CMS_hgg_mass'] < 135)]
        b = np.sum(pp3_weight) + np.sum(dd3_weight)
        significance3 = s / np.sqrt(s + b)
        # 4
        # check signal mass distribution
        hist_sig4, bin_sig4 = np.histogram(events_sig4['CMS_hgg_mass'], bins=100, range=(110,150), weights=events_sig4['weight'])
        # check signal mass FWHM
        fwhm4 = get_fwhm(hist_sig4, bin_sig4)
        # get significance s/sqrt(s+b)
        s = np.sum(events_sig4['weight'][(events_sig4['CMS_hgg_mass'] > 115) & (events_sig4['CMS_hgg_mass'] < 135)]) * 41.5
        pp4_weight = (events_bkg_pp4['weight']*events_bkg_pp4['Norm_SFs'])[(events_bkg_pp4['CMS_hgg_mass'] > 115) & (events_bkg_pp4['CMS_hgg_mass'] < 135)]
        dd4_weight = (events_bkg_dd4['weight']*events_bkg_dd4['Norm_SFs'])[(events_bkg_dd4['CMS_hgg_mass'] > 115) & (events_bkg_dd4['CMS_hgg_mass'] < 135)]
        b = np.sum(pp4_weight) + np.sum(dd4_weight)
        significance4 = s / np.sqrt(s + b)
        
        # combine total significance with square root
        significance = np.sqrt(significance1**2 + significance2**2 + significance3**2 + significance4**2)
        # combine total fwhm with square root
        fwhm = np.sqrt(fwhm1**2 + fwhm2**2 + fwhm3**2 + fwhm4**2)
        print('significance: ', significance)
        print('fwhm: ', fwhm)
        print('significance/fwhm: ', significance/fwhm)
        return significance / fwhm 

    if EBEB is True:
        print("is EBEB")
        pbounds = {
        'bo1': (0.0001, 0.05), 
        'bo2': (0.05, 0.2), 
        'bo3': (0.05, 0.2), 
        'bo4': (0.1, 0.3)
        }
        optimizer = BayesianOptimization(
            f=opt_bo,
            pbounds=pbounds,
            random_state=100,
        )   
        optimizer.maximize(init_points=5,n_iter=2000)

        #Extracting the best parameters
        params = optimizer.max['params']
        print(params)
        # print the real boundary based on bo1, bo2, bo3, bo4
        bo1 = params['bo1']
        bo2 = params['bo2']
        bo3 = params['bo3']
        bo4 = params['bo4']
        # events1 -> [1-bo1, 1]
        # events2 -> [1-bo2-bo1, 1-bo1)
        # events3 -> [1-bo3-bo2-bo1, 1-bo2-bo1]
        # events4 -> [1-bo4-bo3-bo2-bo1, 1-bo3-bo2-bo1)
        print(f'the first category is [%.6f, 1]'%(1-bo1))
        print(f'the second category is [%.6f, %.6f]'%(1-bo2-bo1, 1-bo1))
        print(f'the third category is [%.6f, %.6f]'%(1-bo3-bo2-bo1, 1-bo2-bo1))
        print(f'the fourth category is [%.6f, %.6f]'%(1-bo4-bo3-bo2-bo1, 1-bo3-bo2-bo1))
    else:
        print("is !EBEB")
        pbounds = {
        'bo1': (0.0001, 0.05), 
        'bo2': (0.05, 0.2), 
        'bo3': (0.05, 0.2), 
        'bo4': (0.1, 0.3)
        }
        optimizer = BayesianOptimization(
            f=opt_bo,
            pbounds=pbounds,
            random_state=100,
        )   
        optimizer.maximize(init_points=5,n_iter=2000)

        #Extracting the best parameters
        params = optimizer.max['params']
        print(params)
        # print the real boundary based on bo1, bo2, bo3, bo4
        bo1 = params['bo1']
        bo2 = params['bo2']
        bo3 = params['bo3']
        bo4 = params['bo4']
        # events1 -> [1-bo1, 1]
        # events2 -> [1-bo2-bo1, 1-bo1)
        # events3 -> [1-bo3-bo2-bo1, 1-bo2-bo1]
        # events4 -> [1-bo4-bo3-bo2-bo1, 1-bo3-bo2-bo1)
        print(f'the first category is [%.6f, 1]'%(1-bo1))
        print(f'the second category is [%.6f, %.6f]'%(1-bo2-bo1, 1-bo1))
        print(f'the third category is [%.6f, %.6f]'%(1-bo3-bo2-bo1, 1-bo2-bo1))
        print(f'the fourth category is [%.6f, %.6f]'%(1-bo4-bo3-bo2-bo1, 1-bo3-bo2-bo1))






if __name__ == "__main__":
    samples = "UL17 masswindow no sigmrv"
    input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmawv"]
    print("sample is:", samples)
    # events_sig_UL16PostVFP = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/output_sig125_new.root:Sig125")
    events_sig_17 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/Summer20/Sig125_v2_negReweighting.root:Sig125")
    events_bkg_17 = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/Summer20/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root")
    model = "/hpcfs/cms/cmsgpu/zhangzhx/BDT/data/forcheck_Fri_Dec__8_02:06:31UL17_nomasswindow_withmsweight_withoutsigmarv_with_reweight_negweight_tuenedHP_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
    df = get_score(events_sig_17,model, input_features)    
    df_bkg_pp = get_score(events_bkg_17["pp"],model, input_features)
    df_bkg_dd = get_score(events_bkg_17["DataDriven_QCD"],model, input_features)
        
    # EBEB photon
    df_EBEB = df[(np.abs(df["leadSCeta"].values) < 1.4443) & (np.abs(df["subleadSCeta"].values) < 1.4443)]
    df_bkg_pp_EBEB = df_bkg_pp[(np.abs(df_bkg_pp["leadSCeta"].values) < 1.4443) & (np.abs(df_bkg_pp["subleadSCeta"].values) < 1.4443)]
    df_bkg_dd_EBEB = df_bkg_dd[(np.abs(df_bkg_dd["leadSCeta"].values) < 1.4443) & (np.abs(df_bkg_dd["subleadSCeta"].values) < 1.4443)]
    
    calculate_FWHM_significance(df_EBEB, df_bkg_pp_EBEB, df_bkg_dd_EBEB, EBEB=True)
    
    # not EBEB photon
    df_notEBEB = df[(np.abs(df["leadSCeta"].values) > 1.4443) | (np.abs(df["subleadSCeta"].values) > 1.4443)]
    df_bkg_pp_notEBEB = df_bkg_pp[(np.abs(df_bkg_pp["leadSCeta"].values) > 1.4443) | (np.abs(df_bkg_pp["subleadSCeta"].values) > 1.4443)]
    df_bkg_dd_notEBEB = df_bkg_dd[(np.abs(df_bkg_dd["leadSCeta"].values) > 1.4443) | (np.abs(df_bkg_dd["subleadSCeta"].values) > 1.4443)]
    calculate_FWHM_significance(df_notEBEB, df_bkg_pp_notEBEB, df_bkg_dd_notEBEB, EBEB=False)

    

