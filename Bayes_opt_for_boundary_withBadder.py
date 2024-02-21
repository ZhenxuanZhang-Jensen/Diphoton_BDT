import scipy
from bayes_opt import BayesianOptimization
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os

def get_err_variance(err_mean, variance, mean, x, w):
    """ uncertainty on variance is (m4 - m2^2) / (4 n m2)"""
    if err_mean == np.inf:
        return np.inf
    if sum(w) == 0:
        return np.inf

    m4 = np.average(np.power(x - mean, 4), weights=w)
    return (m4 - variance**2)/(4 * len(x) * variance)
def check_bad_iteration(events_sig, events_bkg_pp, events_bkg_dd):
    check_bad = False
    ''' check bad iteration which have too small events and mass turn on'''
    # check if events_sig['weight'] is empty
    if len(events_sig['weight']) == 0:
        print("events_sig['weight'] is empty")
        check_bad = True
    # check if bkg pp + dd events are smaller than 100
    if np.sum(events_bkg_pp['weight']) + np.sum(events_bkg_dd['weight']) < 100:
        print("bkg pp + dd events are smaller than 100")
        check_bad = True
    # check if mass turn on
    # mass turn on is defined that small bins have lower events for bkg
    # get the hist of events_bkg
    events_bkg =  pd.concat([events_bkg_pp, events_bkg_dd])
    hist_bkg, bin_bkg = np.histogram(events_bkg['CMS_hgg_mass'], bins=10, range=(100, 115))
    # check if each bin events is in order
    if np.sum(np.sort(hist_bkg)[::-1] == hist_bkg) == 1:
        print("mass turn on")
        check_bad = True
    return check_bad
def get_significance_resolution(MVA_boundary_list, events_sig, events_bkg_pp, events_bkg_dd):
    ''' get significance and resolution for each category'''
    # cut by MVA boundary list
    events_sig =    events_sig[(events_sig['score'] >= MVA_boundary_list[0]) & (events_sig['score'] <= MVA_boundary_list[1]) ]
    events_bkg_pp = events_bkg_pp[(events_bkg_pp['score'] >= MVA_boundary_list[0]) & (events_bkg_pp['score'] <= MVA_boundary_list[1]) ]
    events_bkg_dd = events_bkg_dd[(events_bkg_dd['score'] >= MVA_boundary_list[0]) & (events_bkg_dd['score'] <= MVA_boundary_list[1]) ]
    # check any bad iteration
    check_bad = check_bad_iteration(events_sig, events_bkg_pp, events_bkg_dd)
    # before mass mask
    events_sig =    events_sig[(events_sig['CMS_hgg_mass'] > 115) & (events_sig['CMS_hgg_mass'] < 135)]
    events_bkg_pp = events_bkg_pp[(events_bkg_pp['CMS_hgg_mass'] > 115) & (events_bkg_pp['CMS_hgg_mass'] < 135)]
    events_bkg_dd = events_bkg_dd[(events_bkg_dd['CMS_hgg_mass'] > 115) & (events_bkg_dd['CMS_hgg_mass'] < 135)]
    if check_bad == True:
        return 0, 0, check_bad
    events_sig['weight'] = events_sig['weight']
    s = np.sum(events_sig['weight'].values)
    pp_weight = (events_bkg_pp['weight'].values)
    dd_weight = (events_bkg_dd['weight'].values)
    b = np.sum(pp_weight) + np.sum(dd_weight)
    significance = s**2 / (b)
    
    # get the resolution
    # calculate mean
    mean = np.average(events_sig['CMS_hgg_mass'], weights=events_sig['weight'])
    # calculate variance
    variance = np.average(np.power((events_sig['CMS_hgg_mass'] - mean), 2), weights=events_sig['weight'])
    # calculate error of mean
    err_mean = np.sqrt(variance)/np.sum(events_sig['weight'])
    # calculate error of variance
    err_variance = get_err_variance(err_mean=err_mean, variance=variance, mean=mean, x=events_sig['CMS_hgg_mass'], w=events_sig['weight'])
    # create a res_dict dictionary to save mean, variance, err_mean, err_variance
    res_dict = {'mean': mean, 'variance': variance, 'err_mean': err_mean, 'err_variance': err_variance}
    return significance, res_dict, check_bad

def get_combined_resolution(category_info_list):
    ''' get combined resolution'''
    debug = False
    sigma = 0
    sigma_weight = 0
    mean = 0
    mean_weight = 0
    for cat in category_info_list:
        if debug:
            print('cat["mean"]', cat["mean"])
            print('cat["err_mean"]', cat["err_mean"])
            print('cat["variance"]', cat["variance"])
            print('cat["err_variance"]', cat["err_variance"])
        sigma += cat["variance"] / cat["err_variance"]
        sigma_weight += 1. / cat["err_variance"]
        mean += cat["mean"] / cat["err_mean"]**2
        mean_weight += 1. / cat["err_mean"]**2
    if mean_weight == 0:
        return -9999., -9999.
    w_mean = mean/mean_weight
    w_mean_unc = np.sqrt(1/mean_weight)
    w_sigma = np.sqrt(sigma/sigma_weight)
    w_sigma_unc = np.sqrt(1/sigma_weight)
    res = w_sigma/w_mean
    res_err = res * np.sqrt((w_mean_unc/w_mean)**2 
                            + (w_sigma_unc/w_sigma)**2)
    
    return res, res_err


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
        print("bo1: ", bo1)
        print("bo2: ", bo2)
        print("bo3: ", bo3)
        print("bo4: ", bo4)
        # keep five digits
        bo1 = np.round(bo1, 6)
        bo2 = np.round(bo2, 6)
        bo3 = np.round(bo3, 6)
        bo4 = np.round(bo4, 6)
        # get each category mva boundary
        cat1 = 1 - bo1
        cat2 = 1 - bo2 - bo1
        cat3 = 1 - bo3 - bo2 - bo1
        cat4 = 1 - bo4 - bo3 - bo2 - bo1
        # [0.69007544 0.83496144 0.91988657 0.954461  ]
        cat1 = 0.954461
        cat2 = 0.91988657
        cat3 = 0.83496144
        cat4 = 0.69007544
        s1, res_dict1, check_bad1 = get_significance_resolution([cat1, 1], events_sig, events_bkg_pp, events_bkg_dd)
        s2, res_dict2, check_bad2 = get_significance_resolution([cat2, cat1], events_sig, events_bkg_pp, events_bkg_dd)
        s3, res_dict3, check_bad3 = get_significance_resolution([cat3, cat2], events_sig, events_bkg_pp, events_bkg_dd)
        s4, res_dict4, check_bad4 = get_significance_resolution([cat4, cat3], events_sig, events_bkg_pp, events_bkg_dd)
        if check_bad1 or check_bad2 or check_bad3 or check_bad4:
            return -999
        res_list = [res_dict1, res_dict2, res_dict3, res_dict4]
        soverb = s1 + s2 + s3 + s4
        ret = 999
        res, res_err =  get_combined_resolution(res_list)
        if soverb != 0 and not np.isnan(soverb):
            ret = 1000*res/soverb
        target = -ret # maximize -ret = minimize ret
        # print cat
        print("cat1: ", 1-bo1, " - ", 1)
        print("cat2: ", 1-bo2-bo1, " - ", 1-bo1)
        print("cat3: ", 1-bo3-bo2-bo1, " - ", 1-bo2-bo1)
        print("cat4: ", 1-bo4-bo3-bo2-bo1, " - ", 1-bo3-bo2-bo1)
        
        print('soverb: ', soverb)
        print('res: ', res)
        print('soverb/res: ', soverb/res)
        print("target", target)
        
        return target

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
        optimizer.maximize(init_points=10,n_iter=1000)

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
    df_EBEB = df[(np.abs(df["leadeta"].values) < 1.479) & (np.abs(df["subleadeta"].values) < 1.479)]
    df_bkg_pp_EBEB = df_bkg_pp[(np.abs(df_bkg_pp["leadeta"].values) < 1.479) & (np.abs(df_bkg_pp["subleadeta"].values) < 1.479)]
    df_bkg_dd_EBEB = df_bkg_dd[(np.abs(df_bkg_dd["leadeta"].values) < 1.479) & (np.abs(df_bkg_dd["subleadeta"].values) < 1.479)]
    df_EBEB.to_parquet("df_EBEB.parquet")
    
    calculate_FWHM_significance(df_EBEB, df_bkg_pp_EBEB, df_bkg_dd_EBEB, EBEB=True)
    
    # not EBEB photon
    # df_notEBEB = df[(np.abs(df["leadSCeta"].values) > 1.4443) | (np.abs(df["subleadSCeta"].values) > 1.4443)]
    # df_bkg_pp_notEBEB = df_bkg_pp[(np.abs(df_bkg_pp["leadSCeta"].values) > 1.4443) | (np.abs(df_bkg_pp["subleadSCeta"].values) > 1.4443)]
    # df_bkg_dd_notEBEB = df_bkg_dd[(np.abs(df_bkg_dd["leadSCeta"].values) > 1.4443) | (np.abs(df_bkg_dd["subleadSCeta"].values) > 1.4443)]
    # calculate_FWHM_significance(df_notEBEB, df_bkg_pp_notEBEB, df_bkg_dd_notEBEB, EBEB=False)

    

