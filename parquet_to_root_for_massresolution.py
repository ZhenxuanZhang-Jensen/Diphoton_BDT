import uproot
import os
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
import mplhep as hep

def get_score(events,file_name,input_features):
    import pickle
    # file_name = "DiphotonXGboost_afterTune_new_withAllSigs.pkl"
    # load
    xgb_model_loaded = pickle.load(open(file_name, "rb"))

    # xgb_model_loaded.predict(test)[0] == xgb_model.predict(test)[0]
    all_variables = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv",'sigmarvDecorr','CMS_hgg_mass',"diphoMVA",'weight', 'leadSigEOverE', 'subleadSigEOverE']
    df = events.arrays(all_variables,library='pd')
    df_mass = events.arrays(['CMS_hgg_mass'],library='pd')
    try:
        df_weight = events.arrays(['weight','Norm_SFs'],library='pd')
    except:
        df_weight = events.arrays(['weight'],library='pd')
    df['score'] = xgb_model_loaded.predict_proba(df[input_features].values)[:,1]
    return df, df_mass, df_weight

def get_score_for_root(events, file_name):
    # xgb_model_loaded.predict(test)[0] == xgb_model.predict(test)[0]
    all_variables = ["diphoMVA","Tran_DiphotonMVA_self", "DiphotonMVA_self"]
    df = events.arrays(all_variables,library='pd')
    df_mass = events.arrays(['CMS_hgg_mass'],library='pd')
    try:
        df_weight = events.arrays(['weight','Norm_SFs'],library='pd')
    except:
        df_weight = events.arrays(['weight'],library='pd')
    df['score'] = df['DiphotonMVA_self']
    return df, df_mass, df_weight

def get_score_arrays(events,file_name,input_features):
    import pickle
    # file_name = "DiphotonXGboost_afterTune_new_withAllSigs.pkl"
    # load
    xgb_model_loaded = pickle.load(open(file_name, "rb"))

    # xgb_model_loaded.predict(test)[0] == xgb_model.predict(test)[0]
    all_variables = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv","sigmawv",'sigmarvDecorr','CMS_hgg_mass',"diphoMVA",'weight', 'leadSigEOverE', 'subleadSigEOverE']
    df = events[all_variables]
    df_mass = events['CMS_hgg_mass']
    try:
        df_weight = events['weight','Norm_SFs']
    except:
        df_weight = events['weight']
    df['score'] = xgb_model_loaded.predict_proba(df[input_features].values)[:,1]
    return df, df_mass, df_weight


# read root
events_UL16_postVFP_pp = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:pp")
events_UL16_postVFP_sig = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/output_sig125.root:Sig125")
events_UL16_postVFP_dd = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:DataDriven_QCD")
events_UL16_postVFP_datasideband = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/New_UL2016data_postVFP.root:UL2016data_postVFP")

events_UL16_preVFP_pp = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:pp")
events_UL16_preVFP_datasideband = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/New_UL2016data_preVFP.root:UL2016data_preVFP")
events_UL16_preVFP_sig = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/output_sig125.root:Sig125")
events_UL16_preVFP_dd = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:DataDriven_QCD")

# get score
model_postVFP_powheg_normal = "forcheck_Fri_Mar_17_17:23:42_CSTUL16PostVFP_powheg_normal_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_powheg_normal = "forcheck_Fri_Mar_17_18:01:41_CSTUL16PreVFP_powheg_normal_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_powheg_masswindow_110140 = "forcheck_Fri_Mar_17_18:11:32_CSTUL16PreVFP_powheg_masswindow_110140_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_all_samples_masswindow = "forcheck_Mon_Mar_13_23:55:38_CSTUL16PreVFP_allsamples_masswindow_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_masswindow_with_sigmarvDecorr_signal_reweight = "forcheck_Tue_May_16_21:42:17_CSTUL16PreVFP_masswindow_with_sigmarvDecorr_signal_reweight__DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_all_samples_masswindow_110140 = "forcheck_Thu_Mar_16_16:24:42_CSTUL16PreVFP_masswindow_110140_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_masswindow_without_msweight = "forcheck_Mon_May_15_16:30:49_CSTUL16PreVFP_masswindow_without_msweight__DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_no_msweight_no_sigmawv = "forcheck_Tue_May_16_22:10:55_CSTUL16PreVFP_no_masswindow_no_msweight_no_sigmawv_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_no_msweight_no_sigmawv_sigmarv = "forcheck_Wed_May_17_02:20:02_CSTUL16PreVFP_no_masswindow_no_msweight_no_sigmawv_sigmarv_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_no_msweight_no_sigmarv = "forcheck_Wed_May_17_03:13:56_CSTUL16PreVFP_no_masswindow_no_msweight_no_sigmarv_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_masswindow_no_msweight_no_sigmarv = "forcheck_Fri_May_26_15:29:15_CSTUL16PreVFP_masswindow_no_msweight_no_sigmarv_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_no_msweight_no_vtxprob = "forcheck_Thu_May_18_03:39:08_CSTUL16PreVFP_no_masswindow_no_msweight_no_vtxprob_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_masswindow_msweight_combined = "Thu_May_18_04:23:52_CST_all_combined_mass_window_with_msweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_no_msweight_no_sigmarv_with_sigmarvDecorr  = "forcheck_Sun_May_21_22:00:24_CSTUL16PreVFP_no_masswindow_no_msweight_no_sigmarv_with_sigmarvDecorr_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_msweight_no_sigmarv  = "forcheck_Mon_May_22_04:03:04_CSTUL16PreVFP_no_masswindow_msweight_no_sigmarv_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_masswindow_msweight_no_sigmarv  = "forcheck_Mon_May_22_04:32:42_CSTUL16PreVFP_masswindow_msweight_no_sigmarv_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_msweight_no_sigmarv_with_sigmarvDecorr  = "forcheck_Mon_May_22_05:19:56_CSTUL16PreVFP_no_masswindow_msweight_no_sigmarv_with_sigmarvDecorr_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_msweight_no_sigmarv_no_sigmawv  = "forcheck_Mon_May_22_16:10:13_CSTUL16PreVFP_no_masswindow_msweight_no_sigmarv_no_sigmawv_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_msweight_W_flat_mgg  = "forcheck_Fri_May_26_04:26:50_CSTUL16PreVFP_no_masswindow_msweight_W_flat_mgg_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_msweight_W_flat_mgg_only_sig122127  = "forcheck_Fri_May_26_14:54:11_CSTUL16PreVFP_no_masswindow_msweight_W_flat_mgg_onlysig_122127_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_msweight_W_flat_mgg_only_pp  = "forcheck_Fri_May_26_15:05:31_CSTUL16PreVFP_no_masswindow_msweight_W_flat_mgg_onlypp_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_no_sigmarvwv_with_leadsublead_sigEoverE_withmsweight  = "forcheck_Sun_Jun__4_04:05:04UL16PreVFP_no_masswindow_no_sigmarvwv_with_leadsublead_sigEoverE_withmsweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_msweight_with_negativeweight  = "forcheck_Wed_Jun_14_18:25:21_CSTUL16PreVFP_no_masswindow_withmsweight_withsigmarvwv_DiphotonXGboost_afterTune_new_withAllSigs_withnegeight.pkl"
model_preVFP_no_masswindow_msweight_with_abs_negativeweight  = "forcheck_Thu_Jun_15_16:04:37_CSTUL16PreVFP_no_masswindow_withmsweight_withsigmarvwv_with_abs_negativeweights_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_msweight_with_reweight_negweight  = "forcheck_Sun_Jun_18_18:48:07_CSTUL16PreVFP_no_masswindow_withmsweight_withsigmarvwv_with_reweight_negweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_no_masswindow_msweight_with_reweight_negweight_wosigmarv  = "forcheck_Mon_Jun_19_02:44:29_CSTUL16PreVFP_no_masswindow_withmsweight_withoutsigmarv_with_reweight_negweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl"

model_combined_mass_window_with_msweight  = "Thu_May_18_04:23:52_CST_all_combined_mass_window_with_msweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl"

model_preVFP_benmark = "forcheck_Mon_May_22_03:17:37_CSTUL16PreVFP_normal_benchmark_no_masswindow_msweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_postVFP_masswindow_without_msweight = "forcheck_Mon_May_15_16:30:49_CSTUL16PostVFP_masswindow_without_msweight__DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_preVFP_masswindow_without_msweight = "forcheck_Mon_May_15_16:30:49_CSTUL16PreVFP_masswindow_without_msweight__DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_postVFP_all_samples_masswindow = "forcheck_Mon_Mar_13_23:55:38_CSTUL16PostVFP_allsamples_masswindow_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_postVFP_masswindow_without_msweight = "forcheck_Mon_May_15_16:30:49_CSTUL16PostVFP_masswindow_without_msweight__DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_postVFP_benmark = "forcheck_Tue_Mar_14_00:18:33_CSTUL16PostVFP_normal_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_postVFP_no_masswindow_msweight_with_reweight_negweight_wosigmarv  = "forcheck_Tue_Jun_27_16:16:52_CSTUL16PostVFP_no_masswindow_with_msweight_nosigmarv_with_reweight_negweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl"



# UL17
events_UL17_sig = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/Sig125_negReweighting.root:Sig125")
events_UL17_pp = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:pp")
events_UL17_dd = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:DataDriven_QCD")

model_UL17_no_masswindow_msweight_with_reweight_negweight_wosigmarv = "forcheck_Mon_Aug_14_22:57:29_CSTUL17_no_masswindow_withmsweight_withoutsigmarv_with_reweight_negweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
# model_UL17_benmark = "Sun_Feb_26_23:43:22_CST_UL17__DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_UL17_benmark = "UL17_DiphotonXGboost_afterTune.pkl"

#UL18 
events_UL18_sig = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/Sig125_negReweighting.root:Sig125")
events_UL18_pp = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:pp")
events_UL18_dd = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:DataDriven_QCD")

model_UL18_no_masswindow_msweight_with_reweight_negweight_wosigmarv = "forcheck_Wed_Aug_16_02:14:51_CSTUL18_no_masswindow_withmsweight_withoutsigmarv_with_reweight_negweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
model_UL18_benchmark = "forcheck_Wed_Mar_29_15:08:31_CSTUL18_normal_DiphotonXGboost_afterTune_new_withAllSigs.pkl"
# need to change
events_pp = events_UL17_pp
events_dd = events_UL17_dd
events_sig = events_UL17_sig
# events_data = events_UL16_preVFP_datasideband
model_name = model_postVFP_no_masswindow_msweight_with_reweight_negweight_wosigmarv
# ---------------
# input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","leadSigEOverE","subleadSigEOverE"]
# input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv", "sigmawv"]
input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob", "sigmawv"]
# input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob",'sigmarvDecorr',"sigmawv"]
# input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob"]
# df_pp,df_pp_mass, df_pp_weight = get_score(events_pp,model_name, input_features)
# df_data,df_data_mass, df_data_weight = get_score(events_data,model_name, input_features)
# df_dd,df_dd_mass, df_dd_weight = get_score(events_dd,model_name, input_features)
# df_dd,df_dd_mass, df_dd_weight = get_score(events_dd,model_name, input_features)

# df_signal,df_signal_mass, df_signal_weight = get_score(events_sig,model_name)
# df_signal_benchmark,df_signal_mass_benchmark, df_signal_weight_benchmark = get_score(events_sig,model_benmark_name)

input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob", "sigmawv"]

# model_postVFP_no_masswindow_msweight_with_reweight_negweight_wosigmarv
# model_preVFP_no_masswindow_msweight_with_reweight_negweight_wosigmarv
# read all the files
# input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv", "sigmawv"]
df_pp,df_pp_mass, df_pp_weight = get_score(events_UL18_pp,"/hpcfs/cms/cmsgpu/zhangzhx/BDT/forcheck_Fri_Aug_18_01:08:16_CSTUL18_no_masswindow_withmsweight_withoutsigmarv_with_reweight_negweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl",input_features)
# events_UL18_sig = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/output_sig125.root:Sig125")
df_signal ,df_signal_mass, df_signal_weight = get_score(events_UL18_sig,"/hpcfs/cms/cmsgpu/zhangzhx/BDT/forcheck_Fri_Aug_18_01:08:16_CSTUL18_no_masswindow_withmsweight_withoutsigmarv_with_reweight_negweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl",input_features)
df_dd,df_dd_mass, df_dd_weight = get_score(events_UL18_dd,"/hpcfs/cms/cmsgpu/zhangzhx/BDT/forcheck_Fri_Aug_18_01:08:16_CSTUL18_no_masswindow_withmsweight_withoutsigmarv_with_reweight_negweight_DiphotonXGboost_afterTune_new_withAllSigs.pkl",input_features)

input_features = ["leadmva","subleadmva","leadptom","subleadptom","leadeta","subleadeta","CosPhi","vtxprob","sigmarv", "sigmawv"]
df_signal_benchmark ,df_signal_mass_benchmark, df_signal_weight_benchmark = get_score(events_UL18_sig,"/hpcfs/cms/cmsgpu/zhangzhx/BDT/forcheck_Wed_Mar_29_16:22:52_CSTUL18_normal_DiphotonXGboost_afterTune_new_withAllSigs.pkl",input_features)
df_dd_benchmark,df_dd_mass_benchmark, df_dd_weight_benchmark = get_score(events_UL18_dd,"/hpcfs/cms/cmsgpu/zhangzhx/BDT/forcheck_Wed_Mar_29_16:22:52_CSTUL18_normal_DiphotonXGboost_afterTune_new_withAllSigs.pkl",input_features)
df_pp_benchmark,df_pp_mass_benchmark, df_pp_weight_benchmark = get_score(events_UL18_pp,"/hpcfs/cms/cmsgpu/zhangzhx/BDT/forcheck_Wed_Mar_29_16:22:52_CSTUL18_normal_DiphotonXGboost_afterTune_new_withAllSigs.pkl",input_features)

postVFP_option_n1_cut_value_list = [0.7200, 0.8619, 0.9329, 0.9708,1.0]
postVFP_option9_cut_value_list = [0.54137095, 0.82419164, 0.90482946, 0.97426596,1.0]
preVFP_option9_cut_value_list = [0.71068919, 0.85628904, 0.9154733,  0.95630979, 1.0]
preVFP_optino3_cut_value_list = [0.728399, 0.857225, 0.933583, 0.973383, 1.0]
preVFP_option_n1_value_list = [0.7455, 0.8817, 0.9376, 0.9685, 1.0]
postVFP_option9_cut_value_list_new = [0.729792, 0.8701, 0.9301, 0.9701,1.0]
UL17_option9_cut_value = [0.4980, 0.8136, 0.9804, 0.9967]
UL18_option9_cut_value = [0.7265, 0.8344, 0.9077, 0.967]

# create root file for df_signal with cuts
df = df_signal_benchmark[df_signal_benchmark['score'] > UL18_option9_cut_value[3]]
df['CMS_hgg_mass'] = df_signal_mass_benchmark['CMS_hgg_mass']
df['weight'] = df['weight']
df['dZ'] = np.ones(len(df))
from parquet_to_root import parquet_to_root
df.to_parquet("benchmark_tmp.parquet")
# parquet to root
from parquet_to_root import parquet_to_root
parquet_to_root("benchmark_tmp.parquet", "UL18_old_strategy_mass.root", treename="gghh_125_13TeV_RECO_untagged", verbose=False)
