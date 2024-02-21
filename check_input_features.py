# find out removing the sigmarv , mass turn on problem will be cured
import uproot
import os
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# so check sigmarv
# distribution in each year
# read root
events_UL16_postVFP_pp = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:pp")
events_UL16_postVFP_sig = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/output_sig125.root:Sig125")
events_UL16_postVFP_dd = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PostVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:DataDriven_QCD")

events_UL16_preVFP_pp = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:pp")
events_UL16_preVFP_datasideband = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/New_UL2016data_preVFP.root:UL2016data_preVFP")
events_UL16_preVFP_sig = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/output_sig125.root:Sig125")
events_UL16_preVFP_dd = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2016PreVFP/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:DataDriven_QCD")

events_UL17_sig = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/output_sig125_ForBDT.root:Sig125")
events_UL17_pp = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:pp")
events_UL17_dd = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2017/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:DataDriven_QCD")

events_UL18_sig = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/output_sig125.root:Sig125")
events_UL18_pp = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:pp")
events_UL18_dd = uproot.open("/hpcfs/cms/cmsgpu/zhangzhx/eos/UL2018/MassUL2018_ETSS_PhoID/New_MCpp_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt.root:DataDriven_QCD")

# plot sigmarv in each year
# set figure size
plt.figure(figsize=(15, 8))
# UL16_postVFP
plt.hist(events_UL16_postVFP_pp.arrays("sigmarv", library="np")['sigmarv'], bins=100, range=(0, 0.04), histtype='step', label='UL16_postVFP', density=True);
# ul16_preVFP
plt.hist(events_UL16_preVFP_pp.arrays("sigmarv", library="np")['sigmarv'], bins=100, range=(0, 0.04), histtype='step', label='UL16_preVFP', density=True);
# ul17
plt.hist(events_UL17_pp.arrays("sigmarv", library="np")['sigmarv'], bins=100, range=(0, 0.04), histtype='step', label='UL17', density=True);
# ul18
plt.hist(events_UL18_pp.arrays("sigmarv", library="np")['sigmarv'], bins=100, range=(0, 0.04), histtype='step', label='UL18', density=True);
plt.legend(loc='upper right')
plt.xlabel('sigmarv')
plt.ylabel('Events')
plt.savefig("years_signal_sigmarv.png")