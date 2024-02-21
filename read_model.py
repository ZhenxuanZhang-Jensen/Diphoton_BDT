import pickle
file_name_list = ["DiphotonXGboost_afterTune_withAllSigs.pkl","DiphotonXGboost_afterTune_withAllSigs_bayes_v1.pkl","DiphotonXGboost_afterTune_withAllSigs_bayes.pkl","Tue_Nov__8_UL18_DiphotonXGboost_afterTune_new_withAllSigs.pkl","DiphotonXGboost_afterTune_new_withAllSigs.pkl","Wed_Dec__7UL18_DiphotonXGboost_afterTune_new_withAllSigs.pkl","Thu_Dec__8UL18_DiphotonXGboost_afterTune_new_withAllSigs.pkl","Sun_Feb_26_23:43:22_CST_UL17__DiphotonXGboost_afterTune_new_withAllSigs.pkl","Mon_Feb_27_00:02:12_CST_UL17__DiphotonXGboost_afterTune_new_withAllSigs.pkl"]
for file_name in file_name_list:
    print("reading: ", file_name)
    xgb_model_loaded = pickle.load(open(file_name, "rb"))
    # print the max depth of this model
    print("max_depth:", xgb_model_loaded.max_depth)
    # print the estimator of this model
    print(xgb_model_loaded.n_estimators)