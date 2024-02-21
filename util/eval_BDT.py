import uproot
import os
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
import pickle

class Eval_BDT():
    def __init__(self):
        pass
    @staticmethod
    def get_score(df,file_name,input_features):    
        '''
        get the score of events
        param df: the dataframe of events contain all the input features
        param file_name: the file name of xgboost model
        param input_features: the input features of xgboost model
        return score: the score of events
        '''
        # load
        xgb_model_loaded = pickle.load(open(file_name, "rb"))
        # predict
        score = xgb_model_loaded.predict_proba(df[input_features].values)[:,1]
        return score

