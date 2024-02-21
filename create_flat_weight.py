import uproot
import os
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak

def create_flat_weight(events, title):
    # if title string include "signal", do a mass cut as >122, < 127
    if 'signal' in title:
        events = events[(events['CMS_hgg_mass'] > 122) & (events['CMS_hgg_mass'] < 127)]
        other_events = events[(events['CMS_hgg_mass'] < 122) | (events['CMS_hgg_mass'] > 127)]
        other_events['flat_mass_weight'] = 1
    # create a numpy histogram for cms_hgg_mass
    hist, bins = np.histogram(events['CMS_hgg_mass'], bins=30)
    # calculate the mean of the hist
    mean = np.mean(hist)
    # calculate the weight
    weight = mean / hist
    # create a list of range based on bins numbers
    range_list = []
    for i in range(len(bins)-1):
        range_list.append([bins[i], bins[i+1]])
    # cut the events into 10 parts based on the range_list, and give the weight to each part
    for i in range(len(range_list)):
        events_cut = events[(events['CMS_hgg_mass'] >= range_list[i][0]) & (events['CMS_hgg_mass'] < range_list[i][1])]
        # if is the last one, all set to larger or euqual
        if i == len(range_list)-1:
            events_cut = events[(events['CMS_hgg_mass'] >= range_list[i][0]) & (events['CMS_hgg_mass'] <= range_list[i][1])]
        events_cut['flat_mass_weight'] = weight[i]
        if i == 0:
            events_cut_total = events_cut
        else:
            events_cut_total = ak.concatenate([events_cut_total, events_cut])
    # if signal
    if 'signal' in title:
        events_cut_total = ak.concatenate([events_cut_total, other_events])

    # give the events with cms_hgg_mass > 110 a weight of 1, otherwise 0
    weights = events_cut_total['flat_mass_weight'].to_numpy()
    # plot CMS_hgg_mass without weights
    plt.hist(events_cut_total['CMS_hgg_mass'], bins=100)
    plt.title(title + 'CMS_hgg_mass without weights')
    plt.savefig(title + 'CMS_hgg_mass_without_weights.png')     
    plt.close()
    # plot CMS_hgg_mass with weights
    plt.hist(events_cut_total['CMS_hgg_mass'], bins=100, weights = weights)
    plt.title(title + 'CMS_hgg_mass with weights')
    plt.savefig(title + 'CMS_hgg_mass_with_weights.png')
    plt.close()
    return events_cut_total