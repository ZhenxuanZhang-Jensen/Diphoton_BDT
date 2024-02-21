import uproot
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
import mplhep as hep
from scipy.optimize import curve_fit

class MassResolution():
    def __init__(self, file_path, tree_name, branches):
        self.file_path = file_path
        self.tree_name = tree_name
        self.branches = branches
        self.data = None
        self.mass = None
        self.mass_resolution = None
        self.mass_resolution_mean = None
        self.mass_resolution_std = None
        self.m
    @staticmethod
    def get_data(self):
        self.data = uproot.open(self.file_path)[self.tree_name].arrays(self.branches)
    @staticmethod
    def get_mass(self):
        self.mass = np.sqrt(2*self.data['pt1']*self.data['pt2']*(np.cosh(self.data['eta1']-self.data['eta2'])-np.cos(self.data['phi1']-self.data['phi2'])))
    @staticmethod
    def gaussian(x, A, mu, sigma):
        ''' Define a Gaussian function'''
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    @staticmethod
    def fit_2guassion(df_mass, df_score, df_weight, list_cut_value=[0,0.1,0.2,0.3], plot_bin=80):
        '''
        Fit the mass distribution with two Gaussians
        param df_mass: the dataframe of signal mass
        param df_score: the dataframe of signal score
        param df_weight: the dataframe of signal weight
        param list_cut_value: the list of cut value
        param plot_bin: the bin of plot
        '''
        if (len(list_cut_value != 4)):
            raise ValueError("The length of list_cut_value must be 4")
        mass_hist, bin_edges, patch = plt.hist(
            df_mass[df_score > list_cut_value[3]],
            weights=df_weight[df_score > list_cut_value[3]],
            bins=plot_bin, range=(100, 150), histtype="step", label="old strategy", color="red"
        )
        # print whole events
        print(f"debug whole events {sum(mass_hist)}")

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        num_gaussians = 2
        initial_guess = [1000, 125, 1, 500, 135, 1]  # Initial guesses for the parameters of the two Gaussians
        params, covariance = curve_fit(
            lambda x, *params: sum(MassResolution.gaussian(x, params[i], params[i + 1], params[i + 2]) for i in range(0, len(params), 3)),
            bin_centers, mass_hist, p0=initial_guess
        )
        x_fit = np.linspace(100, 150, 1000)  # Create x values for the fit curve
        y_fit = sum(MassResolution.gaussian(x_fit, params[i], params[i + 1], params[i + 2]) for i in range(0, len(params), 3))
        peak_position = x_fit[np.argmax(y_fit)]

        cumulative_sum = np.cumsum(y_fit) / np.sum(y_fit)

        # 初始化左右边界和累积面积
        left_boundary = peak_position
        right_boundary = peak_position
        cumulative_area = 0.0
        target_area = 0.683  # 目标累积面积

        while cumulative_area < target_area:
            # 同时向左右两边扫描，步长逐渐增加
            left_boundary -= 0.0001
            right_boundary += 0.0001

            # 计算当前左右边界内的累积面积
            left_idx = np.where(x_fit <= left_boundary)[0][-1]
            right_idx = np.where(x_fit >= right_boundary)[0][0]
            cumulative_area = np.sum(y_fit[left_idx:right_idx]) / np.sum(y_fit)

        window_width = right_boundary - left_boundary

        effective_sigma_value = window_width / 2
        return effective_sigma_value, peak_position, mass_hist, bin_centers, x_fit, y_fit 
    @staticmethod
    def fit_3gaussians(df_mass, df_score, df_weight, cut_value, plot_bin=80):
        ''' Fit the mass distribution with three Gaussians'''
        if (len(cut_value) != 2):
            raise ValueError("The length of list_cut_value must be 2")
        mass_hist, bin_edges, patch = plt.hist(
            df_mass[(df_score > cut_value[0]) & (df_score < cut_value[1])],
            weights=df_weight[(df_score > cut_value[0]) & (df_score < cut_value[1])],
            bins=plot_bin, range=(100, 150), histtype="step", label="old strategy", color="red"
        )
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        num_gaussians = 3
        initial_guess = [100, 120, 2, 500, 130, 2, 0, 140, 2]  # Initial guesses for the parameters of the three Gaussians
        params, covariance = curve_fit(
        lambda x, *params: sum(MassResolution.gaussian(x, params[i], params[i + 1], params[i + 2]) for i in range(0, len(params), 3)),
        bin_centers, mass_hist, p0=initial_guess, maxfev=50000)

        x_fit = np.linspace(100, 150, 10000)  # Create x values for the fit curve
        y_fit = sum(MassResolution.gaussian(x_fit, params[i], params[i + 1], params[i + 2]) for i in range(0, len(params), 3))
        peak_position = x_fit[np.argmax(y_fit)]

        cumulative_sum = np.cumsum(y_fit) / np.sum(y_fit)

        # 初始化左右边界和累积面积
        left_boundary = peak_position
        right_boundary = peak_position
        cumulative_area = 0.0
        target_area = 0.683  # 目标累积面积

        while cumulative_area < target_area:
            # 同时向左右两边扫描，步长逐渐增加
            left_boundary -= 0.0001
            right_boundary += 0.0001

            # 计算当前左右边界内的累积面积
            left_idx = np.where(x_fit <= left_boundary)[0][-1]
            right_idx = np.where(x_fit >= right_boundary)[0][0]
            cumulative_area = np.sum(y_fit[left_idx:right_idx]) / np.sum(y_fit)

        window_width = right_boundary - left_boundary

        effective_sigma_value = window_width / 2

        return effective_sigma_value, peak_position, mass_hist, bin_centers, x_fit, y_fit
    @staticmethod
    def fit_4gaussians(df_mass, df_score, df_weight, cut_value,  plot_bin=80):
        '''fit the mass distribution with four Gaussians'''
        
        if (len(cut_value) != 2):
            raise ValueError("The length of list_cut_value must be 2")
        mass_hist, bin_edges, patch = plt.hist(
            df_mass[(df_score > cut_value[0]) & (df_score < cut_value[1])],
            weights=df_weight[(df_score > cut_value[0]) & (df_score < cut_value[1])],
            bins=plot_bin, range=(100, 150), histtype="step", label="old strategy", color="red"
        )
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        num_gaussians = 4  # Change the number of Gaussians to 4
        
        initial_guess = [
            1000, 120, 2, 500, 130, 2, 200, 140, 2, 300, 145, 2
        ]  # Initial guesses for the parameters of the four Gaussians
        
        params, covariance = curve_fit(
            lambda x, *params: sum(MassResolution.gaussian(x, params[i], params[i + 1], params[i + 2]) for i in range(0, len(params), 3)),
            bin_centers, mass_hist, p0=initial_guess, maxfev=500000
        )

        x_fit = np.linspace(100, 150, 10000)  # Create x values for the fit curve
        y_fit = sum(MassResolution.gaussian(x_fit, params[i], params[i + 1], params[i + 2]) for i in range(0, len(params), 3))
        peak_position = x_fit[np.argmax(y_fit)]

        cumulative_sum = np.cumsum(y_fit) / np.sum(y_fit)

        left_boundary = peak_position
        right_boundary = peak_position
        cumulative_area = 0.0
        target_area = 0.683

        while cumulative_area < target_area:
            left_boundary -= 0.0001
            right_boundary += 0.0001

            left_idx = np.where(x_fit <= left_boundary)[0][-1]
            right_idx = np.where(x_fit >= right_boundary)[0][0]
            cumulative_area = np.sum(y_fit[left_idx:right_idx]) / np.sum(y_fit)

        window_width = right_boundary - left_boundary

        effective_sigma_value = window_width / 2
        
        return effective_sigma_value, peak_position, mass_hist, bin_centers, x_fit, y_fit
    @staticmethod
    def get_boundary_with_same_event(df_weight_with_bo, df_weight_need_bo,df_score_with_bo,df_score_need_bo, bo_list):
        # if input df_weight_with_bo, df_weight_need_bo, df_score_with_bo, df_score_need_bo are not dataframe, turn it to dataframe
        if type(df_weight_with_bo) != pd.core.series.Series:
            df_weight_with_bo = pd.Series(df_weight_with_bo)
            
        # bo3 events
        events_bo3 = df_weight_with_bo[(df_score_with_bo>bo_list[3]) & (df_score_with_bo<1)]
        num_events_bo3 = sum(events_bo3)
        # bo2 events
        events_bo2 = df_weight_with_bo[(df_score_with_bo>bo_list[2]) & (df_score_with_bo<bo_list[3])]
        num_events_bo2 = sum(events_bo2)
        # bo1 events
        events_bo1 = df_weight_with_bo[(df_score_with_bo>bo_list[1]) & (df_score_with_bo<bo_list[2])]
        num_events_bo1 = sum(events_bo1)
        # bo0 events
        events_bo0 = df_weight_with_bo[(df_score_with_bo>bo_list[0]) & (df_score_with_bo<bo_list[1])]
        num_events_bo0 = sum(events_bo0)
        
        # change the cut value to have the same events as the default
        # get the cut value for bo3
        tmp_cut_value = 999
        for i in range(9000,10000):
            i = i / 10000
            events_need_bo3 = df_weight_need_bo[(df_score_need_bo>i) & (df_score_need_bo<1)]
            num_events_need_bo3 = sum(events_need_bo3)
            if abs(num_events_need_bo3 - num_events_bo3) <1:
                if (abs(num_events_need_bo3 - num_events_bo3) < tmp_cut_value):
                    tmp_cut_value = abs(num_events_need_bo3 - num_events_bo3)
                    cut_value_3 = i
        print("cut value", cut_value_3)
        print("events diff %:",tmp_cut_value*100/num_events_bo3)
        # get the cut value for bo2
        tmp_cut_value = 999
        for i in range(800, int(cut_value_3*1000)):
            i = i / 1000
            events_need_bo2 = df_weight_need_bo[(df_score_need_bo>i) & (df_score_need_bo<cut_value_3)]
            num_events_need_bo2 = sum(events_need_bo2)
            if abs(num_events_need_bo2 - num_events_bo2) <1:
                if (abs(num_events_need_bo2 - num_events_bo2) < tmp_cut_value):
                    tmp_cut_value = abs(num_events_need_bo2 - num_events_bo2)
                    cut_value_2 = i
        print("cut value", cut_value_2)
        print("events diff %:",tmp_cut_value*100/num_events_bo2)    
        # get the cut value for bo1
        tmp_cut_value = 999
        for i in range(500, int(cut_value_2*1000)):
            i = i / 1000
            events_need_bo1 = df_weight_need_bo[(df_score_need_bo>i) & (df_score_need_bo<cut_value_2)]
            num_events_need_bo1 = sum(events_need_bo1)
            if abs(num_events_need_bo1 - num_events_bo1) <3:
                if (abs(num_events_need_bo1 - num_events_bo1) < tmp_cut_value):
                    tmp_cut_value = abs(num_events_need_bo1 - num_events_bo1)
                    cut_value_1 = i
        print("cut value", cut_value_1)
        print("events diff %:",tmp_cut_value*100/num_events_bo1)
        # get the cut value for bo0
        tmp_cut_value = 999
        for i in range(300, int(cut_value_1*1000)):
            i = i / 1000
            events_need_bo0 = df_weight_need_bo[(df_score_need_bo>i) & (df_score_need_bo<cut_value_1)]
            num_events_need_bo0 = sum(events_need_bo0)
            if abs(num_events_need_bo0 - num_events_bo0) < 2:
                if (abs(num_events_need_bo0 - num_events_bo0) < tmp_cut_value):
                    tmp_cut_value = abs(num_events_need_bo0 - num_events_bo0)
                    cut_value_0 = i
        print("cut value", cut_value_0)
        print("events diff %:",tmp_cut_value*100/num_events_bo0)
        cut_value_list = [cut_value_0, cut_value_1, cut_value_2, cut_value_3]
        return cut_value_list
    
    # @staticmethod
    # def 