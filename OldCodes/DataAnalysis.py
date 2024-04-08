# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 22:10:52 2024

@author: MA11201
"""
import sys
sys.path.append(r'C:\a_PhD Research\RUL\Codes')
from CSVfileGenOrg import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import os
import pymannkendall as mk

#signals = ['pcd_axis_x', 'pcd_axis_y', 'tp_x', 'tp_y', 'tp_z', 'ta_x', 'ta_y', 'ta_z', 'ts']
signals = ['m1t1_xy', 'm1t2_xy', 'm1t3_xy']
f = ['mean', 'std', 'var', 'rms', 'max_val', 'skewness', 'kurt', 'sf', 'cf', 'mf']    

# =============================================================================
# dir_path = r'C:\a_PhD Research\RUL\Dataset\cnc_data\CSVfiles\M1T1\signals_sensor'
# 
# 
# #creating dataset
# data = create_dataset(dir_path)
# data['force_sensor_x'].shape
# 
# for k, v in data.items():
#     print(k)
# 
# #smoothing
# data_smooth = smoothing(data, 35)
# 
# data_smooth
# =============================================================================



#file paths
path1 = r"C:\a_PhD Research\RUL\Dataset\cnc_data\Processed_data\force\M1T1\kpca_fxy.csv"
path2 = r"C:\a_PhD Research\RUL\Dataset\cnc_data\Processed_data\force\M1T2\kpca_fxy.csv"
path3 = r"C:\a_PhD Research\RUL\Dataset\cnc_data\Processed_data\force\M1T3\kpca_fxy.csv"
path = [path1, path2, path3]


data = {}
for i in range(len(path)):
    dataset = pd.read_csv(path[i])
    dataset = dataset.drop(['Unnamed: 0'], axis=1)
    data[signals[i]] = dataset

data['m1t2_xy']


#plotting
for i in range(len(f)):
  fig, ax = plt.subplots(figsize=(15, 5))
  data['force_sensor_z'][f[i]].plot(ax=ax)
  ax.set_title(f[i])
  
# =============================================================================
# #monotonicity score
# mon_score = {}
# for k, v in data_smooth.items():
#     mon_score1 = monotonicity(data_smooth[k])
#     mon_score[k] =mon_score1
#     
# mon_score['force_sensor_z']   
# =============================================================================

#mk test result
mk_scores = mk_test(data)

for j in range(len(signals)):
    mk_df = []    
    for i in range(10):
        mk_scores_df = pd.DataFrame.from_dict(mk_scores[signals[j]][f[i]], orient='index')
        mk_scores_df.loc[len(mk_scores_df.index)] = [f[i]]
        mk_df.append(mk_scores_df)                
    total = pd.concat([mk_df[0],mk_df[1],mk_df[2],mk_df[3],mk_df[4],mk_df[5],mk_df[6],mk_df[7],mk_df[8], mk_df[9]], axis=1)
    total.to_csv(r'C:\a_PhD Research\RUL\Dataset\cnc_data\mk test results\Force\{m}__mkTest.csv'.format(m=signals[j]))

