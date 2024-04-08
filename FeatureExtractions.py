# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:52:06 2024

@author: MA11201
"""

import numpy as np
import pandas as pd
import os
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, kurtosis, skew
from scipy.signal import savgol_filter
import pymannkendall as mk



def get_feature(data, column_num):  
        size = data.size
        data = np.array(data.iloc[:, column_num])  
        
        #Time domain features
        mean = np.mean(data)
        std = np.std(data)
        var = np.var(data)
        rms = np.sqrt(np.sum(np.square(data)) / size)
        max_val = np.max(np.abs(data))
        skewness = skew(data)
        kurt = kurtosis(data)
        sf = rms / mean
        cf = max_val/rms
        mf = max_val/var
        
        f = [mean, std, var, rms, max_val, skewness, kurt,sf, cf, mf]

        return f

features = ['mean', 'std', 'var', 'rms', 'max_val', 'skewness', 'kurt','sf', 'cf', 'mf']    

def dataset_create(data_path, no_of_files, dc, dc1, data_type, label_path=None):
  data_columns =dc
  data_columns1 = dc1
  data_files = []
  features = np.empty([no_of_files, len(data_columns1), 10])
  # file reading from the directory
  for filename in os.listdir(data_path):
    dataset = pd.read_csv(os.path.join(data_path, filename), names=data_columns)
    
    if data_type == 'signals_sensor':
        dataset = dataset.drop(['Unnamed: 0', 'time'], axis=1)
    else:
        dataset = dataset.drop(['Unnamed: 0', 'tm'], axis=1)
        
    data_files.append(dataset)
  #feature extraction using the get_feature function
  for file in range(len(data_files)):
    for column in range(len(data_columns1)):
      features[file, column, :] = get_feature(data_files[file], column)
  #labels = get_label(label_path)
  return features

def create_dataset(data_path, data_type, label_path=None):
    
    if data_type == 'signals_sensor':
        data_columns = ['Unnamed: 0','force_sensor_x', 'force_sensor_y', 'force_sensor_z', 'time']
        data_columns1 = ['force_sensor_x', 'force_sensor_y', 'force_sensor_z']
        
    else:
        data_columns = ['Unnamed: 0', 'pcd_axis_x', 'pcd_axis_y', 'tm', 'tp_x', 'tp_y', 'tp_z', 'ta_x', 'ta_y', 'ta_z', 'ts']
        data_columns1 = ['pcd_axis_x', 'pcd_axis_y', 'tp_x', 'tp_y', 'tp_z', 'ta_x', 'ta_y', 'ta_z', 'ts']
        
    
    number_of_files = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])
    data = dataset_create(data_path, number_of_files, data_columns, data_columns1, data_type, label_path=None)
    m1t1_df = {}
    for i in range(len(data_columns1)):
      d1 = []
      for j in range(number_of_files): #number of files
        d1.append(data[j][i])
      d2 = np.vstack(d1)
      d3 = MinMaxScaler().fit_transform(d2)
      m1t1_df[data_columns1[i]] = pd.DataFrame(d3, columns= features)
      
    return m1t1_df

def smoothing(data, wind_len):
    data_dict = {}
    for k, v in data.items():
        data1 = data[k]
        cols = data1.columns
        df = pd.DataFrame()
        for c in range(data1.shape[1]):
            smooth_data = savgol_filter(np.array(data1[cols[c]]), wind_len, 2)
            df[cols[c]] = smooth_data

        data_dict[k] = df
    return data_dict

#MK test
mk_features = ['trend', 'h', 'p', 'z', 'Tau', 's', 'Var_s', 'slope', 'intrcept']
def mk_test(data):
    mk_score = {}
    for k, v in data.items():
        data1 = data[k]
        cols = data1.columns
        result1 = {}
        for c in range(data1.shape[1]):
            mk_test_res = mk.original_test(np.array(data1[cols[c]]))
            result2 = {}
            for i in range(9):
                result2[mk_features[i]] = mk_test_res[i]
            result1[cols[c]] = result2
        mk_score[k] = result1
    return mk_score

