# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

def create_data(file_path, sensor_type, item_num):
    dataset = []
    data = h5py.File(file_path, 'r')
    data1 = data.get(sensor_type)
    data2 = list(data1.items())
    data3 = []
    for i in range(item_num):
        signal = np.array(data2[i][1])
        data3.append(signal)
        
    data4 = np.vstack(data3)
    data5 = data4.T
    dataset.append(data5)
    return dataset

def get_feature(data, column_num):  
        size = data.size
        data = np.array(data.iloc[:, column_num])  
        
       
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
    
#features = ['absolute_mean_value', 'max', 'root_mean_score', 'Root_amplitude', 'skewness', 'Kurtosis_value',
#             'shape_factor', 'pulse_factor', 'skewness_factor', 'crest_factor', 'clearance_factor', 'Kurtosis_factor']

#data_columns = ['pcd_axis_x', 'pcd_axis_y', 'tm', 'tp_x', 'tp_y', 'tp_z', 'ta_x', 'ta_y', 'ta_z', 'ts']
#data_columns1 = ['pcd_axis_x', 'pcd_axis_y', 'tp_x', 'tp_y', 'tp_z', 'ta_x', 'ta_y', 'ta_z', 'ts']
data_columns = ['time','force_sensor_x', 'force_sensor_y', 'force_sensor_z']
data_columns1 = ['force_sensor_x', 'force_sensor_y', 'force_sensor_z']




def dataset_create(data_path, no_of_files, label_path=None):
  data_files = []
  features = np.empty([no_of_files, len(data_columns1), 10])
  # file reading from the directory
  for filename in os.listdir(data_path):
    dataset = pd.read_csv(os.path.join(data_path, filename), names=data_columns)
    dataset = dataset.drop(['time'], axis=1)
    data_files.append(dataset)
  #data_clean = data_denoise(data_files)
  #feature extraction using the get_feature function
  for file in range(len(data_files)):
    for column in range(len(data_columns1)):
      features[file, column, :] = get_feature(data_files[file], column)
  #labels = get_label(label_path)
  return features

def create_dataset(data_path, label_path=None):
    number_of_files = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])
    data = dataset_create(data_path, number_of_files, label_path=None)
    m1t1_df = {}
    for i in range(len(data_columns1)):
      d1 = []
      for j in range(number_of_files): #number of files
        d1.append(data[j][i])
      d2 = np.vstack(d1)
      d3 = MinMaxScaler().fit_transform(d2)
      m1t1_df[data_columns1[i]] = pd.DataFrame(d3, columns= features)
      
    return m1t1_df

def monotonicity(data):
    cols = data.columns
    mon_score = {}
    size = data.shape[0]
    for c in range(len(cols)):
        feature = np.array(data[cols[c]])
        sum = 0    
        for i in range(size-1):
            val = np.sign(feature[i+1]-feature[i])
            sum=sum+val
        res = np.abs(sum/(size-1))
        mon_score[cols[c]] = res*100
    return mon_score

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


# =============================================================================
# def plot_features(data, column_num):  # column_num∈[1, 6]
#     features = data[:, column_num-1, :]
#     x1 = range(0, features.shape[0])
#     plt.figure(num=0, figsize=(12, 5))
#     plt.plot(x1, features[:, 0], '-g', label='Absolute mean')
#     plt.plot(x1, features[:, 1], '--c', label='Max')
#     plt.plot(x1, features[:, 2], '-.k', label='Root mean square')
#     plt.plot(x1, features[:, 3], ':r', label='Square root amplitude')
#     plt.plot(x1, features[:, 4], '-y', label='Skewness')
#     plt.plot(x1, features[:, 5], '-m', label='Kurtosis')
#     plt.plot(x1, features[:, 6], '-og', label='Shape factor')
#     plt.plot(x1, features[:, 7], '-*c', label='Pulse factor')
#     plt.plot(x1, features[:, 8], '-xk', label='Skewness factor')
#     plt.plot(x1, features[:, 9], '-vr', label='Crest factor')
#     plt.plot(x1, features[:, 10], '-sy', label='Clearance factor')
#     plt.plot(x1, features[:, 11], '-+c', label='Kurtosis factor')
#     plt.xlabel('Times of cutting')
#     plt.ylabel('Time domain features')
#     plt.legend(loc=1)
#     plt.show()
#     plt.figure(num=1, figsize=(12, 5))
#     plt.plot(x1, features[:, 12], '-vr', label='FC')
#     plt.plot(x1, features[:, 13], '-k', label='MSF')
#     plt.plot(x1, features[:, 14], '-xk', label='RMSF')
#     plt.plot(x1, features[:, 15], '-og', label='VF')
#     plt.xlabel('Times of cutting')
#     plt.ylabel('Frequency domain features')
#     plt.legend(loc=1)
#     plt.show()
#     plt.figure(num=2, figsize=(12, 5))
#     plt.plot(x1, features[:, 16], '-g', label='Feature 1')
#     plt.plot(x1, features[:, 17], '--c', label='Feature 2')
#     plt.plot(x1, features[:, 18], '-.k', label='Feature 3')
#     plt.plot(x1, features[:, 19], ':r', label='Feature 4')
#     plt.plot(x1, features[:, 20], '-y', label='Feature 5')
#     plt.plot(x1, features[:, 21], '-og', label='Feature 6')
#     plt.plot(x1, features[:, 22], '-*c', label='Feature 7')
#     plt.plot(x1, features[:, 23], '-vr', label='Feature 8')
#     plt.xlabel('Times of cutting')
#     plt.ylabel('Time-frequency domain features')
#     plt.legend(loc=1)
#     plt.show()
# =============================================================================
