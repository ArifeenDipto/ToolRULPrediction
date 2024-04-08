# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:38:39 2024

@author: MA11201
"""

import os
import sys
sys.path.append(r'C:\a_PhD Research\RUL\Codes')
from CSVfileGenOrg import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')
from sys import stdout
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

#signals = ['pcd_axis_x', 'pcd_axis_y', 'tp_x', 'tp_y', 'tp_z', 'ta_x', 'ta_y', 'ta_z', 'ts']
signals = ['time','force_sensor_x', 'force_sensor_y', 'force_sensor_z']
f = ['mean', 'std', 'var', 'rms', 'max_val', 'skewness', 'kurt', 'sf', 'cf', 'mf']
    
dir_path = r'C:\a_PhD Research\RUL\Dataset\cnc_data\CSVfiles\M1T3\signals_sensor' #tool selection
label_path = r'C:\a_PhD Research\RUL\Dataset\cnc_data\filelist.csv'

#creating dataset
data = create_dataset(dir_path)

data_smooth = smoothing(data, 35)

select_data = data_smooth['force_sensor_z'] #signal selection

#select_data.shape
#select_data.head(3)

#select_data1 = select_data.drop(['max_val'], axis=1) #irrelevant feature drop
#select_data1.shape
#M1T1 = 0:609
#M1T2 = 609:1218
#M1T3 = 1218:1856

label = pd.read_csv(label_path)
#label.iloc[609:1219, :]['wear'].tail(10) #label index selection
wear_val = np.array(label.iloc[1218:1856]['wear'])#label index selection
select_data['label'] = wear_val
select_data.to_csv(r"C:\a_PhD Research\RUL\Dataset\cnc_data\Processed_data\force\M1T3\force_sensor_z.csv")


