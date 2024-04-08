# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:01:51 2024

@author: MA11201
"""

import sys
import numpy as np
import pandas as pd
import os
import h5py

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

def H5_to_CSVFile(data_type, dir_path, no_features):
    
    """
    data_type = signals_machine/signals_sensor
    dir_path = path of the H5 files
    save_path = dir to save the csv files
    no_features = number of features in the dataset (data_type)
    
    """

    item_no = no_features
    sensor_type = data_type
    data_dir = dir_path

    
    for file_name in sorted(os.listdir(data_dir)):
        key = file_name[file_name.find('R'):file_name.find('C')]
        file_path = os.path.join(data_dir, file_name)
        data = create_data(file_path, sensor_type, item_no)
        df = pd.DataFrame(data[0])
        df.to_csv(r'C:\a_PhD Research\RUL\Codes\CNCDataset\M1T3\S_Sensor\file_{}.csv'.format(key))
        print(f'{file_name}\t done !!!')
        
    














