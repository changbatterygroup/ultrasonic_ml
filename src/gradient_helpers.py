import sys
# sys.path.append('..') # path to the src directory
# sys.path.append('/Users/xz498/Library/CloudStorage/OneDrive-DrexelUniversity/Chang Lab - Documents/General/Individual/Xinqiao Zhang/data_analysis/ultrasonicTesting/')

import pickleJar as pj
import sqliteUtils as squ
import numpy as np
from numpy import fft
import os.path
import matplotlib.pyplot as plt

import pickle

import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets
from datetime import datetime
import matplotlib.cm as cm
import pandas as pd

from ultrasonic_ml.src import data


def stitch_interp_normalize_pickles(pickleFiles, saveName='combined.pickle', save_directory=None, num_points=2500, overwrite=False):
    """
    Stitch together multiple pickle files into one, interpolating the data to a common time base.
    
    Parameters:
    - pickleFiles: list of str, paths to the pickle files to be combined
    - saveName: str, name of the output combined pickle file
    - save_directory: str, path to the directory where the output file will be saved
    - num_points: int, number of points to interpolate the data to
    - overwrite: bool, whether to overwrite an existing output file
    
    Returns:
    - combined_data: dict, the combined data from all pickle files
    """
    if save_directory: saveName = f'{save_directory}/{saveName}'
    
    combined_data = {'fileName': saveName, 
                     'parameters': {i: {} for i in range(len(pickleFiles))}}
    
    # check if the pickle file already exists. If it does, print an error message and exit early
    if os.path.isfile(combined_data['fileName']):
        if overwrite:  os.remove(combined_data['fileName'])
        else:
            print("sqliteToPickle Warning: pickle file " + combined_data['fileName'] + " already exists. Conversion aborted.")
            return -1
    
    start, end = 0, 100000
    for i, pickleFile in enumerate(pickleFiles):
        data = pj.loadPickle(pickleFile)
        combined_data['parameters'][i]['original_params'] = data['parameters']
        if data[0]['time'][0]>=start: start = data[0]['time'][0]
        if data[0]['time'][-1]<=end: end = data[0]['time'][-1]
        
    combined_data['parameters']['dt'] = (end-start)/num_points
    idx=0
    
    for i, pickleFile in enumerate(pickleFiles):
        data = pj.loadPickle(pickleFile)
        combined_data['parameters'][i]['file_start_idx'] = int(idx)
        for k,v in data.items():
            if str(k).isdigit(): 
                combined_data[idx] = v.copy()
                combined_data[idx]['time'] = np.linspace(start, end, num_points)
                combined_data[idx]['exp_idx'] = i
                for key in v.keys():
                    if 'voltage' in key:
                        combined_data[idx][key] = np.interp(combined_data[idx]['time'], v['time'], v[key])
                        # combined_data[idx][key] -= combined_data[idx][key][-50:].mean() # is there a better way to get the wave on 0?
                        # combined_data[idx][key] /= np.max(np.abs(combined_data[idx][key])) # normalize the voltage to [-1, 1]
                idx+=1
        combined_data['parameters'][i]['file_end_idx'] = int(idx)
    
    # save the dataDict as a pickle. We checked if the file exists earlier, so this operation is safe
    with open(combined_data['fileName'], 'wb') as f:
        pickle.dump(combined_data, f)
    
    return combined_data



def load_and_merge_to_data_df(monitor_file_path_list, settings_file_path_list, data, saveName='merged_df.pickle', save_directory='.', overwrite=False):
    """
    Load and merge monitor and settings data from CSV files, and combine with waveform data.
    
    Parameters:
    - monitor_file_path_list: list of str, paths to the monitor CSV files
    - settings_file_path_list: list of str, paths to the settings CSV files
    - data: dict, the combined data from pickle files
    
    Returns:
    - merged_df: pd.DataFrame, the merged DataFrame containing waveform and controller data
    """
    
    measurements = {k:v for k,v in data.items() if isinstance(k, int)}
    waveform_df = pd.DataFrame(measurements).transpose() 

    controller_monitor_dfs = [pd.read_csv(file, delimiter=';', usecols=[ 'Time', 'Milliseconds',
                                            '1000.1: CH1 Object', '1000.2: CH2 Object', # ###.#: is the firmware output for that channel.
                                            '1001.1: CH1 Sink', '1001.2: CH2 Sink',
                                            '1045.1: HR1 Temp', '1045.2: HR2 Temp',
                                            '1044.1: LR1 Temp', '1044.2: LR2 Temp',
                                            '3000.1: CH1 Target', '3000.2: CH2 Target',
                                            '1044.3: LR3 Temp',
                                        ]) for file in monitor_file_path_list]

    controller_monitor_df = (
        pd.concat(controller_monitor_dfs, ignore_index=True)
        .sort_values(['Time', 'Milliseconds'])
        .drop_duplicates(subset=['Time'], keep='last')
        .reset_index(drop=True)
    )

    controller_monitor_df['Time_str'] = controller_monitor_df['Time']
    controller_monitor_df['Time'] = controller_monitor_df['Time'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p').timestamp())
    controller_monitor_df['Time'] += controller_monitor_df['Milliseconds']/1000
    controller_monitor_df.drop(columns=['Milliseconds'], inplace=True)  

    ## Read in settings data, add ms to time
    controller_settings_dfs = [pd.read_csv(file, delimiter=';', usecols=[ 'Time', 'Milliseconds',
                                            '2010.1: Output Enable', '2010.2: Output Enable', 
                                            '3034.1: Peltier Polarity', '3034.2: Peltier Polarity',
                                            '1200.1: Temperature is Stable','1200.2: Temperature is Stable',
                                            '105.1: Error Number','106.1: Error Inst','107.1: Error Param'
                                        ]) for file in settings_file_path_list]

    controller_settings_df = (
        pd.concat(controller_settings_dfs, ignore_index=True)
        .sort_values(['Time', 'Milliseconds'])
        .drop_duplicates(subset=['Time'], keep='last')
        .reset_index(drop=True)
    )

    controller_settings_df['Time_str'] = controller_settings_df['Time']
    controller_settings_df['Time'] = controller_settings_df['Time'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p').timestamp())
    controller_settings_df['Time'] += controller_settings_df['Milliseconds']/1000
    controller_settings_df.drop(columns=['Milliseconds'], inplace=True)
    
    controller_monitor_df.sort_values('Time', inplace=True)
    controller_settings_df.sort_values('Time', inplace=True)

    ## Merge monitor and settings data
    controller_df = pd.merge_asof(
        left=controller_monitor_df,
        right=controller_settings_df,
        left_on='Time',
        right_on='Time',
        direction='nearest' 
    )

    # unify Time_str columns and remove the originals
    if 'Time_str_x' in controller_df.columns or 'Time_str_y' in controller_df.columns:
        controller_df['Time_str'] = controller_df.get('Time_str_x').fillna(controller_df.get('Time_str_y'))
        controller_df.drop(columns=[c for c in ['Time_str_x', 'Time_str_y'] if c in controller_df.columns], inplace=True)

    # Convert time to float in all dfs
    waveform_df['time_collected'] = waveform_df['time_collected'].astype(float)
    waveform_df = waveform_df.sort_values('time_collected').reset_index(drop=True)
    controller_monitor_df['Time'] = controller_monitor_df['Time'].astype(float)
    controller_settings_df['Time'] = controller_settings_df['Time'].astype(float)
    controller_settings_df['Time'] = controller_settings_df['Time'].astype(float)

    ## Merge waveform and controller data
    merged_df = pd.merge_asof(
        left=waveform_df,
        right=controller_df,
        left_on='time_collected',
        right_on='Time',
        direction='nearest' 
    )

    ## reformat time Time_str to '%m/%d/%y %H:%M:%S'
    merged_df['Time_str'] = pd.to_datetime(merged_df['Time_str']).dt.strftime('%m/%d/%y %H:%M:%S')
    if overwrite: merged_df.to_csv(f'{save_directory}/{saveName}.csv')
    else: merged_df.to_csv(f'{save_directory}/{saveName}.csv', mode='a', header=False)  
    
    return merged_df







