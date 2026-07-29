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
from scipy.optimize import curve_fit

### data
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
                        combined_data[idx][key] -= combined_data[idx][key][-50:].mean() # is there a better way to get the wave on 0?
                        combined_data[idx][key] /= np.max(np.abs(combined_data[idx][key])) # normalize the voltage to [-1, 1]
                idx+=1
        combined_data['parameters'][i]['file_end_idx'] = int(idx)
    
    # save the dataDict as a pickle. We checked if the file exists earlier, so this operation is safe
    with open(combined_data['fileName'], 'wb') as f:
        pickle.dump(combined_data, f)
    
    return combined_data


def load_and_merge_to_data_df(data, monitor_file_path_list, settings_file_path_list, saveName='merged_df.pickle', save_directory='.', overwrite=False):
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
    
    ##get rid of nans
    merged_df = merged_df.dropna()
    
    ## save merged dataframe to csv
    if overwrite: merged_df.to_csv(f'{save_directory}/{saveName}.csv')
    else: merged_df.to_csv(f'{save_directory}/{saveName}.csv', mode='a', header=False)  
    
    return merged_df


### calculations

def calculate_mean_diff(merged_df):
    '''Calculate the mean and difference of temperatures from the merged dataframe.'''
    merged_df['index'] = merged_df.index
    merged_df['mean_T (°C)'] = merged_df[['1000.1: CH1 Object', '1000.2: CH2 Object']].mean(axis=1)
    merged_df['diff_T (°C)'] = merged_df['1000.1: CH1 Object'] - merged_df['1000.2: CH2 Object']
    return merged_df


def calculate_waveform_features(merged_df):
    '''Calculate waveform features from the merged dataframe.'''
    for key_ in ['_forward', '_reverse']:    
        merged_df['amplitude'+key_+' (mV)'] = list(map(pj.maxMinusMin, merged_df['voltage_transmission'+key_]))
        merged_df['Hilbert_ToF'+key_+' (ns)'] = list(map(pj.envelopeThresholdTOF, merged_df['voltage_transmission'+key_], merged_df['time'], [0.15,] * len(merged_df)))
        merged_df['Hilbert_noise_Tof'+key_+' (ns)'] = list(map(pj.firstIndexAboveNoise, merged_df['voltage_transmission'+key_]))
        merged_df['max'+key_+' (mV)'] = list(map(lambda x: max(abs(x)), merged_df['voltage_transmission'+key_]))
    return merged_df

# TODO: test
def detrend_linear(df, keys=None):
    """
    Detrend amplitude and ToF data using polynomial and logarithmic fits.
    
    Parameters:
    - df: pandas.DataFrame, the filtered dataframe to detrend
    - amplitude_keys: list of str, column names for amplitude data to detrend (default: forward and reverse)    
    Returns:
    - df: pandas.DataFrame, the detrended dataframe (modifies in place)
    """
    if amplitude_keys is None:  amplitude_keys = ['amplitude_forward (mV)', 'amplitude_reverse (mV)']
    
    # Detrend amplitude with linear fit
    for k in amplitude_keys:
        a, b = np.polyfit(df['time_collected'], df[k], 1)
        fit = a * df['time_collected'] + b
        df[k] -= fit

# TODO: test
def detrend_log(df, amplitude_keys=None, tof_keys=None):
    """
    Detrend amplitude and ToF data using polynomial and logarithmic fits.
    
    Parameters:
    - df: pandas.DataFrame, the filtered dataframe to detrend
    - tof_keys: list of str, column names for ToF data to detrend (default: Hilbert and Simple ToF)
    
    Returns:
    - df: pandas.DataFrame, the detrended dataframe (modifies in place)
    """
    if tof_keys is None: tof_keys = ['Hilbert_ToF_forward (ns)', 'Hilbert_ToF_reverse (ns)']
    # Detrend ToF with logarithmic fit
    def log_func(t_rel, a, b): 
        return a + b * np.log(t_rel)
    
    t = df['time_collected'].to_numpy(dtype=float)
    t0 = t.min()
    t_rel = t - t0 + 1.0  # strictly > 0 for log
    
    for k in tof_keys:
        y = df[k].to_numpy(dtype=float)
        popt, _ = curve_fit(log_func, t_rel, y, maxfev=20000)
        fit = log_func(t_rel, *popt)
        df[k] -= fit
    
    return df



### plotting functions
# TODO
def plot_detrended_amplitude(df,):
    """
    Plot the detrended amplitude of the merged dataframe.
    
    Parameters:
    - merged_df: pd.DataFrame, the merged DataFrame containing waveform and controller data
    """
    # before/ater detrending amplitude
    k = 'amplitude_forward_normalized'
    a,b = np.polyfit(df['time_collected'], df[k], 1)
    fit = a * df['time_collected'] + b
    t_rel = df['time_collected'] - df['time_collected'].iloc[0]

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
        
    ax1.plot(t_rel, df[k], label='data', alpha=0.8, color='blue', lw=1)
    ax1.plot(t_rel, fit, label=f'{a:.3f} + {b:.3f} * t', lw=1, linestyle='--', color='orange')
    ax1.set_xlabel('Time since start (s)')
    ax1.set_ylabel('Amplitude (ns)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2.plot(t_rel, df[k] - fit, label='detrended', alpha=0.8, color='green')
    ax2.set_ylabel('Detrended Amplitude (ns)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(k)
    plt.tight_layout()
    plt.show()

# TODO
def plot_detrended_tof(merged_df):
    """
    Plot the detrended ToF of the merged dataframe.
    
    Parameters:
    - merged_df: pd.DataFrame, the merged DataFrame containing waveform and controller data
    """
    df = merged_df.iloc[1500:].copy()
    k = 'Hilbert_ToF_forward (ns)'

    # use relative time to keep log-fit numerically stable
    t = df['time_collected'].to_numpy(dtype=float)
    y = df[k].to_numpy(dtype=float)
    t0 = t.min()
    t_rel = t - t0 + 1.0  # strictly > 0 for log

    def log_func(t_rel, a, b): return a + b * np.log(t_rel)

    popt, _ = curve_fit(log_func, t_rel, y, maxfev=20000)
    fit = log_func(t_rel, *popt)

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(t_rel, y, label='data', alpha=0.8, color='blue', lw=1)
    ax1.plot(t_rel, fit, label=f'{popt[0]:.3f} + {popt[1]:.3f} * ln(t)', lw=1, linestyle='--', color='orange')
    ax1.set_xlabel('Time since start (s)')
    ax1.set_ylabel('ToF (ns)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2.plot(t_rel, y - fit, label='detrended', alpha=0.8, color='green')
    ax2.set_ylabel('Detrended ToF (ns)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(k)
    plt.tight_layout()
    plt.show()

def waveform_interact_decorator(func):
    def wrapper(*args, **kwargs):
        merged_df = kwargs.get('merged_df', None)
        if merged_df is None: raise ValueError("merged_df is required")
        max_ = merged_df['max_forward (mV)'].max()
        
        slider = widgets.IntSlider(value=0,
                                    min=0, max=len(merged_df)-1,
                                    step=1,
                                    description='measurement:',
                                    continuous_update=False
                                )
        function = lambda i: func(i, merged_df, max_)
        interact(function, i=slider)
    return wrapper

@waveform_interact_decorator
def plot_waveform(i, merged_df=None, max_=None):
    fig,ax = plt.subplots(figsize=(8, 4))
            
    ax.plot(merged_df['time'][i], merged_df['voltage_transmission_forward'][i], lw=1, label='forward')
    ax.vlines(merged_df['Hilbert_ToF_forward (ns)'][i], -max_, max_, color='blue', lw=1, linestyles='--', label='Hilbert ToF forward')
    
    ax.plot(merged_df['time'][i], merged_df['voltage_transmission_reverse'][i], lw=1, label='reverse')
    ax.vlines(merged_df['Hilbert_ToF_reverse (ns)'][i], -max_, max_, color='green', lw=1, linestyles='--', label='Hilbert ToF reverse')
    
    
    ax.set_title(f'Measurement {i} at ({merged_df["Time_str"][i]})')
    ax.set_ylim(-max_, max_)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Voltage Normalized')
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    fig.tight_layout()
  

def experimental_data_interact_decorator(func):
    def wrapper(*args, **kwargs):
        df = kwargs.get('df', None)
        if df is None:  raise ValueError("merged_df is required")
        idx_label_type = kwargs.get('idx_label_type', '')
        x_scaling = kwargs.get('x_scaling', 'collection_time')
        temp_values = df[['1000.1: CH1 Object', '1000.2: CH2 Object', '1044.3: LR3 Temp']].values
        ylims_ = (temp_values.min() - 5, temp_values.max() + 5)
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(df)-1,
            step=1,
            description='measurement:',
            continuous_update=False,
            layout=widgets.Layout(width='800px')  # or '100%'
        )
        selected_keys = [col for col in df.columns if col.startswith(('Hilbert', 'amplitude'))]
        selected_keys.sort()
        keys = widgets.SelectMultiple(
            options=selected_keys,
            value=tuple([]),
            description='Variable:',
            disabled=False,
        )
        function = lambda i, keys_: func(i=i, keys_=list(keys_), df=df, ylims_=ylims_, idx_label_type=idx_label_type, x_scaling=x_scaling)
        interact(function, i=slider, keys_=keys)
    return wrapper

@experimental_data_interact_decorator
def plot_experimental_data(i=0, keys_=(), df=None, idx_label_type='', x_scaling='collection_time',ylims_=(0, 60),
                           ):
    """Plot temperature data with optional secondary variables.

    Parameters
    ----------
    i : int, optional
        Index of the measurement to highlight.
    df : pd.DataFrame, optional
        The DataFrame containing the temperature data.
    keys_ : tuple or list of str, optional
        Additional columns to plot on a secondary y-axis.
    idx_label_type : str, optional
        Type of x-axis labels to use ('time', 'dT', 'T').
    x_scaling : str, optional
        Type of x-axis scaling to use ('collection_time', 'index').
    """
    xticks_labels = df['index'].astype(str) + ': ' + df['Time_str']
    if x_scaling=='collection_time':
        xticks = df['time_collected'].values
    else:
        xticks = df['index'].values

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # plot temperature data vs ΔT
    ax1.plot(xticks, df['mean_T (°C)'], label='mean', marker='o', linestyle='-', linewidth=0.01, markersize=2)
    lower_bound = df['1000.1: CH1 Object']
    ax1.plot(xticks, lower_bound, label='transmitter side', linestyle='--', linewidth=0.01, marker='o', markersize=2)
    upper_bound = df['1000.2: CH2 Object']
    ax1.plot(xticks, df['1000.2: CH2 Object'], label='receiver side', linestyle='--', linewidth=0.01, marker='o', markersize=2)
    ax1.fill_between(xticks, df['1000.1: CH1 Object'], df['1000.2: CH2 Object'], alpha=0.25)
    ax1.axvline(xticks[i], color='red', linestyle='--', linewidth=0.8, alpha=0.4)
    ax1.plot(xticks, df['1044.3: LR3 Temp'], label='chamber', marker='o', linestyle='-', linewidth=0.01, markersize=1)
    ax1.plot(xticks, df['1001.1: CH1 Sink'], label='trans sink', marker='o', linestyle='-', linewidth=0.01, markersize=1)
    ax1.plot(xticks, df['1001.2: CH2 Sink'], label='rec. sink', marker='o', linestyle='-', linewidth=0.01, markersize=1)

    # plot secondary keys
    if len(keys_) > 0:
        ax2 = ax1.twinx()  # second y-axis sharing x-axis
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        start_idx = 6
        for idx, key_ in enumerate(keys_):
            ax2.scatter(xticks, df[key_], label=key_, s=10, color=colors[start_idx + idx])
        ax2.set_ylabel('\n'.join([f'{key_}' for key_ in keys_]))
        # ax2.set_ylim(df[keys_].values.mean() - df[keys_].values.std()*5, 
        #              df[keys_].values.mean() + df[keys_].values.std()*5)
        
    # formatting
    ax1.set_title('Temperature gradient vs collection time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_ylim(*ylims_)
    if idx_label_type == 'time':
        ax1.set_xticks(ticks=xticks[::max(1, len(xticks_labels)//10)] , 
                       labels=xticks_labels[::max(1, len(xticks_labels)//10)], 
                       rotation=45, ha='right')
        ax1.set_xlabel('Collection idx: Time')
        
    elif idx_label_type == 'dT':
        x = (df['3000.1: CH1 Target'] - df['3000.2: CH2 Target']).values
        t_change_idx = np.flatnonzero(np.diff(x, prepend=x[0]) != 0).tolist()
        diffs = np.diff(t_change_idx, prepend=t_change_idx[0])
        for j in range(len(t_change_idx) - 1, 0, -1):
            if diffs[j] < 20: t_change_idx.pop(j)
        t_change_idx = np.asarray(t_change_idx, dtype=int)
        
        # for idx in t_change_idx: # plot temp changes
        ax1.set_xticks(ticks=xticks[t_change_idx][::2], 
                       labels=x[t_change_idx][::2], rotation=45, ha='right')
        ax1.set_xlabel('Target ΔT (°C)')
        for idx in t_change_idx: ax1.axvline(xticks[idx], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
        
    elif idx_label_type == 'T':
        x = df['3000.1: CH1 Target'].values
        t_change_idx = np.flatnonzero(np.diff(x, prepend=x[0]) != 0).tolist()
        diffs = np.diff(t_change_idx, prepend=t_change_idx[0])
        for j in range(len(t_change_idx) - 1, 0, -1):
            if diffs[j] < 30: t_change_idx.pop(j)
        t_change_idx = np.asarray(t_change_idx, dtype=int)
        ax1.set_xticks(ticks=xticks[t_change_idx], 
                       labels=x[t_change_idx], rotation=45, ha='right')
        ax1.set_xlabel('Target Temperature (°C)')
        for idx in t_change_idx: ax1.axvline(xticks[idx], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
        
    fig.legend(bbox_to_anchor=(0.5, 0.03), loc='upper center', ncols=min(max(len(keys_)//2+2,1), 3))
    fig.tight_layout()

    # # highlight specific indices based on conditions
    # if highlight_where:
    #     for key, value in highlight_where.items():
    #         highlight_indices = df.index[ value(df[key]) ].tolist()
    #         if highlight_indices:
    #             highlight_indices = np.asarray(sorted(highlight_indices), dtype=int)
    #             blocks = np.split(highlight_indices, np.where(np.diff(highlight_indices) != 1)[0] + 1)
    #             for block in blocks:
    #                 ax1.axvspan(xticks[block.min()], xticks[block.max()], color='gray', alpha=0.15)
    