
import glob
import json
import numpy as np
import h5py
from scipy.signal import iirfilter, zpk2sos, sosfilt, sosfreqz
# from scipy.signal.windows import hann

from pydvs.h5_reader import das_reader as reader
from pydvs.preprocess_utils import *
from pydvs.plot_utils import *


# Load Seismic Data:
# ID: [event_time, file_name, start_time, end_time, start_channel, end_channel]
event_times = {
    0: ["2020-07-27 08:17:34.6", "rhone1khz_UTC_20200727_081708.575.h5"], # Category 1, Receiver ALH
    5: ["2020-07-27 19:43:30.5", "rhone1khz_UTC_20200727_194308.575.h5", 20, 25, 550, 720], # Category 1, Receiver ALH (plotted in paper)
    11: ["2020-07-27 19:43:01.4", "rhone1khz_UTC_20200727_194238.575.h5"], # Category 1, Receiver ALH
    35: ["2020-07-27 03:03:20.2", "rhone1khz_UTC_20200727_030308.575.h5"], # Category 1, Receiver ALH (dominant frequency at 100 Hz)
    83: ["2020-07-27 01:03:00.2", "rhone1khz_UTC_20200727_010238.575.h5"], # Category 1, Receiver ALH

    20: ["2020-07-27 00:21:46.3", "rhone1khz_UTC_20200727_002138.575.h5", 5, 10, 580, 700], # Category 2, Receiver ALH (plotted in paper)
    24: ["2020-07-27 05:21:48.4", "rhone1khz_UTC_20200727_052138.575.h5"], # Category 2, Receiver ALH (dominant frequency at 100 Hz)
    36: ["2020-07-27 20:47:34.8", "rhone1khz_UTC_20200727_204708.575.h5"], # Category 2, Receiver ALH
    52: ["2020-07-27 20:00:30.4", "rhone1khz_UTC_20200727_200008.575.h5"], # Category 2, Receiver ALH
    67: ["2020-07-27 23:17:54.0", "rhone1khz_UTC_20200727_231738.575.h5"], # Category 2, Receiver ALH
    107: ["2020-07-27 01:25:19.6", "rhone1khz_UTC_20200727_012508.575.h5"], # Category 2, Receiver ALH

    82: ["2020-07-27 05:04:55.8", "rhone1khz_UTC_20200727_050438.575.h5", 15, 20, 560, 770], # Category 3, Receiver ALH (plotted in paper) (dominant frequency at 100 Hz)
                }

ids = [82]
fs_origin = 1000
dec_factor = 3
fs=fs_origin // dec_factor

for id in ids:

    ################################
    # load raw data and filter it: #
    ################################
    with h5py.File("data/raw_DAS/" + event_times[id][1], "r") as f:
        group_key = list(f.keys())[0]
        # als numpy array einlesen:
        data_raw = f[group_key][()]
        # zuschneiden auf accumulation zone daten:
        data_raw = data_raw[:, ::3]
        data_raw = data_raw[:, 0:770]

        npts = data_raw.shape[0]
        nch = data_raw.shape[1]

        # Applying filters:
        # 1. Remove mean along time axis for each channel (demean each channel)
        time_mean = data_raw.mean(axis=0).reshape(1, -1)
        data_demean = data_raw - time_mean
        print(data_demean.shape)

        # 2. Taper the data before filtering
        data_tapered = taper(data_demean, max_percentage=0.05)
        print(data_tapered.shape)
        # Band-Pass Filter between 10-90 Hz
        data_filtered = bandpass_filter(data_tapered, freqmin=10, freqmax=90, fs=fs_origin, dec_factor=dec_factor)
        print(data_filtered.shape)

        # 3. Automatic Gain Control with a window length of 2 seconds
        data_filtered_agc, tosum = AGC(data_filtered, fs * 2)
        print(data_filtered_agc.shape)

        # 4. Remove mean along spatial axis for each time sample (to remove common mode noise), optional
        data_filtered_time_mean = data_filtered_agc.mean(axis=1).reshape(-1, 1)
        data_filtered_demean = data_filtered_agc - data_filtered_time_mean
        print(data_filtered_demean.shape)

        # 5. Normalize each channel by its energy
        data_enorm = energy_norm(data_filtered_demean)

    #######################
    # load denoised data: #
    #######################
    with h5py.File("experiments/03_accumulation_horizontal/denoisedDAS/denoised_" + event_times[id][1], "r") as f:
        group_key = list(f.keys())[0]
        group_key2 = list(f[group_key].keys())[1]
        group_key3 = list(f[group_key][group_key2].keys())[0]
        denoised_data = f[group_key][group_key2][group_key3][()]
        denoised_data = denoised_data[:, 0:770]
        print(denoised_data.shape)

        # cut denoised data to size


    ############################
    # Visualize all 30 seconds #
    ############################
    data1 = data_raw[:, :]
    data2 = data_filtered[:, :]
    data3 = data_filtered_agc
    data4 = data_enorm

    # Create 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot different data in each subplot
    plot_channel_range_ax(axs[0, 0], data1, fs=fs_origin, cmap="seismic", title="Raw", clip_percentage=99.99)
    plot_channel_range_ax(axs[0, 1], data3, fs=fs, cmap="seismic", title="BP + AGC", clip_percentage=99.99)
    plot_channel_range_ax(axs[1, 0], data4, fs=fs, cmap="seismic", title="BP + AGC + EN", clip_percentage=99.99)
    plot_channel_range_ax(axs[1, 1], denoised_data, fs=400, cmap="seismic", title="J-invariant", clip_percentage=99.99)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    ###########################
    # Zoomed in Visualization #
    ###########################
    # Zoom into event
    start_time = 15
    end_time = 20
    start_channel = 560
    end_channel = 770

    # Generate some sample data
    data1 = data_demean[start_time * fs_origin:end_time * fs_origin, start_channel:end_channel]
    data2 = data_filtered[start_time * fs:end_time * fs, start_channel:end_channel]
    data3 = data_filtered_agc[start_time * fs:end_time * fs, start_channel:end_channel]
    data4 = data_enorm[start_time * fs:end_time * fs, start_channel:end_channel]
    data5 = denoised_data[start_time * 400: end_time*400, start_channel:end_channel]

    # Create 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot different data in each subplot
    plot_channel_range_ax(axs[0, 0], data1, fs=fs_origin, cmap="seismic", title="Raw", clip_percentage=99.9)
    plot_channel_range_ax(axs[0, 1], data3, fs=fs, cmap="seismic", title="BP + AGC", clip_percentage=99.9)
    plot_channel_range_ax(axs[1, 0], data4, fs=fs, cmap="seismic", title="BP + AGC + EN", clip_percentage=99.9)
    plot_channel_range_ax(axs[1, 1], data5, fs=400, cmap="seismic", title="J-invariant", clip_percentage=99.9)

    # Adjust layout
    plt.tight_layout()
    plt.show()