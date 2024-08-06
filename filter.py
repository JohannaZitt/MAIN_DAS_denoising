
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
    0: ["2020-07-27 08:17:34.6", "rhone1khz_UTC_20200727_081708.575.h5", 25, 30, 580, 770], # Category 1, Receiver ALH
    5: ["2020-07-27 19:43:30.5", "rhone1khz_UTC_20200727_194308.575.h5", 20, 25, 550, 720], # Category 1, Receiver ALH (plotted in paper)
    11: ["2020-07-27 19:43:01.4", "rhone1khz_UTC_20200727_194238.575.h5", 22, 26, 580, 720], # Category 1, Receiver ALH
    35: ["2020-07-27 03:03:20.2", "rhone1khz_UTC_20200727_030308.575.h5", 10, 15, 580, 720], # Category 1, Receiver ALH (dominant frequency at 100 Hz)
    83: ["2020-07-27 01:03:00.2", "rhone1khz_UTC_20200727_010238.575.h5", 20, 25, 570, 700], # Category 1, Receiver ALH

    20: ["2020-07-27 00:21:46.3", "rhone1khz_UTC_20200727_002138.575.h5", 5, 10, 580, 700], # Category 2, Receiver ALH (plotted in paper)
    24: ["2020-07-27 05:21:48.4", "rhone1khz_UTC_20200727_052138.575.h5", 9, 13, 550, 770], # Category 2, Receiver ALH (dominant frequency at 100 Hz)
    36: ["2020-07-27 20:47:34.8", "rhone1khz_UTC_20200727_204708.575.h5", 25, 30, 580, 700], # Category 2, Receiver ALH
    52: ["2020-07-27 20:00:30.4", "rhone1khz_UTC_20200727_200008.575.h5", 21, 25, 550, 770], # Category 2, Receiver ALH
    67: ["2020-07-27 23:17:54.0", "rhone1khz_UTC_20200727_231738.575.h5", 15, 20, 570, 700], # Category 2, Receiver ALH
    107: ["2020-07-27 01:25:19.6", "rhone1khz_UTC_20200727_012508.575.h5", 10, 15, 600, 700], # Category 2, Receiver ALH

    82: ["2020-07-27 05:04:55.8", "rhone1khz_UTC_20200727_050438.575.h5", 15, 20, 560, 770], # Category 3, Receiver ALH (plotted in paper) (dominant frequency at 100 Hz)
                }

ids = [0, 5, 11, 35, 83, 20, 24, 36, 52, 67, 107, 82]
fs_origin = 1000
dec_factor = 3
fs=fs_origin // dec_factor
save_plot = False

for id in ids:
    print(id)
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
        #print(data_demean.shape)

        # 2. Taper the data before filtering
        data_tapered = taper(data_demean, max_percentage=0.05)
        #print(data_tapered.shape)
        # Band-Pass Filter between 10-90 Hz
        data_filtered = bandpass_filter(data_tapered, freqmin=1, freqmax=120, fs=fs_origin, dec_factor=dec_factor) #best values: 10-90
        #print(data_filtered.shape)

        # 3. Automatic Gain Control with a window length of 2 seconds
        data_filtered_agc, tosum = AGC(data_filtered, fs * 2)
        #print(data_filtered_agc.shape)

        # 4. Remove mean along spatial axis for each time sample (to remove common mode noise), optional
        data_filtered_spatio_mean = data_filtered_agc.mean(axis=1).reshape(-1, 1)
        data_filtered_demean = data_filtered_agc - data_filtered_spatio_mean
        #print(data_filtered_demean.shape)

        # 5. Normalize each channel by its energy
        data_enorm = energy_norm(data_filtered_demean)

    ##################################
    # load denoised data vandenende: #
    ##################################
    with h5py.File("experiments/13_vandenende/denoisedDAS/denoised_" + event_times[id][1], "r") as f:
        group_key = list(f.keys())[0]
        group_key2 = list(f[group_key].keys())[1]
        group_key3 = list(f[group_key][group_key2].keys())[0]
        denoised_data_vandenende = f[group_key][group_key2][group_key3][()]
        denoised_data_vandenende = denoised_data_vandenende[:, 0:770]
        #print(denoised_data_vandenende.shape)


    #######################
    # load denoised data: #
    #######################
    with h5py.File("experiments/03_accumulation_horizontal/denoisedDAS/denoised_" + event_times[id][1], "r") as f:
        group_key = list(f.keys())[0]
        group_key2 = list(f[group_key].keys())[1]
        group_key3 = list(f[group_key][group_key2].keys())[0]
        denoised_data = f[group_key][group_key2][group_key3][()]
        denoised_data = denoised_data[:, 0:770]
        #print(denoised_data.shape)



    ############################
    # Visualize all 30 seconds #
    ############################
    data1 = data_raw[:, :]
    data2 = data_filtered[:, :]
    data3 = data_filtered_agc
    data4 = data_enorm
    data5 = denoised_data_vandenende
    data6 = denoised_data

    # Create 2x2 subplot
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    clip_percentage = 99.99

    # Plot different data in each subplot
    plot_channel_range_ax(axs[0, 0], data1, fs=fs_origin, cmap="seismic", title="Raw", clip_percentage=clip_percentage)
    plot_channel_range_ax(axs[0, 1], data2, fs=fs, cmap="seismic", title="BP", clip_percentage=clip_percentage)
    plot_channel_range_ax(axs[1, 0], data3, fs=fs, cmap="seismic", title="BP + AGC", clip_percentage=clip_percentage)
    plot_channel_range_ax(axs[1, 1], data4, fs=fs, cmap="seismic", title="BP + AGC + EN", clip_percentage=clip_percentage)
    plot_channel_range_ax(axs[2, 0], data5, fs=400, cmap="seismic", title="J-invariant van den Ende", clip_percentage=clip_percentage)
    plot_channel_range_ax(axs[2, 1], data6, fs=400, cmap="seismic", title="J-invariant", clip_percentage=clip_percentage)

    # Adjust layout
    plt.tight_layout()

    if save_plot:
        plt.savefig("plots/comparison_with_filter/" + str(id) + "_comparison_filter.pdf", dpi=300)
    else:
        plt.show()

    ###########################
    # Zoomed in Visualization #
    ###########################
    # Zoom into event
    start_time = event_times[id][2]
    end_time = event_times[id][3]
    start_channel = event_times[id][4]
    end_channel = event_times[id][5]

    # Generate some sample data
    data1 = data_demean[start_time * fs_origin:end_time * fs_origin, start_channel:end_channel]
    data2 = data_filtered[start_time * fs:end_time * fs, start_channel:end_channel]
    data3 = data_filtered_agc[start_time * fs:end_time * fs, start_channel:end_channel]
    data4 = data_enorm[start_time * fs:end_time * fs, start_channel:end_channel]
    data5 = denoised_data_vandenende[start_time * 400: end_time*400, start_channel:end_channel]
    data6 = denoised_data[start_time * 400: end_time * 400, start_channel:end_channel]

    # Create 2x2 subplot
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))

    # Plot different data in each subplot
    plot_channel_range_ax(axs[0, 0], data1, fs=fs_origin, cmap="seismic", title="Raw", clip_percentage=clip_percentage)
    plot_channel_range_ax(axs[0, 1], data2, fs=fs, cmap="seismic", title="BP", clip_percentage=clip_percentage)
    plot_channel_range_ax(axs[1, 0], data3, fs=fs, cmap="seismic", title="BP + AGC", clip_percentage=clip_percentage)
    plot_channel_range_ax(axs[1, 1], data4, fs=fs, cmap="seismic", title="BP + AGC + EN", clip_percentage=clip_percentage)
    plot_channel_range_ax(axs[2, 0], data5, fs=400, cmap="seismic", title="J-invariant van den Ende", clip_percentage=clip_percentage)
    plot_channel_range_ax(axs[2, 1], data6, fs=400, cmap="seismic", title="J-invariant", clip_percentage=clip_percentage)

    # Adjust layout
    plt.tight_layout()

    if save_plot:
        plt.savefig("plots/comparison_with_filter/" + str(id) + "_zoomed_comparison_filter.pdf", dpi=300)
    else:
        plt.show()
