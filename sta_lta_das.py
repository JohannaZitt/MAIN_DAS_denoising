import os

import numpy as np
import matplotlib.pyplot as plt
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from scipy.signal import butter, lfilter
from datetime import datetime, timedelta
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, trigger_onset
from obspy.signal.trigger import plot_trigger
from obspy import Trace

"""


There are different approaches to use sta/lta for das data

1. from jousset2022.pdf volcanic events
    STA/LTA is computed for every channel along the fibre and then averaged.
    For this characteristic function the median absolute deviation is calculated.
    An event is declared when a threshold defined as the median plus three times the MDA
    is exceeded.

2. from klaasen2021.pdf (Paitz and Walter) Volcaon-Glacial-environment
    STA/LTA is computed for the stacked DAS channels (corresponding to 30-40 meters)
    The STA window length was 0.3 s, the LTA window length 60 s, the trigger value 7.5,
    and the detrigger value 1.5.


There are a lot of parameters to vary:
1. sta window: STA duration must be longer than a few periods of a typical expected seismic signal
               0.2-0.5 seconds in cryoseismolgy
               Since we consider multiple channels at once, sta duration can be chosen longer than considering single trace
2. lta window: LTA duration must be longer than a few periods of a typically irregular seismic noise fluctuation
               5-30 seconds in cryoseismology
3. trigger threshold: 4 worked well
4. detrgger threshold: 1 worked well
5. amount of channels: depends strongly on the extend of the icequake. using 20 channel, corresponding to 240 m cable length 


"""


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def resample(data, ratio):
    try:
        res = np.zeros((int(data.shape[0]/ratio) + 1, data.shape[1]))
        for i in range(data.shape[1]):

            res[:,i] = np.interp(np.arange(0, len(data), ratio), np.arange(0, len(data)), data[:,i])
    except ValueError as e:
        res = np.zeros((int(data.shape[0] / ratio), data.shape[1]))
        for i in range(data.shape[1]):
            res[:, i] = np.interp(np.arange(0, len(data), ratio), np.arange(0, len(data)), data[:, i])
    return res


def get_event_time_from_id(id, cat):

    folder_to_filenames = []
    if cat == 1:
        folder_to_filenames = "experiments/03_accumulation_horizontal/plots/accumulation/1_raw:visible_denoised:better_visible"
    if cat == 2:
        folder_to_filenames = "experiments/03_accumulation_horizontal/plots/accumulation/2_raw:not_visible_denoised:visible"

    strings = os.listdir(folder_to_filenames)
    filtered_strings = [s for s in strings if "ID:"+str(id)+"_" in s]
    filtered_string = filtered_strings[0]
    time_stamp = filtered_string[-21:-13]

    return time_stamp

def load_das_data(folder_path, t_start, t_end, raw):

    # 1. load data
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=folder_path, convert=False)
    data = data.astype("f")

    # 2. downsample data in space:
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::6]
        else:
            data = data[:, ::3]
        headers["dx"] = 12


    # 3. cut to size
    #ch_middel = get_middel_channel(receiver)  # get start and end channel:
    data = data[400:]

    # 4. downsample in time
    if raw:
        data = resample(data, headers["fs"] / 400)
        headers["fs"] = 400

    # 5. bandpasfilter and normalize
    if raw:
        for i in range(data.shape[1]):
            data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers["fs"], order=4)
            data[:,i] = data[:,i] / np.abs(data[:,i]).max()


    return data, headers, axis




ids_cat1 = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 14, 15, 22, 28, 32, 34, 35, 39, 41, 48, 62, 66, 74, 83, 101, 104, 108, 119]
ids_cat2 = [7, 9, 16, 17, 18, 20, 21, 24, 25, 36, 40, 42, 45, 46, 50, 52, 56, 59, 64, 65, 67, 75, 78, 80, 85, 87, 90, 94, 95, 96, 98, 102, 105, 107, 116, 118]

ids = [0, 1, 2, 3, 11, 34, 35, 83, 104]

start_ch = 600 # must be smaller than 640
end_ch = 700
middle_ch = 640 - start_ch

fs = 400
short_window_length = 0.3 # 0.1
long_window_length = 3 #5
# short_window_length values: 0.1
# long_window_length values: 5
short_wl = int(short_window_length * fs)
long_wl = int(long_window_length * fs)
chanel_range = 20
chanel_gap = 10

for id in ids:
    event_date = "2020-07-27"
    event_time = get_event_time_from_id(id=id, cat=1)

    t_start = datetime.strptime(event_date + " " + event_time + ".0", "%Y-%m-%d %H:%M:%S.%f")
    t_start = t_start - timedelta(seconds=15)
    t_end = t_start + timedelta(seconds=30)

    # load raw DAS data:
    raw_folder_path = "data/raw_DAS/"
    raw_data, raw_headers, raw_axis = load_das_data(folder_path=raw_folder_path, t_start=t_start, t_end=t_end, raw=True)
    raw_data = raw_data[:, start_ch:end_ch]
    #print(raw_data.shape)

    # load denoised DAS data:
    denoised_folder_path = "experiments/03_accumulation_horizontal/denoisedDAS/"
    denoised_data, denoised_headers, denoised_axis = load_das_data(folder_path=denoised_folder_path, t_start=t_start, t_end=t_end, raw=False)
    denoised_data = denoised_data[:, start_ch:end_ch]
    #print(denoised_data.shape)



    # 1.1 Compute LTA/STA for every chanel
    csl_raw = np.zeros(raw_data.shape)
    csl_denoised = np.zeros(denoised_data.shape)
    for ch in range(raw_data.shape[1]):
        csl_raw[:, ch] = recursive_sta_lta(raw_data[:, ch], short_wl, long_wl)
        csl_denoised[:, ch] = recursive_sta_lta(denoised_data[:, ch], short_wl, long_wl)

    # 1.2 Plot data as imshow:
    fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=300, figsize=(6, 6), sharex=True)

    im1 = ax1.imshow(csl_raw.T, aspect="auto", cmap="magma")
    cbar1 = plt.colorbar(im1, ax=ax1, label="STA/LTA ratio")
    ax1.set_ylabel("Chanel ID")
    ax1.set_title("ID: " + str(id) + "  STA/LTA ratios of raw data")

    im2 = ax2.imshow(csl_denoised.T, aspect="auto", cmap="magma")
    cbar2 = plt.colorbar(im2, ax=ax2, label="STA/LTA ratio")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Chanel ID")
    ax2.set_title("ID: " + str(id) + "  STA/LTA ratios of denoised data")

    plt.tight_layout()
    plt.show()

    # 1.3 Plot LTA/STA for singe chanel:
    objective_chanel = 50

    fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=300, figsize=(6, 6), sharex=True)

    ax1.plot(csl_raw[:, objective_chanel])
    ax1.set_ylabel("STA/LTA ratio")
    ax1.set_title("ID: " + str(id) + ", Channel: " + str(objective_chanel) + ", Type: Raw")
    ax2.plot(csl_denoised[:, objective_chanel])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("STA/LTA ratio")
    ax1.set_title("ID: " + str(id) + ", Channel: " + str(objective_chanel) + ", Type: Denosied")

    plt.tight_layout()
    plt.show()




    x = np.arange((end_ch-start_ch)/chanel_gap)
    for i in x:
        #print(i)
        if i == 1:
            break
        stacked_raw = np.sum(csl_raw[:, int(i*chanel_gap) : int((i*chanel_gap)+chanel_range)], axis=1)
        stacked_raw /= chanel_range

        stacked_denoised = np.sum(csl_denoised[:, int(i * chanel_gap): int((i * chanel_gap) + chanel_range)], axis=1)
        stacked_denoised /= chanel_range

        fig, ax = plt.subplots(ncols=2, nrows=2, dpi=300, figsize=(6,6), sharex=True, sharey=True)

        ax[0, 0].plot(stacked_raw[long_wl:])
        ax[0, 0].set_title("Stacked LTA/STA value for raw data")
        ax[0, 0].set_ylabel("LTA/STA Ratio")

        ax[0, 1].plot(stacked_denoised[long_wl:])
        ax[0, 1].set_title("Stacked LTA/STA value for denoised data")

        ax[1, 0].plot(raw_data[long_wl:, int(i*chanel_gap) + int(chanel_range/2)])
        ax[1, 0].set_title("DAS channel raw")
        ax[1, 0].set_ylabel("Strain Rate")


        ax[1, 1].plot(denoised_data[long_wl:, int(i * chanel_gap) + int(chanel_range / 2)])
        ax[1, 1].set_title("DAS channel denoised")

        plt.tight_layout()
        plt.show()

        #plt.plot(stacked_raw)
        #plt.title("Raw, ID:" + str(id))
        #plt.show()
        #plt.plot(stacked_denoised)
        #plt.title("Denoised, ID:" + str(id))
        #plt.show()



    # 3. Stack DAS data and than compute LTA/STA:
    # stacked_data = np.sum(data, axis=1)
    # cft = classic_sta_lta(stacked_data, sta, lta)






