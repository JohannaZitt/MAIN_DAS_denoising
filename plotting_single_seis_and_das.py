import re
from datetime import datetime, timedelta, timezone
import numpy as np
from obspy import UTCDateTime
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from obspy import read
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from scipy.signal import butter, lfilter


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


def get_middel_channel(receiver):
    channel = 0
    if receiver == "AKU":
        channel = 3740
    elif receiver == "AJP":
        channel = 3460
    elif receiver == "ALH":
        channel = 3842
    elif receiver == "RA82":
        channel = 1300
    elif receiver == "RA85":
        channel = 1070
    elif receiver == "RA87":
        channel = 1230
    elif receiver == "RA88":
        channel = 1615 # 1600
    else:
        print("There is no start nor end channel for receiver " + receiver + '.')

    channel = int(channel/6)
    return channel

def load_das_data(folder_path, t_start, t_end, receiver, raw, ch_delta_start, ch_delta_end):

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
    ch_middel = get_middel_channel(receiver)  # get start and end channel:
    data = data[:, ch_middel-ch_delta_start:ch_middel+ch_delta_end]

    if raw:
        # 4. downsample in time
        data = resample(data, headers["fs"] / 400)
        headers["fs"] = 400

    # 5. bandpasfilter and normalize
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers["fs"], order=4)
        data[:,i] = data[:,i] / np.abs(data[:,i]).max()
        #data[:, i] = data[:,i] / np.std(data[:, i])

    return data, headers, axis

def plot_sectionplot(raw_data, denoised_data, seis_data, seis_stats, saving_path, middle_channel):

    # different fonts:
    font_s = 12
    font_m = 14
    font_l = 16

    # parameters:
    channels = raw_data.shape[0]
    ch_spacing = 12
    fs = 400
    alpha = 0.7
    alpha_dashed_line = 0.2

    fig, ax = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={"height_ratios": [7, 1]}, dpi=400)

    plt.rcParams.update({"font.size": font_l})
    plt.tight_layout()

    # Plotting raw_data!
    plt.subplot(221)
    n = 0
    for ch in range(channels):
        plt.plot(raw_data[-ch][:] + 1.5 * n, "-k", alpha=alpha)
        n += 1
    for i in range(3):
        plt.axvline(x=(i + 1) * (fs / 2), color="black", linestyle="dashed", alpha=alpha_dashed_line)
    plt.xticks([], [])
    # plt.xlabel("Time[s]", size=font_m)
    plt.ylabel("Offset [km]", size=font_m)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
    #print(n*1.5)
    #print(channels*ch_spacing/1000)
    #plt.yticks(np.arange(0, n * 1.5, 12), np.arange(0, channels*ch_spacing/1000, 0.1), size=font_s)
    plt.yticks(np.arange(0, n * 1.5, 45), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=font_s)
    plt.title("Noisy DAS Data", loc="left")
    plt.annotate("", xy=(0, (channels-middle_channel) * 1.5), xytext=(-1, (channels-middle_channel) * 1.5),
                 arrowprops=dict(color="red", arrowstyle="->", linewidth=2))

    # Plotting Denoised Data:
    plt.subplot(222)
    i = 0
    for ch in range(channels):
        plt.plot(denoised_data[-ch][:] + 1.5 * i, "-k", alpha=alpha)
        i += 1
    for i in range(3):
        plt.axvline(x=(i + 1) * (fs / 2), color="black", linestyle="dashed", alpha=alpha_dashed_line)
    plt.yticks(np.arange(0, n * 1.5, 45), []) #25, 12, 45
    plt.xticks([], [])
    plt.title("Denoised DAS Data", loc="left")
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    plt.annotate("", xy=(0, (channels - middle_channel) * 1.5), xytext=(-1, (channels - middle_channel) * 1.5),
                 arrowprops=dict(color="red", arrowstyle="->", linewidth=2))
    plt.subplots_adjust(wspace=0.05)

    # plotting seismometer data 1
    seis_fs = seis_stats["sampling_rate"]
    plt.subplot(223)
    plt.plot(seis_data, color="black", alpha=0.4)
    for i in range(3):
        plt.axvline(x=(i + 1) * seis_fs / 2, color="black", linestyle="dashed", alpha=alpha_dashed_line)
    plt.xlabel("Time[s]", size=font_m)
    if seis_fs == 500:
        plt.xticks(np.arange(0, 1001, 250), np.arange(0, 2.1, 0.5), size=font_s)
    else:  # if seis_fs==400
        plt.xticks(np.arange(0, 801, 200), np.arange(0, 2.1, 0.5), size=font_s)
    plt.ylabel("Seismometer", size=font_l)
    plt.yticks([])
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    # plotting seismometer data 2
    plt.subplot(224)
    plt.plot(seis_data, color="black", alpha=0.4)
    for i in range(3):
        plt.axvline(x=(i + 1) * seis_fs / 2, color="black", linestyle="dashed", alpha=alpha_dashed_line)
    plt.xlabel("Time[s]", size=font_m)
    if seis_fs == 500:
        plt.xticks(np.arange(0, 1001, 250), np.arange(0, 2.1, 0.5), size=font_s)
    else:  # if seis_fs==400
        plt.xticks(np.arange(0, 801, 200), np.arange(0, 2.1, 0.5), size=font_s)
    plt.yticks([])
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    plt.tight_layout()
    plt.show()
    #plt.savefig(saving_path + ".png", bbox_inches="tight", pad_inches=0.5, dpi=400)

def plot_wiggle_comparison(raw_data, denoised_data, seis_data, middle_channel, saving_path):

    custom_red = "#BF4843"
    custom_pink = "#9A1967"

    fs=14

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

    # plotting data
    axs[0].plot(raw_data[middle_channel], color=custom_red, label="Noisy DAS Channel", linewidth=1.5)
    axs[0].plot(denoised_data[middle_channel], color="black", label="Denoised DAS Channel", linewidth=1.5)
    axs[1].plot(seis_data, color=custom_pink, label="Seismometer", linewidth=1.5)
    axs[1].plot(denoised_data[middle_channel], color="black", label="Denoised DAS Channel", linewidth=1.5)

    # labels
    axs[0].set_xticks([100, 200, 300], [])
    axs[1].set_xlabel("Time [s]", fontsize=fs)
    axs[1].set_xticks([100, 200, 300], [0.25, 0.5, 0.75], fontsize=fs)
    axs[0].set_ylabel("Strain Rate [norm.]", fontsize=fs)
    axs[0].set_yticks([], [])
    axs[1].set_yticks([], [])
    axs[1].set_ylabel("Ground Velocity [norm.]", fontsize=fs)
    axs[0].legend(loc="upper right", fontsize=fs)
    axs[1].legend(loc="upper right", fontsize=fs)


    plt.tight_layout()
    plt.show()
    #plt.savefig(saving_path, dpi=400)



# ID : [starttime, start channel delta, end channel delta, category, closts seismometer]
events = {5: ["2020-07-27 19:43:30.5", 45, 75, 1, "ALH", "5_"],
         20: ["2020-07-27 00:21:46.3", 30, 30, 2, "ALH", "20"],
         82: ["2020-07-27 05:04:55.0", 80, 150, 3, "ALH", "82"]
         }

id = 82
event_time = events[id][0]
t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
t_end = t_start + timedelta(seconds=2)
experiment = "03_accumulation_horizontal"
receiver = "ALH"

# load seismometer data:
string_list = os.listdir("data/test_data/accumulation/")
filtered_strings = [s for s in string_list if s.startswith("ID:" + events[id][5])]
seis_data_path = "data/test_data/accumulation/" + filtered_strings[0]
seis_stream = read(seis_data_path, starttime=UTCDateTime(t_start-timedelta(seconds=1)), endtime=UTCDateTime(t_end))
seis_data = seis_stream[0].data
seis_stats = seis_stream[0].stats
seis_data = butter_bandpass_filter(seis_data, 1, 120, fs=seis_stats.sampling_rate, order=4)
seis_data = seis_data/np.abs(seis_data).max()
seis_data = seis_data[400:]

# load raw DAS data:
raw_folder_path = "data/raw_DAS/"
raw_data, raw_headers, raw_axis = load_das_data(folder_path =raw_folder_path, t_start=t_start, t_end=t_end, receiver=receiver, raw=True, ch_delta_start=events[id][1], ch_delta_end=events[id][2])

# load denoised DAS data
denoised_folder_path = "experiments/" + experiment + "/denoisedDAS/"
denoised_data, denoised_headers, denoised_axis = load_das_data(folder_path=denoised_folder_path, t_start=t_start, t_end=t_end, receiver=receiver, raw=False, ch_delta_start=events[id][1], ch_delta_end=events[id][2])

saving_path = "plots/section_plots/" + str(id) + "_sectionplot"


plot_sectionplot(raw_data=raw_data.T, denoised_data=denoised_data.T, seis_data=seis_data, seis_stats=seis_stats, saving_path=saving_path, middle_channel=events[id][1])
plot_wiggle_comparison(raw_data=raw_data[200:600].T, denoised_data=denoised_data[200:600].T, seis_data=seis_data[200:600], middle_channel=events[id][1], saving_path="plots/wiggle_comparison/" + str(id) + "_wiggle_comparison.png")











