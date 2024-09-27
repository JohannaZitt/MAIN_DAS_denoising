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


def get_event_time_from_id(id):

    folder_to_filenames = "experiments/03_accumulation_horizontal/plots/accumulation/folder_for_sta_lta"

    strings = os.listdir(folder_to_filenames)
    filtered_strings = [s for s in strings if "ID:"+str(id)+"_" in s]
    filtered_string = filtered_strings[0]
    time_stamp = filtered_string[-21:-13]

    return time_stamp

def load_das_data(folder_path, t_start, t_end, raw):

    # 1. load data
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=folder_path, convert=False)
    data = data.astype("f")

    print(data.shape)

    # 2. downsample data in space:
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::6]
        else:
            data = data[:, ::3]
        headers["dx"] = 12


    # 3. cut to size
    #ch_middel = get_middel_channel(receiver)  # get start and end channel:
    #data = data[400:]

    # 4. downsample in time
    if raw:
        data = resample(data, headers["fs"] / 400)
        headers["fs"] = 400

    # 5. bandpasfilter and normalize
    if raw:
        for i in range(data.shape[1]):
            data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers["fs"], order=4)
            data[:, i] = data[:,i] / np.abs(data[:,i]).max()


    return data, headers, axis

def plot_sta_lta_imshow(data_raw, data_denoised, id):

    fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=300, figsize=(6, 6), sharex=True)

    im1 = ax1.imshow(data_raw.T, aspect="auto", cmap="magma")
    cbar1 = plt.colorbar(im1, ax=ax1, label="STA/LTA ratio")
    ax1.set_ylabel("Chanel ID")
    ax1.set_title("ID: " + str(id) + "  STA/LTA ratios of raw data")

    im2 = ax2.imshow(data_denoised.T, aspect="auto", cmap="magma")
    cbar2 = plt.colorbar(im2, ax=ax2, label="STA/LTA ratio")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Chanel ID")
    ax2.set_title("ID: " + str(id) + "  STA/LTA ratios of denoised data")

    plt.tight_layout()
    plt.show()

def plot_sta_lta_single_channel(data_raw, data_denoised, objective_channel, id):
    fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=300, figsize=(6, 6), sharex=True)

    ax1.plot(data_raw[:, objective_channel])
    ax1.set_ylabel("STA/LTA ratio")
    ax1.set_title("ID: " + str(id) + ", Channel: " + str(objective_channel) + ", Type: Raw")
    ax2.plot(data_denoised[:, objective_channel])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("STA/LTA ratio")
    ax1.set_title("ID: " + str(id) + ", Channel: " + str(objective_channel) + ", Type: Denosied")

    plt.tight_layout()
    plt.show()




ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 22, 28, 32, 34, 35, 39, 41, 48, 62, 66, 74,
            82, 83, 101, 104, 108, 119, 16, 17, 18, 20, 21, 24, 25, 36, 40, 42, 45, 46, 50, 52, 56, 59, 64,
            65, 67, 75, 78, 80, 85, 87, 90, 94, 95, 96, 98, 102, 105, 107, 116, 118]

ids = [5]

# params for 5:
#       - tstart = 25
#       - start_ch = 580
#       - end_ch = 700
#       -



start_ch = 580 # must be smaller than 640
end_ch = 700

fs = 400
short_window_length = 0.1 # 0.1
long_window_length = 5 #5
# short_window_length values: 0.1
# long_window_length values: 5
short_wl = int(short_window_length * fs)
long_wl = int(long_window_length * fs)
chanel_range = 20

for id in ids:
    event_date = "2020-07-27"
    event_time = get_event_time_from_id(id=id)

    t_start = datetime.strptime(event_date + " " + event_time + ".0", "%Y-%m-%d %H:%M:%S.%f")
    t_start = t_start - timedelta(seconds=25)
    t_end = t_start + timedelta(seconds=35)

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
    # plot_sta_lta_imshow(csl_raw, csl_denoised, id)


    # 1.3 Plot LTA/STA for singe chanel:
    # objective_chanel = 50
    # plot_sta_lta_single_channel(csl_raw, csl_denoised, objective_chanel, id)

    # CHECK FOR DIFFERNENT TIMES
    if not raw_data.shape[0] == denoised_data.shape[0]:
        print("IN IF")
        difference = raw_data.shape[0] - denoised_data.shape[0]
        raw_data = raw_data[difference:, :]


    # 1.4 plot for paper:

    # Params
    cmap = "plasma"
    aspect = "auto"
    color = "black"
    alpha = 0.7
    line_width1=0.5
    line_width2=1
    vmin = -0.3
    vmax = 0.3
    fs_s = 10
    fs_m = 12
    fs_l = 14

    # Cut to size:
    raw_data = raw_data[long_wl:]
    denoised_data = denoised_data[long_wl:]
    csl_raw = csl_raw[long_wl:]
    csl_denoised = csl_denoised[long_wl:]

    # Compute Stacked Data
    ch1 = 44
    ch2 = ch1 + chanel_range
    stacked_raw = np.sum(raw_data[:,ch1:ch2], axis=1)
    stacked_denoised = np.sum(denoised_data[:, ch1:ch2], axis=1)
    stacked_csl_raw = np.sum(csl_raw[:, ch1:ch2], axis=1)
    stacked_csl_denoised = np.sum(csl_denoised[:, ch1:ch2], axis=1)

    fig, ax = plt.subplots(nrows=4, ncols=2, dpi=300, figsize=(7, 9), sharex=False, sharey=False,
                           gridspec_kw={
                               "height_ratios": [4, 1, 4, 1],
                               "width_ratios": [1, 1]})

    im1 = ax[0, 0].imshow(raw_data.T, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)
    #plt.colorbar(im1, ax=ax[0, 0], label="Strain Rate [norm]")
    im2 = ax[0, 1].imshow(denoised_data.T, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)
    #plt.colorbar(im2, ax=ax[0, 1], label="Strain Rate [norm.]")
    #print(np.abs(denoised_data).max())
    #print(np.abs(raw_data).max())
    ax[1, 0].plot(stacked_raw, color=color, linewidth=line_width1, alpha=alpha)
    ax[1, 1].plot(stacked_denoised, color=color, linewidth=line_width1, alpha=alpha, )

    im3 = ax[2, 0].imshow(csl_raw.T, aspect=aspect, cmap=cmap, vmin=0, vmax=45)#np.abs(csl_raw).max()
    #plt.colorbar(im3, ax=ax[2, 0], label="STA/LTA Ratio")
    im4 = ax[2, 1].imshow(csl_denoised.T, aspect=aspect, cmap=cmap, vmin=0, vmax=45)#np.abs(csl_denoised).max()
    #plt.colorbar(im4, ax=ax[2, 1], label="STA/LTA Ratio")
    ax[3, 0].plot(stacked_csl_raw, color=color, linewidth=line_width2, alpha=alpha)
    ax[3, 1].plot(stacked_csl_denoised, color=color, linewidth=line_width2, alpha=alpha)

    # Set Labels:
    ax[0, 0].set_yticks([20, 40, 60, 80, 100], [600, 480, 360, 240, 120], fontsize=fs_s)
    ax[0, 0].set_ylabel("Offset [m]", fontsize=fs_m)
    ax[0, 0].set_xticks([2000, 4000, 6000, 8000, 10000], [])
    ax[0, 0].set_title("Noisy", fontsize=fs_l)

    ax[0, 1].set_xticks([2000, 4000, 6000, 8000, 10000], [])
    ax[0, 1].set_yticks([20, 40, 60, 80, 100], [])
    ax[0, 1].set_title("Denoised", fontsize=fs_l)

    ax[1, 0].set_yticks([])
    ax[1, 0].set_ylabel("Stacked\nDAS", fontsize=fs_m)
    ax[1, 0].set_xticks([2000, 4000, 6000, 8000, 10000], [])
    ax[1, 0].set_xlim(0, raw_data.shape[0])

    ax[1, 1].set_xticks([2000, 4000, 6000, 8000, 10000], [])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim(0, raw_data.shape[0])

    ax[2, 0].set_yticks([20, 40, 60, 80, 100], [600, 480, 360, 240, 120], fontsize=fs_s)
    ax[2, 0].set_ylabel("Offset [m]", fontsize=fs_m)
    ax[2, 0].set_xticks([2000, 4000, 6000, 8000, 10000], [])

    ax[2, 1].set_xticks([2000, 4000, 6000, 8000, 10000], [])
    ax[2, 1].set_yticks([20, 40, 60, 80, 100], [])

    ax[3, 0].set_xlabel("Time [s]", fontsize=fs_m)
    ax[3, 0].set_xticks([2000, 4000, 6000, 8000, 10000], [5, 10, 15, 20, 25], fontsize = fs_s)
    ax[3, 0].set_yticks([])
    ax[3, 0].set_ylim((-20, 800))
    ax[3, 0].set_ylabel("Stacked\nSTA/LTA Ratio", fontsize=fs_m)
    ax[3, 0].set_xlim(0, csl_raw.shape[0])

    ax[3, 1].set_xlabel("Time [s]", fontsize=fs_m)
    ax[3, 1].set_xticks([2000, 4000, 6000, 8000, 10000], [5, 10, 15, 20, 25], fontsize=fs_s)
    ax[3, 1].set_yticks([])
    ax[3, 1].set_ylim((-20, 900))
    ax[3, 1].set_xlim(0, csl_raw.shape[0])

    # Plot arroes
    arrow_style = "fancy,head_width=0.5,head_length=1.0"
    arrow_style2 = "fancy,head_width=0.5,head_length=0.5"
    ax[0, 0].annotate("", xy=(0, ch1),
                       xytext=(-0.05, ch1),
                       arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=2))
    ax[0, 0].annotate("", xy=(0, ch2),
                      xytext=(-0.05, ch2),
                      arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=2))
    ax[2, 0].annotate("", xy=(0, ch1),
                      xytext=(-0.05, ch1),
                      arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=2))
    ax[2, 0].annotate("", xy=(0, ch2),
                      xytext=(-0.05, ch2),
                      arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=2))

    # Add letters in plots:
    letter_params = {
        "fontsize": fs_m,
        "verticalalignment": "top",
        "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w"}
    }
    letters = ["a", "b", "c", "d", "e", "f ", "g", "h", "i ", "j ", "k", "l ", "m", "n", "o", "p"]

    for i in range(4):
        for j in range(2):
            ax[i, j].text(x=0.0, y=1.0, transform=ax[i, j].transAxes, s=letters[i*2+j], **letter_params)

    plt.tight_layout()
    #plt.savefig("plots/stalta/" + str(id) + "_stalta_paper.pdf", dpi=300)
    plt.show()

    print("Denoised", str(max(stacked_csl_denoised)))
    print("Raw", str(max(stacked_csl_raw)))









