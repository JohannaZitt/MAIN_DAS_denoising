import os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import butter, lfilter
'''

Goal: Giving an overview of training data: surface events, stick-slip events, horizontal, vertical events, ablation, accumulation zone

'''
def resample(stream, ratio):
    """
    :param stream: stream object, which has to be resampled
    :param ratio: resample ratio = fs_old/fs_new
    :return: resampled data as np array
    """

    # convert stream to np array
    n_trc = len(stream)
    n_t = stream[0].stats.npts
    data: ndarray = np.zeros((n_trc, n_t))
    for i in range(n_trc):
        data[i] = stream[i].data[0:n_t]

    # resample
    res = np.zeros((data.shape[0], int(data.shape[1]/ ratio) + 1))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = 'band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order = 4):
    b, a = butter_bandpass(lowcut, highcut, fs, order = order)
    y = lfilter(b, a, data)
    return y

event_path="data/training_data/raw_seismometer/"
events=["ablation_horizontal_stick_slip/ID:59_2020-08-04_00:27:46_RA88_EH2.mseed", "ablation_horizontal_surface/ID:46_2020-07-30_11:52:46_RA88_EH3.mseed",
        "ablation_vertical_stick_slip/ID:12_2020-08-03_23:44:33_RA88_EHZ.mseed", "ablation_vertical_surface/ID:12_2020-07-23_11:39:19_RA82_EHZ.mseed",
        "accumulation_horizontal_stick_slip/ID:29_2020-07-12_00:09:12_c0AKU_p2.mseed", "accumulation_horizontal_surface/ID:20_2020-07-27_13:27:42_c0ALH_p2.mseed",
        "accumulation_vertical_stick_slip/ID:18_2020-07-15_16:52:08_c0AJP_p0.mseed", "accumulation_vertical_surface/ID:40_2020-07-15_10:40:55_c0AKU_p0.mseed"]
titels = ["Ablation - Horizontal - Stick-Slip", "Ablation - Horizontal - Surface", "Ablation - Vertical - Stick-Slip", "Ablation - Vertical - Surface",
          "Accumulation - Horizontal - Stick-Slip", "Accumulation - Horizontal - Surface", "Accumulation - Vertical - Stick-Slip", "Accumulation - Vertical - Surface"]
t_start = [950, 950, 1050, 900, 1050, 1200, 900, 1100]
t = 500
fontsize = 12
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
letter_params = {
    "fontsize": fontsize,
    "verticalalignment": "top",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
}

fig, axes = plt.subplots(4, 2, figsize=(12,8))

for i, event in enumerate(events):
    # load data:
    stream = read(os.path.join(event_path, event))

    print(stream[0].stats)

    # preprocessing data: resample and filter:
    if stream[0].stats["sampling_rate"] == 500:
        stream = resample(stream, 500/400)
    data = stream[0].data
    data = butter_bandpass_filter(data, lowcut=1, highcut=120, fs=400, order=4)
    data = data / np.std(data)

    # plot data:
    row = i // 2
    col = i % 2

    axes[row, col].plot(data[t_start[i]:t_start[i]+t], color="black", linewidth=1)
    axes[row, col].set_title(titels[i])
    axes[row, col].set_yticks([])
    axes[row, col].text(x=0.0, y=1.0, transform=axes[row, col].transAxes, s=letters[i], **letter_params)
    #axes[row, col].set_ylim(-30, 30)

    if col==0:
        axes[row, col].set_ylabel("ground velocity [norm]", fontsize=fontsize-2)

    if row==3:
        axes[row, col].set_xticks([0, 100, 200, 300, 400, 500], [0, 0.25, 0.5, 0.75, 1, 1.25], fontsize = fontsize-2)
        axes[row, col].set_xlabel("Time [s]", fontsize=fontsize-2)
    else:
        axes[row, col].set_xticks([])

plt.tight_layout()
plt.show()
