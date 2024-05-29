import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import butter, lfilter
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from datetime import datetime, timedelta

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

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def load_das_data(folder_path, t_start, t_end, raw, channel_delta_start, channel_delta_end):

    # 1. load data
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=folder_path, convert=False)
    data = data.astype('f')

    # 2. downsample data in space:
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::6]
        else:
            data = data[:, ::3]
        headers['dx'] = 12

    # 3. cut to size
    ch_middel = int(3842/6)
    data = data[:, ch_middel - channel_delta_start:ch_middel + channel_delta_end]

    if raw:
        # 4. downsample in time
        data = resample(data, headers['fs'] / 400)
        headers['fs'] = 400

    # 5. bandpasfilter and normalize
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers['fs'], order=4)
        data[:,i] = data[:,i] / np.std(data[:,i])

    return data.T, headers, axis


def xcorr(x, y):  # Code by Martijn van den Ende

    # FFT of x and conjugation
    X_bar = np.fft.rfft(x).conj()
    Y = np.fft.rfft(y)

    # Compute norm of data
    norm_x_sq = np.sum(x ** 2)
    norm_y_sq = np.sum(y ** 2)
    norm = np.sqrt(norm_x_sq * norm_y_sq)

    # Correlation coefficients
    R = np.fft.irfft(X_bar * Y) / norm

    # Return correlation coefficient
    return np.max(R)


def compute_xcorr_window(x):  # Code by Martijn van den Ende
    Nch = x.shape[0]
    Cxy = np.zeros((Nch, Nch)) * np.nan

    for i in range(Nch):
        for j in range(i):
            Cxy[i, j] = xcorr(x[i], x[j])

    return np.nanmean(Cxy)


def compute_moving_coherence(data, bin_size):  # Code by Martijn van den Ende

    N_ch = data.shape[0]

    cc = np.zeros(N_ch)

    for i in range(N_ch):
        start = max(0, i - bin_size // 2)
        stop = min(i + bin_size // 2, N_ch)
        ch_slice = slice(start, stop)
        cc[i] = compute_xcorr_window(data[ch_slice])

    return cc


def plot_das_data(data, type="nix"):
    channels = data.shape[0]
    alpha=0.7
    i = 0
    plt.figure(figsize=(20, 12))
    for ch in range(channels):
        plt.plot(data[ch][:] + 12 * i, '-k', alpha=alpha)
        i += 1
    plt.show()
    #plt.savefig("plots/plot_" + type + ".png")


# Mögliche Events, die man plotten kann:
#  - 2020-07-06 19:10:51.0
#  - 2020-07-06 19:11:34.0
# ID : [starttime, start channel delta, end channel delta, category, closts seismometer]
event_times = {0: ["2020-07-27 08:17:34.5", 40, 40, 1, "ALH"], # Category 1, Receiver ALH
               5: ["2020-07-27 19:43:30.5", 45, 75, 1, "ALH"], # Category 1, Receiver ALH
               11: ["2020-07-27 19:43:01.0", 1, "ALH"], # Category 1, Receiver ALH
               35: ["2020-07-27 03:03:20.0", 1, "ALH"], # Category 1, Receiver ALH
               83: ["2020-07-27 01:03:00.0", 1, "ALH"], # Category 1, Receiver ALH

               9: ["2020-07-27 16:39:55.0", 2, "ALH"], # Category 2, Receiver ALH
               20: ["2020-07-27 00:21:46.3", 30, 30, 2, "ALH"], # Category 2, Receiver ALH
               24: ["2020-07-27 05:21:48.0", 2, "ALH"], # Category 2, Receiver ALH
               36: ["2020-07-27 20:47:35.0", 2, "ALH"], # Category 2, Receiver ALH
               52: ["2020-07-27 20:00:30.0", 2, "ALH"], # Category 2, Receiver ALH
               67: ["2020-07-27 23:17:54.0", 2, "ALH"], # Category 2, Receiver ALH
               107: ["2020-07-27 01:25:20.0", 2, "ALH"], # Category 2, Receiver ALH

               82: ["2020-07-27 05:04:55.0", 80, 150, 3, "ALH"], # Category 3, Receiver ALH
               113: ["2020-07-27 18:22:59.0", 3, "ALH"] # Category 3, Receiver ALH
               }
experiment = "03_accumulation_horizontal"

raw_path = os.path.join("data", "raw_DAS/")
denoised_path = os.path.join("experiments", experiment, "denoisedDAS/")

ids = [5, 20, 82]

# Basic Figure Settup:
fig, axs = plt.subplots(len(ids), 3,
                       gridspec_kw={
                           "width_ratios": [4, 4, 1],
                           "height_ratios": [1, 1, 1]},
                      sharey = False)
fig.set_figheight(13)
fig.set_figwidth(10)

for i, id in enumerate(ids):
    event_time = event_times[id][0]
    t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
    t_end = t_start + timedelta(seconds=2)

    raw_data, raw_headers, raw_axis = load_das_data(raw_path, t_start, t_end, raw=True, channel_delta_start=event_times[id][1], channel_delta_end=event_times[id][2])
    denoised_data, denoised1_headers, denoised1_axis = load_das_data(denoised_path, t_start, t_end, raw=False, channel_delta_start=event_times[id][1], channel_delta_end=event_times[id][2])

    channels = raw_data.shape[0]
    middle_channel = event_times[id][1]

    # Calculate CC
    bin_size = 11
    raw_cc = compute_moving_coherence(raw_data, bin_size)
    denoised_cc = compute_moving_coherence(denoised_data, bin_size)
    raw_denoised_cc = denoised_cc / raw_cc

    # Parameters for Plotting:
    cmap = "plasma" # verschiednene colormaps:  cividis, plasma, inferno, viridis, magma, (cmocean.cm.curl, seismic)
    t_start = 0
    t_end = denoised_data.shape[1]
    ch_start = 0
    ch_end = denoised_data.shape[0]
    ch_ch_spacing = 12
    vmin=-7
    vmax=7
    fs = 16

    # Plotting Raw Data
    axs[i, 0].imshow(raw_data, cmap=cmap, aspect="auto", interpolation="antialiased",
              extent=(0 ,(t_end-t_start)/400,0,ch_end * ch_ch_spacing/1000),
              vmin=vmin, vmax=vmax)
    axs[i, 0].set_ylabel("Offset [km]", fontsize=fs)
    axs[i, 0].tick_params(axis='y', labelsize=fs-2)

    # Plotting Denoised Data
    im = axs[i, 1].imshow(denoised_data, cmap=cmap, aspect="auto", interpolation="antialiased",
              extent=(0 ,(t_end-t_start)/400,0,ch_end * ch_ch_spacing/1000),
              vmin=vmin, vmax=vmax)
    axs[i, 1].set_yticklabels([])

    #cbar=fig.colorbar(im, ax=axs[i, 1])
    #cbar.set_label("Strain Rate [norm.]", fontsize=fs)
    #cbar.ax.tick_params(labelsize=fs-2, rotation=90)

    # Damit Graph gekippt angezeigt werden kann:
    x = np.arange(ch_end-ch_start)
    y_seis = raw_denoised_cc[ch_start:ch_end]
    X_seis = np.vstack((x, y_seis)).T
    X_seis = np.vstack((X_seis[:, 1], X_seis[:, 0])).T

    # Plotting CC Gain
    axs[i, 2].plot(X_seis[:, 0], X_seis[:, 1], color="black")
    axs[i, 2].invert_yaxis()
    axs[i, 2].axvline(x=1, color="black", linestyle="dotted")
    axs[i, 2].set_ylim(0, raw_denoised_cc.shape[0]-1)
    axs[i, 2].set_xlim(0, 6)
    axs[i, 2].set_yticks([])
    axs[i, 2].set_yticklabels([])

    # plot arrow where wiggle for wiggle comparison takes place:
    axs[i, 0].annotate("", xy=(0, (channels - middle_channel) * 0.0125), xytext=(-0.05, (channels - middle_channel) * 0.0125),
                       arrowprops=dict(color="red", arrowstyle="->", linewidth=2))
    axs[i, 1].annotate("", xy=(0, (channels - middle_channel) * 0.0125),
                       xytext=(-0.05, (channels - middle_channel) * 0.0125),
                       arrowprops=dict(color="red", arrowstyle="->", linewidth=2))

    print((channels - middle_channel) * 0.015)

    # Add letters in plots:
    letter_params = {
        "fontsize": fs + 2,
        "verticalalignment": "top",
        "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w", }
    }
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m"]

    for j in range(3):
        axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i * 3 + j], **letter_params)

# titles:
axs[0, 0].set_title("Noisy DAS Data", y=1.0, fontsize=fs+2)
axs[0, 1].set_title("Denoised DAS Data", y=1.0, fontsize=fs+2)
axs[0, 2].set_title("LWC", y=1.0, fontsize=fs+2)
# axs labels:
axs[2, 0].set_xlabel("Time [s]", fontsize=fs)
axs[2, 0].set_xticks([0.5, 1, 1.5], [0.5, 1, 1.5], fontsize=fs-2)
axs[2, 1].set_xlabel("Time [s]", fontsize=fs)
axs[2, 1].set_xticks([0.5, 1, 1.5], [0.5, 1, 1.5], fontsize=fs-2)
axs[2, 2].set_xlabel("Gain [-]", fontsize=fs)

axs[0, 2].set_xticks([1, 3, 5])
axs[0, 2].set_xticklabels([])
axs[1, 2].set_xticks([1, 3, 5])
axs[1, 2].set_xticklabels([])
axs[2, 2].set_xticks([1, 3, 5], [1, 3, 5], fontsize=fs-2)


axs[0, 0].set_xticks([0.5, 1.0, 1.5])
axs[0, 0].set_xticklabels([])
axs[0, 1].set_xticks([0.5, 1.0, 1.5])
axs[0, 1].set_xticklabels([])
axs[1, 0].set_xticks([0.5, 1.0, 1.5])
axs[1, 0].set_xticklabels([])
axs[1, 1].set_xticks([0.5, 1.0, 1.5])
axs[1, 1].set_xticklabels([])

plt.tight_layout()
#plt.show()
plt.savefig("plots/waterfall/waterfall_all_in_one.pdf", dpi=400)
