import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os
from scipy.signal import butter, lfilter
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from datetime import datetime, timedelta
from obspy import UTCDateTime
from obspy import read
import re

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

def load_das_data(folder_path, t_start, t_end, raw):

    # 1. load data
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=folder_path, convert=False)
    data = data.astype('f')

    # 2. downsample data in space:
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::4]
        else:
            data = data[:, ::2]
        headers['dx'] = 8

    # 3. cut to size
    ch_middel = int(3460/4) # get start and end channel:
    data = data[:, ch_middel-40:ch_middel+40]

    if raw:
        # 4. downsample in time
        data = resample(data, headers['fs'] / 400)
        headers['fs'] = 400

    # 5. bandpasfilter and normalize
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers['fs'], order=4)
        data[:,i] = data[:,i] / np.std(data[:,i])

    return data.T, headers, axis


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

def extract_id(filename):
    id = re.search(r'ID:(\d+)', filename).group(1)
    return int(id)

def get_seismometer_event(id, seismometer_path):
    events = os.listdir(seismometer_path)
    for event in events:
        if extract_id(event) == id:
            return event
    print("Error: no matching ID found")

def plot_data(raw_data, denoised_data, seis_data, seis_stats, saving_path, id):

    # parameters:
    font_size = 16
    channels = raw_data.shape[0]
    time_samples = raw_data.shape[1]
    fs = 400
    alpha = 0.7
    alpha_dashed_line = 0.2
    n_hline = 4
    plot_title = id + ': ' + ', ' + str(seis_stats['starttime']) + ' - ' + str(seis_stats['endtime']) + ', ' + str(seis_stats['station'])

    fig, ax = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'height_ratios': [5, 1]})
    #fig.suptitle(plot_title, x=0.2, size=font_s)
    #plt.rcParams.update({'font.size': font_l})
    plt.tight_layout()

    # Plotting raw_data!
    plt.subplot(221)
    i = 0
    for ch in range(channels):
        plt.plot(raw_data[ch][:] + 15 * i, '-k', alpha=alpha, linewidth=0.7)
        i += 1
    for i in range(n_hline):
        plt.axvline(x=(i + 1) * (fs / 2), color="black", linestyle='dashed', alpha=alpha_dashed_line)
    plt.xticks([])
    #plt.xlabel('Time[s]', size=font_m)
    plt.ylabel('Offset [m]', size=font_size)
    plt.yticks(size=font_size)
    plt.title('Raw DAS Data', loc='left', size=font_size)

    # Plotting Denoised Data:
    plt.subplot(222)
    i = 0
    for ch in range(channels):
        plt.plot(denoised_data[ch][:] + 15 * i, '-k', alpha=alpha, linewidth=0.7)
        i += 1
    for i in range(n_hline):
        plt.axvline(x=(i + 1) * (fs / 2), color="black", linestyle='dashed', alpha=alpha_dashed_line)
    plt.xticks([])
    plt.title('Denoised DAS Data', loc='left', size=font_size)
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    plt.subplots_adjust(wspace=0.05)

    # plotting seismometer data 1
    seis_fs = seis_stats['sampling_rate']
    plt.subplot(223)
    plt.plot(seis_data, color='black', alpha = 0.8, linewidth=0.5)
    for i in range(n_hline):
        plt.axvline(x=(i + 1) * seis_fs / 2, color="black", linestyle='dashed', alpha=alpha_dashed_line)
    plt.xlabel('Time[s]', size=font_size)
    plt.xticks(np.arange(0, time_samples, 200), np.arange(0, time_samples, 200)/400, size=font_size)
    plt.ylabel('Seis Data', size=font_size)
    plt.yticks(size=font_size)
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    # plotting seismometer data 2
    plt.subplot(224)
    plt.plot(seis_data, color='black', alpha = 0.8, linewidth=0.5)
    for i in range(n_hline):
        plt.axvline(x=(i + 1) * seis_fs / 2, color="black", linestyle='dashed', alpha=alpha_dashed_line)
    plt.xlabel('Time[s]', size=font_size)
    plt.xticks(np.arange(0, time_samples, 200), np.arange(0, time_samples, 200)/400, size=font_size)
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    if saving_path==None:
        plt.show()
    else:
        plt.savefig(saving_path + '.png', bbox_inches="tight", pad_inches=0.2)

'''
Betrachten Eisbeben 90 und 106 von accumulation zone als wiggle plot mit cc gain geplottet denoist von 2 unterschiedlichen modellen
'''

# Choose id experiment:
id = 6
experiment = "08_combined480"

event_times = { "id": ["event_time",  "start_time", "duration", "start_channel", "end_channel", "category"],
                90: ["2020-07-06 19:10:54.0", "2020-07-06 19:10:53.0", 2.5, 20, 50, 2],
                6:  ["2020-07-06 20:31:20.0", "2020-07-06 20:31:21.5", 2.5, 10, 50, 1],
                84: ["2020-07-06 20:38:47.0", 3],
               106: ["2020-07-06 19:11:34.0", 2]}

start_channel = 10
end_channel = 50

# Data Paths:
raw_data_path = os.path.join("data", "raw_DAS", "0706/")
denoised1_data_path = os.path.join("experiments", experiment, "denoisedDAS", "0706/")
seismometer_path = os.path.join("data", "seismometer_test_data", "accumulation", "0706_AJP")

# Set Time:
event_time = event_times[id][0]
t_start = datetime.strptime(event_time, '%Y-%m-%d %H:%M:%S.%f') - timedelta(seconds=1.5)
t_end = t_start + timedelta(seconds=2.5)

# Load DAS Data:
raw_das_data, raw_headers, raw_axis = load_das_data(raw_data_path, t_start, t_end, raw=True)
denoised1_das_data, denoised1_headers, denoised1_axis = load_das_data(denoised1_data_path, t_start, t_end, raw=False)
raw_das_data = raw_das_data[start_channel:end_channel]
denoised1_das_data = denoised1_das_data[start_channel:end_channel]

# Load Seismometer Data:
seis_stream = read(seismometer_path + '/' + get_seismometer_event(id, seismometer_path))
seis_data = seis_stream[0].data
seis_stats = seis_stream[0].stats
seis_data = butter_bandpass_filter(seis_data, 1, 120, fs=seis_stats.sampling_rate, order=4)
seis_data = seis_data/np.std(seis_data)
seis_data = seis_data[600:1601]
print(seis_stats)
# TODO: OBACHT noch nicht in strain rate umgerechnet

print(raw_das_data.shape)
print(denoised1_das_data.shape)
print(seis_data.shape)

#plot_das_data(raw_das_data)
#plot_das_data(denoised1_das_data)

#saving_path=None
#saving_path="plots/single_waveforms/" + str(id) + "_sectionplot"
#plot_data(raw_das_data, denoised1_das_data, seis_data, seis_stats, saving_path, str(id))


''' Wiggle For Wiggle Comparison: '''

channel = 32 #20 good example 23 bad example
linewidth=0.8
fontsize = 12

fig, axs = plt.subplots(2, 1, figsize=(11, 4))

# first plot:
axs[0].plot(denoised1_das_data[channel], linewidth=linewidth, color='red', label="Denoised Data")
axs[0].plot(raw_das_data[channel], linewidth=linewidth, color='black', label="Raw DAS Data")
axs[0].set_yticks([])
axs[0].set_ylabel("Strain Rate [norm]", fontsize=fontsize-1)
axs[0].set_xticks([])
custom_lines = [Line2D([2], [0.5], color='red'),
                Line2D([2], [0.5], color='black'),
                Line2D([2], [0.5], color='green')]
plt.legend(custom_lines, ["Denoised DAS Data", "Raw DAS Data", "Seismometer Data"],
           fontsize=fontsize, frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0.1, 2.48))


# second plot:
axs[1].plot(denoised1_das_data[channel], linewidth=linewidth, color='red', label="Denoised Data")
axs[1].plot(seis_data*0.5, linewidth=linewidth, color='green', label="Seismometer Data")
axs[1].set_yticks([])
axs[1].set_ylabel("Strain Rate [norm]", fontsize=fontsize-1)
axs[1].set_xticks([0, 200, 400, 600, 800, 1000], [0, 0.5, 1, 1.5, 2, 2.5], fontsize=fontsize-2)
axs[1].set_xlabel("Time [s]", fontsize=fontsize)


plt.show()
#plt.savefig("plots/single_waveforms/" + str(id) + "_" + str(channel) + "_wigglecomparison.png")
'''
fig, axs = plt.subplots(1, len(event_names), figsize=(18, 4))

# Laden der Daten
data = np.load(os.path.join(data_path, event_name))[channel, t_start:t_end]
denoised_data = np.load(os.path.join(denoised_data_path, denoised_event_names[i]))[channel, t_start:t_end]

# Plot Data
axs[i].plot(data, color="black", linewidth=1, label="Noise Coruppted")
axs[i].plot(denoised_data, color="red", linewidth=1, label="Denoised")

# Achses
axs[i].set_ylim([-22, 22])
axs[i].set_ylim([-22, 22])
axs[i].set_yticks([])
if i == 0:
    axs[i].set_ylabel("Strain Rate [norm]", fontsize=fontsize)
    axs[i].legend(fontsize=fontsize-2, loc="lower left")

axs[i].set_xticks([0, 100, 200], [0, 0.25, 0.5], fontsize=fontsize)
axs[i].set_xlabel("Time [s]", fontsize=fontsize)


# Beschriftungen
#letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
#snr_values = ["No Noise Added", "SNR: 10", "SNR: 3.2", "SNR: 1.0", "SNR: 0.3"]
#letter_params = {
#    "fontsize": fontsize,
#    "verticalalignment": "top",
#    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
#}

#axs[i].text(x=0.0, y=1.0, transform=axs[i].transAxes, s=letters[i], **letter_params)
#axs[i].text(x=0.5, y=1.03, transform=axs[i].transAxes, s=snr_values[i], fontsize=fontsize + 2, ha="center")

plt.tight_layout()
plt.savefig("plots/synthetics/seis_wiggle_comparison.png")
#plt.show()
'''