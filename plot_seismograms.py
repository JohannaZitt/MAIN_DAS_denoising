
import os
from scipy.signal import butter, lfilter
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


'''

1. make sure which events you wanna look at:
    events on ablation zone with and without icequake 
    events on accumulation zone with and without icequake
    events where denoising failed
    gleiche plots für wiggle for wiggle comparison 
2. load raw das data and preprocess it like for denoising (bandpassfilter, downsample, normalize)
3. load denoised data (no preprocessing)
4. plot frequency for one channel
5. plot multi-channel frequency.


'''

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
        #data[:, i] = data[:, i] / np.abs(data[:, i]).max()

    return data.T, headers, axis

def plot_frequency_content_single_channel(raw_f_frq, raw_f_amp, denoised_f_frq, denoised_f_amp, font_size = 15, show_plot = True):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # First Subplot:
    ax1.plot(raw_f_frq, raw_f_amp, "k-")
    ax1.set_xlabel("Frequency [Hz]", fontsize=font_size)
    ax1.set_ylabel("Amplitude", fontsize=font_size)
    ax1.set_title("Noisy", fontsize=font_size + 4)

    # Second Subplot:
    ax2.plot(denoised_f_frq, denoised_f_amp, "k-")
    ax2.set_yticks([])
    ax2.set_xlabel("Frequency [Hz]", fontsize=font_size)
    ax2.set_title("Denoised", fontsize=font_size + 4)

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        #plot_name = ""
        #plt.savefig("plots/frequency_content/" + plot_name + ".pdf", dpi=400)
        plt.close()

def plot_frequency_content_multiple_channel(raw_data_freq, denoised_data_freq, raw_data, font_size = 15, show_plot = True, cc_spacing = 12, cmap = "plasma", aspect="auto", vmin=0, vmax=150):

    extent = (0, raw_data.shape[0] * cc_spacing, 0, 250)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [4, 5]})

    im1 = ax1.imshow(raw_data_freq[:, ::-1].T, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    ax1.set_title("Raw", fontsize=font_size + 4)
    ax1.set_ylabel("Frequency [Hz]", fontsize=font_size)
    ax1.set_xlabel("Distance [m]", fontsize=font_size)

    im2 = ax2.imshow(denoised_data_freq[:, ::-1].T, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    ax2.set_title("Denoised", fontsize=font_size + 4)
    ax2.set_xlabel("Distance [m]", fontsize=font_size)
    ax2.set_yticks([])
    cbar = fig.colorbar(im2, ax=ax2)
    cbar.set_label("Ampl. spectrum [[m/m]/s]", fontsize=font_size)

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_seismogram_single_channel(raw_data, denoised_data, middle_channel, font_size=15, show_plot = True, NFFT=16, Fs=400, noverlap=8, cmap="plasma", scale_by_freq=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.specgram(raw_data[middle_channel], NFFT=NFFT, Fs=Fs, noverlap=noverlap, cmap=cmap, scale_by_freq=scale_by_freq)
    ax1.set_title("Raw", fontsize=font_size + 4)
    ax1.set_ylabel("Frequency [Hz]", fontsize=font_size)
    ax1.set_xlabel("Time [s]", fontsize=font_size)

    ax2.specgram(denoised_data[middle_channel], NFFT=NFFT, Fs=Fs, noverlap=noverlap, cmap=cmap, scale_by_freq=scale_by_freq)
    ax2.set_title("Denoised", fontsize=font_size + 4)
    ax2.set_xlabel("Time [s]", fontsize=font_size)
    ax2.set_yticks([])

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        plt.close()



# ID : [starttime, channel middle, channel delta, category, closts seismometer]
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

ids = [5, 20, 82]#[5, 20, 82]

for i, id in enumerate(ids):


    event_time = event_times[id][0]
    t_start = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S.%f")
    t_end = t_start + timedelta(seconds=2)
    middle_channel = event_times[id][1]
    fs = 400

    # load DAS data:
    raw_data, raw_headers, raw_axis = load_das_data(raw_path, t_start, t_end,
                                                    raw=True,
                                                    channel_delta_start=event_times[id][1],
                                                    channel_delta_end=event_times[id][2])
    denoised_data, denoised1_headers, denoised1_axis = load_das_data(denoised_path, t_start, t_end,
                                                                     raw=False,
                                                                     channel_delta_start=event_times[id][1],
                                                                     channel_delta_end=event_times[id][2])



    ################################################
    # Calculating frequency content of one channel #
    ################################################
    # rfft = real Fast Fourier Transform for one channel -> converting to spectral domain
    # abs = absolute values of the complex numbers of fft, which represents the amplitude of each frequency component
    # raw_f_amp = the amlplitudes spectrum of one channel
    # 1/fs computes time interval between successive samples
    # raw_f_frq holds the array of frequency bins corrsponging to the amplitudes in raw_f_amp
    raw_f_amp = np.abs(np.fft.rfft(raw_data[middle_channel])) #shape = (401,)
    raw_channel_size = raw_data[middle_channel].size # = 801 data points in time
    raw_f_frq = np.fft.rfftfreq(raw_channel_size, 1./fs) # shape = (401, )
    denoised_f_amp = np.abs(np.fft.rfft(denoised_data[middle_channel]))
    denoised_size = denoised_data[middle_channel].size
    denoised_f_frq = np.fft.rfftfreq(denoised_size, 1. / fs)

    #######################################################
    # Calculating frequency content for multiple channels #
    #######################################################
    raw_data_freq = np.zeros((raw_data.shape[0], raw_f_amp.shape[0]))
    for i in range(raw_data.shape[0]):
        raw_data_freq[i] = np.abs(np.fft.rfft(raw_data[i]))
    raw_data_freq = raw_data_freq[:, 0:250]
    denoised_data_freq = np.zeros((denoised_data.shape[0], denoised_f_amp.shape[0]))
    for i in range(denoised_data.shape[0]):
        denoised_data_freq[i] = np.abs(np.fft.rfft(denoised_data[i]))
    denoised_data_freq = denoised_data_freq[:, 0:250]

    ##############################
    # Plotting frequency content #
    ##############################
    plot_frequency_content_single_channel(raw_f_frq, raw_f_amp, denoised_f_frq, denoised_f_amp,
                                          font_size=15,
                                          show_plot=False)
    plot_frequency_content_multiple_channel(raw_data_freq, denoised_data_freq, raw_data,
                                            font_size=15,
                                            show_plot=False,
                                            cc_spacing=12,
                                            cmap="plasma",
                                            aspect="auto",
                                            vmin=0, vmax=150)


    plot_seismogram_single_channel(raw_data, denoised_data, middle_channel,
                                   font_size=15,
                                   show_plot=True,
                                   NFFT=32,
                                   Fs=400,
                                   noverlap=16,
                                   cmap="plasma",
                                   scale_by_freq=True)








    # Plotting spectogram of one channel.
    #plt.specgram(raw_data[middle_channel], NFFT=256, Fs=400, noverlap=128, cmap="plasma")
    #plt.colorbar()
    #plt.show()

    #plt.specgram(denoised_data[middle_channel], NFFT=256, Fs=400, noverlap=128, cmap="plasma")
    #plt.colorbar()
    #plt.show()













