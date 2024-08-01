
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

def plot_frequency_content_single_channel(raw_f_frq, raw_f_amp, denoised_f_frq, denoised_f_amp, id, font_size = 15, show_plot = True):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # First Subplot:
    ax1.plot(raw_f_frq, raw_f_amp, "k-")
    ax1.set_xlabel("Frequency [Hz]", fontsize=font_size)
    ax1.set_ylabel("Amplitude", fontsize=font_size)
    ax1.set_title("Noisy", fontsize=font_size + 4)
    ax1.set_ylim(0, 120)

    # Second Subplot:
    ax2.plot(denoised_f_frq, denoised_f_amp, "k-")
    ax2.set_yticks([])
    ax2.set_xlabel("Frequency [Hz]", fontsize=font_size)
    ax2.set_title("Denoised", fontsize=font_size + 4)
    ax2.set_ylim(0, 120)

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        plot_name = str(id) + "_fcontent_single_channel.pdf"
        plt.savefig("plots/spectograms/fcontent_single_channel/" + plot_name, dpi=400)
        plt.close()

def plot_frequency_content_multiple_channel(raw_data_freq, denoised_data_freq, raw_data, id, font_size = 15, show_plot = True, cc_spacing = 12, cmap = "plasma", aspect="auto", vmin=0, vmax=150):

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
        plot_name = str(id) + "_fcontent_multiple_channel.pdf"
        plt.savefig("plots/spectograms/fcontent_multiple_channel/" + plot_name, dpi=400)
        plt.close()

def plot_seismogram_single_channel(raw_data, denoised_data, middle_channel, id, font_size=15, show_plot = True, NFFT=32, Fs=400, noverlap=16, cmap="plasma", scale_by_freq=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [4, 5]})

    ax1.specgram(raw_data[middle_channel], NFFT=NFFT, Fs=Fs, noverlap=noverlap, cmap=cmap, scale_by_freq=scale_by_freq)
    ax1.set_title("Raw", fontsize=font_size + 4)
    ax1.set_ylabel("Frequency [Hz]", fontsize=font_size)
    ax1.set_xlabel("Time [s]", fontsize=font_size)

    spec2=ax2.specgram(denoised_data[middle_channel], NFFT=NFFT, Fs=Fs, noverlap=noverlap, cmap=cmap, scale_by_freq=scale_by_freq)
    ax2.set_title("Denoised", fontsize=font_size + 4)
    ax2.set_xlabel("Time [s]", fontsize=font_size)
    ax2.set_yticks([])
    cbar = fig.colorbar(spec2[3], ax=ax2)
    cbar.set_label("Power Spectral Density [s⁻²/Hz][dB]") #s=seconds

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        plot_name = str(id) + "_seismogram_single_channel.pdf"
        plt.savefig("plots/spectograms/seismogram_single_channel/" + plot_name, dpi=400)
        plt.close()



# ID : [starttime, channel middle, channel delta, category, closts seismometer]
# Category 1: improved SNR
# Category 2: new event detected
# Category 3: worsening of an icequake through denoising
# Category 4: no event
event_times = {0: ["2020-07-27 08:17:34.6", 60, 70, 1, "ALH"], # Category 1, Receiver ALH
               5: ["2020-07-27 19:43:30.5", 45, 75, 1, "ALH"], # Category 1, Receiver ALH (plotted in paper)
               11: ["2020-07-27 19:43:01.4", 65, 85, 1, "ALH"], # Category 1, Receiver ALH
               35: ["2020-07-27 03:03:20.2", 65, 55, 1, "ALH"], # Category 1, Receiver ALH (dominant frequency at 100 Hz)
               83: ["2020-07-27 01:03:00.2", 70, 25, 1, "ALH"], # Category 1, Receiver ALH

               20: ["2020-07-27 00:21:46.3", 30, 30, 2, "ALH"], # Category 2, Receiver ALH (plotted in paper)
               24: ["2020-07-27 05:21:48.4", 70, 90, 2, "ALH"], # Category 2, Receiver ALH (dominant frequency at 100 Hz)
               36: ["2020-07-27 20:47:34.8", 40, 45, 2, "ALH"], # Category 2, Receiver ALH
               52: ["2020-07-27 20:00:30.4", 100, 85, 2, "ALH"], # Category 2, Receiver ALH
               67: ["2020-07-27 23:17:54.0", 70, 35, 2, "ALH"], # Category 2, Receiver ALH
               107: ["2020-07-27 01:25:19.6", 30, 15, 2, "ALH"], # Category 2, Receiver ALH

               82: ["2020-07-27 05:04:55.8", 80, 150, 3, "ALH"], # Category 3, Receiver ALH (plotted in paper) (dominant frequency at 100 Hz)

               }
experiment = "03_accumulation_horizontal"

raw_path = os.path.join("data", "raw_DAS/")
denoised_path = os.path.join("experiments", experiment, "denoisedDAS/")

# real-world samples in paper: [5, 20, 82]
ids = [0, 5, 11, 35, 83, 20, 24, 36, 52, 67, 107, 82]

for id in ids:


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

    ###############################
    # Plot waveform for reference #
    ###############################
    #channels=denoised_data.shape[0]
    #for ch in range(channels):
    #    plt.plot(denoised_data[-ch][:] + 1.5 * i, "-k", alpha=0.7)
    #    i += 1
    #plt.show()

    #plt.plot(raw_data[middle_channel], color="green")
    #plt.plot(denoised_data[middle_channel], color="pink")
    #plt.show()

    ##############################
    # Plotting frequency content #
    ##############################
    plot_frequency_content_single_channel(raw_f_frq, raw_f_amp, denoised_f_frq, denoised_f_amp,
                                          font_size=15,
                                          show_plot=False,
                                          id=id)
    plot_frequency_content_multiple_channel(raw_data_freq, denoised_data_freq, raw_data,
                                            font_size=15,
                                            show_plot=False,
                                            cc_spacing=12,
                                            cmap="plasma",
                                            aspect="auto",
                                            vmin=0, vmax=150,
                                            id=id)
    plot_seismogram_single_channel(raw_data, denoised_data, middle_channel,
                                   font_size=15,
                                   show_plot=False,
                                   NFFT=32,
                                   Fs=400,
                                   noverlap=16,
                                   cmap="plasma",
                                   scale_by_freq=True,
                                   id=id)














