from datetime import datetime, timedelta
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy import read
from obspy.signal.cross_correlation import correlate, xcorr_max
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from scipy.signal import butter, lfilter


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

def resample(data, ratio):
    res = np.zeros((int(data.shape[0]/ratio) + 1, data.shape[1]))
    for i in range(data.shape[1]):
        res[:,i] = np.interp(np.arange(0, len(data), ratio), np.arange(0, len(data)), data[:,i])
    return res

def xcorr(x, y):
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

def compute_xcorr_window(x):
    Nch = x.shape[0]
    Cxy = np.zeros((Nch, Nch)) * np.nan

    for i in range(Nch):
        for j in range(i):
            Cxy[i, j] = xcorr(x[i], x[j])

    return np.nanmean(Cxy)

def compute_moving_coherence(data, bin_size):
    N_ch = data.shape[0]

    cc = np.zeros(N_ch)

    for i in range(N_ch):
        start = max(0, i - bin_size // 2)
        stop = min(i + bin_size // 2, N_ch)
        ch_slice = slice(start, stop)
        cc[i] = compute_xcorr_window(data[ch_slice])

    return cc

def get_middel_channel(receiver):
    channel = 0
    if receiver == '_AKU':
        channel = 3740
    elif receiver == '_AJP':
        channel = 3460
    elif receiver == '_ALH':
        channel = 3842
    elif receiver == 'RA82':
        channel = 1300
    elif receiver == 'RA87':
        channel = 1230
    elif receiver == 'RA88':
        channel = 1460
    else:
        print('There is no start nor end channel for receiver ' + receiver + '.')

    channel = int(channel/4)
    return channel

def load_das_data(folder_path, t_start, t_end, receiver, raw):

    # 1. load data
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=folder_path, convert=False)
    data = data.astype('f')

    # 2. downsample data in space:
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 :
            data = data[:,::4]
        else:
            data = data[:, ::2]
        headers['dx'] = 8

    # get start and end channel:
    ch_middel = get_middel_channel(receiver)

    # 3. cut to size
    data = data[:, ch_middel-15:ch_middel+15]


    if raw:
        # 4. bandpasfilter and normalize
        for i in range(data.shape[1]):
            butter_bandpass_filter(data[:,i], 1, 120, fs=headers['fs'], order=4)
            #data[:,i] = data[:,i] / data[:,i].std()

        # 5. resample time
        data = resample(data, headers['fs']/400)
        headers['fs'] = 400

    std = data[:][:].std()
    for i in range(data.shape[0]):
        data[i][:] /= std



    return data, headers, axis

def plot_data(raw_data, denoised_data, seis_data, seis_stats, data_type, saving_path):

    # different fonts:
    font_s = 12
    font_m = 14
    font_l = 16

    # parameters:
    channels = raw_data.shape[0]
    fs = 400
    normalize_trace = True
    alpha = 0.7
    alpha_dashed_line = 0.2
    plot_title = data_type + ', ' + str(seis_stats['starttime']) + ' - ' + str(seis_stats['endtime']) + ', ' + str(seis_stats['station'])

    fig, ax = plt.subplots(2, 2, figsize=(20, 12), gridspec_kw={'height_ratios': [5, 1]})
    fig.suptitle(plot_title, x=0.2, size=font_s)
    plt.rcParams.update({'font.size': font_l})
    plt.tight_layout()

    # Plotting raw_data!
    plt.subplot(221)
    i = 0
    for ch in range(channels):
        plt.plot(raw_data[ch][:] + 8 * i, '-k', alpha=alpha)
        i += 1
    for i in range(11):
        plt.axvline(x=(i + 1) * (fs / 2), color="black", linestyle='dashed', alpha=alpha_dashed_line)
    plt.xticks(np.arange(0, 2401, 200), np.arange(0, 6.1, 0.5), size=font_s)
    plt.xlabel('Time[s]', size=font_m)
    plt.ylabel('Offset [m]', size=font_m)
    plt.yticks(size=font_s)
    plt.title('Raw DAS Data', loc='left')

    # Plotting Denoised Data:
    plt.subplot(222)
    i = 0
    for ch in range(channels):
        plt.plot(denoised_data[ch][:] + 8 * i, '-k', alpha=alpha)
        i += 1
    for i in range(11):
        plt.axvline(x=(i + 1) * (fs / 2), color="black", linestyle='dashed', alpha=alpha_dashed_line)
    plt.xticks(np.arange(0, 2401, 200), np.arange(0, 6.1, 0.5), size=font_s)
    plt.title('Denoised DAS Data', loc='left')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    plt.subplots_adjust(wspace=0.05)

    # plotting seismometer data 1
    seis_fs = seis_stats['sampling_rate']
    plt.subplot(223)
    plt.plot(seis_data, color='black', alpha=0.4)
    for i in range(11):
        plt.axvline(x=(i + 1) * seis_fs / 2, color="black", linestyle='dashed', alpha=alpha_dashed_line)
    plt.xlabel('Time[s]', size=font_m)
    if seis_fs == 500:
        plt.xticks(np.arange(0, 3001, 250), np.arange(0, 6.1, 0.5), size=font_s)
    else:  # if seis_fs==400
        plt.xticks(np.arange(0, 2401, 200), np.arange(0, 6.1, 0.5), size=font_s)
    plt.ylabel('Seismometer Data', size=font_l)
    plt.yticks(size=font_s)
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    # plotting seismometer data 2
    plt.subplot(224)
    plt.plot(seis_data, color='black', alpha=0.4)
    for i in range(11):
        plt.axvline(x=(i + 1) * seis_fs / 2, color="black", linestyle='dashed', alpha=alpha_dashed_line)
    plt.xlabel('Time[s]', size=font_m)
    if seis_fs == 500:
        plt.xticks(np.arange(0, 3001, 250), np.arange(0, 6.1, 0.5), size=font_s)
    else:  # if seis_fs==400
        plt.xticks(np.arange(0, 2401, 200), np.arange(0, 6.1, 0.5), size=font_s)
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    if saving_path==None:
        plt.show()
    else:
        plt.savefig(saving_path + '.png')


experiments = os.listdir('experiments/')
experiments = ['06_surface']
#experiments = experiments[0:1]
data_types = ['stick-slip_ablation', 'stick-slip_accumulation', 'surface_ablation', 'surface_accumulation']
data_types = data_types[1:2]

# for saving the data hdf5 format is used:
with h5py.File('evaluation/cc_gain_naive.h5', 'w') as cc_gain_h5:

    # for every experiment
    for experiment in experiments:

        experiment_group = cc_gain_h5.create_group(experiment)

        print('############################################################################')
        print('############################################################################')
        print('NEW EXPERIMENT: ', experiment)
        print('############################################################################')
        print('############################################################################')

        # for every data type
        for data_type in data_types:
            print('############################################################################')
            print('############################################################################')
            print('NEW DATA TYPE: ', data_type)
            print('############################################################################')
            print('############################################################################')

            data_type_group = experiment_group.create_group(data_type)

            seismometer_data_path = 'data/test_data/' + data_type

            seismometer_events = os.listdir(seismometer_data_path)
            #seismometer_events = seismometer_events[0:1]

            # for every seismometer event
            for seismometer_event in seismometer_events:

                print('############################################################################')
                print('New Seismometer Event: ', seismometer_event)
                print('############################################################################')


                seismometer_event_group = data_type_group.create_group(seismometer_event)

                event_time = seismometer_event[-18:-10]
                event_date = seismometer_event[-29:-19]
                receiver = seismometer_event[-34:-30]

                # evaluation time window:
                t_start = datetime.strptime(event_date + ' ' + event_time + '.0', '%Y-%m-%d %H:%M:%S.%f')
                t_start = t_start - timedelta(seconds=3)
                t_end = t_start + timedelta(seconds=6)

                # load seismometer data:
                seis_stream = read(seismometer_data_path + '/' + seismometer_event)
                seis_data = seis_stream[0].data
                seis_stats = seis_stream[0].stats

                # load raw DAS data:
                raw_folder_path = 'data/raw_DAS/' + data_type + '/'
                raw_data, raw_headers, raw_axis = load_das_data(folder_path =raw_folder_path, t_start = t_start, t_end = t_end, receiver = receiver, raw = True)

                # load denoised DAS data
                denoised_folder_path = 'experiments/' + experiment + '/denoisedDAS_mean0/' + data_type + '/'
                denoised_data, denoised_headers, denoised_axis = load_das_data(folder_path =denoised_folder_path, t_start = t_start, t_end = t_end, receiver = receiver, raw = False)

                # plotting data:
                saving_path = os.path.join('experiments', experiment, 'plots_mean0', data_type)
                if not os.path.isdir(saving_path):
                    os.makedirs(saving_path)
                saving_path += '/' + seismometer_event
                # when the plot should be depicted, set
                # saving_path = None
                plot_data(raw_data.T, denoised_data.T, seis_data, seis_stats, data_type, saving_path)


                # calculate CC
                cc_t_start = 688
                cc_t_end = cc_t_start + 1024
                bin_size = 11


                raw_cc = compute_moving_coherence(raw_data.T, bin_size)
                denoised_cc = compute_moving_coherence(denoised_data.T, bin_size)
                cc_gain = denoised_cc / raw_cc
                # save CC gain
                seismometer_event_group.create_dataset('data_cc', data=cc_gain)


                # calculate crosscorrelation between seismometer data and raw_data and denoised data:
                shift = 20
                raw_cc_seis = np.zeros((raw_data.shape[1], (shift * 2)+1))
                denoised_cc_seis = np.zeros((raw_data.shape[1], (shift * 2)+1))
                for i in range(raw_data.shape[1]):
                    raw_cc_seis[i] = correlate(seis_data[cc_t_start:cc_t_end], raw_data[cc_t_start:cc_t_end][i], shift,
                                   normalize='naive', method='fft')
                    denoised_cc_seis[i] = correlate(seis_data[cc_t_start:cc_t_end], denoised_data[cc_t_start:cc_t_end][i], shift,
                                            normalize='naive', method='fft')


                # save CC gain
                #seismometer_event_group.create_dataset('data_cc_seis_raw', data=raw_cc_seis)
                #seismometer_event_group.create_dataset('data_cc_seis_denoised', data=denoised_cc_seis)

