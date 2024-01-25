import re
from datetime import datetime, timedelta
import numpy as np
import h5py
import os
import csv
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
    try:
        res = np.zeros((int(data.shape[0]/ratio) + 1, data.shape[1]))
        for i in range(data.shape[1]):

            res[:,i] = np.interp(np.arange(0, len(data), ratio), np.arange(0, len(data)), data[:,i])
    except ValueError as e:
        res = np.zeros((int(data.shape[0] / ratio), data.shape[1]))
        for i in range(data.shape[1]):
            res[:, i] = np.interp(np.arange(0, len(data), ratio), np.arange(0, len(data)), data[:, i])
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
    if receiver == 'AKU':
        channel = 3740
    elif receiver == 'AJP':
        channel = 3460
    elif receiver == 'ALH':
        channel = 3842
    elif receiver == 'RA82':
        channel = 1300
    elif receiver == 'RA87':
        channel = 1230
    elif receiver == 'RA88':
        channel = 1615 # 1600
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
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::4]
        else:
            data = data[:, ::2]
        headers['dx'] = 8


    # 3. cut to size
    ch_middel = get_middel_channel(receiver)  # get start and end channel:
    data = data[:, ch_middel-20:ch_middel+20]

    if raw:
        # 4. downsample in time
        data = resample(data, headers['fs'] / 400)
        headers['fs'] = 400

    # 5. bandpasfilter and normalize
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers['fs'], order=4)
        data[:,i] = data[:,i] / np.abs(data[:,i]).max()

    return data, headers, axis

def plot_data(raw_data, denoised_data, seis_data, seis_stats, data_type, saving_path, id):

    # different fonts:
    font_s = 12
    font_m = 14
    font_l = 16

    # parameters:
    channels = raw_data.shape[0]
    fs = 400
    alpha = 0.7
    alpha_dashed_line = 0.2
    plot_title = id + ': ' + data_type + ', ' + str(seis_stats['starttime']) + ' - ' + str(seis_stats['endtime']) + ', ' + str(seis_stats['station'])

    fig, ax = plt.subplots(2, 2, figsize=(20, 12), gridspec_kw={'height_ratios': [5, 1]})
    fig.suptitle(plot_title, x=0.2, size=font_s)
    plt.rcParams.update({'font.size': font_l})
    plt.tight_layout()

    # Plotting raw_data!
    plt.subplot(221)
    i = 0
    for ch in range(channels):
        plt.plot(raw_data[ch][:] + 1.5 * i, '-k', alpha=alpha)
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
        plt.plot(denoised_data[ch][:] + 1.5 * i, '-k', alpha=alpha)
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


#experiments = os.listdir('experiments/')
experiments = ['04_accumulation_vertical', '07_combined120']
#experiments = ['01_ablation_horizontal']
data_types = ['ablation/0706_RA88']

for experiment in experiments: # for every experiment

    for data_type in data_types:  # for every data type

        seis_data_path = 'data/seismometer_test_data/' + data_type
        seismometer_events = os.listdir(seis_data_path)
        seismometer_events = seismometer_events[2:]

        with open('experiments/' + experiment +'/cc_evaluation_' + experiment[:2] + '.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['id', 'mean_cc_gain', 'mean_cross_gain', 'zone', 'model'])

            # es wird folgendes abgespeichert:
            # id: id des events, wie auch in experiments/model_name/plots zu finden ist
            # mean_cc_gain: local waveform coherence: es gibt für jedes event n (= Anzahl channel) viele gg_gain Werte.
            #               hier von wird der Mittelwert berechnet und abgespeichert
            # mean_cross_gain: hier wird crosscorrelation zwischen (raw und seismometer) und (denoised und seismometer) berechnet
            # und der crosscorrelation gain berechnet. Der Mittwelwert über alle channel wird abgespeichert.
            # zone: ablation or accumulation zone
            # model: model mit welchem die denoisten Werte denoist wurde.


            # for every seismometer event
            for seismometer_event in seismometer_events:
                print('SEISMOMETER EVENT: ', seismometer_event)
                if data_type[:2] == 'ab':
                    receiver = seismometer_event[-14:-10]
                    event_time = seismometer_event[-23:-15]
                    event_date = seismometer_event[-34:-24]
                    id = ''
                elif data_type[:2] == 'ac':
                    receiver = seismometer_event[-12:-9]
                    event_time = seismometer_event[-23:-15]
                    event_date = seismometer_event[-34:-24]
                else:
                    print('ERROR: No matching data type')


                # pick time window:
                t_start = datetime.strptime(event_date + ' ' + event_time + '.0', '%Y-%m-%d %H:%M:%S.%f')
                t_start = t_start - timedelta(seconds=3)
                t_end = t_start + timedelta(seconds=6)

                # load seismometer data:
                seis_stream = read(seis_data_path + '/' + seismometer_event)
                seis_data = seis_stream[0].data
                seis_stats = seis_stream[0].stats
                seis_data = butter_bandpass_filter(seis_data, 1, 120, fs=seis_stats.sampling_rate, order=4)

                # load raw DAS data:
                raw_folder_path = 'data/raw_DAS/0706/'
                raw_data, raw_headers, raw_axis = load_das_data(folder_path =raw_folder_path, t_start = t_start, t_end = t_end, receiver = receiver, raw = True)

                # load denoised DAS data
                denoised_folder_path = 'experiments/' + experiment + '/denoisedDAS/0706/'
                denoised_data, denoised_headers, denoised_axis = load_das_data(folder_path =denoised_folder_path, t_start = t_start, t_end = t_end, receiver = receiver, raw = False)

                saving_path = os.path.join('experiments', experiment, 'plots', data_type)
                if not os.path.isdir(saving_path):
                    os.makedirs(saving_path)

                saving_path += '/' + seismometer_event  # when the plot should be depicted, set saving_path = None
                #saving_path = None

                id = re.search(r'ID:(\d+)_', seismometer_event).group(1)
                if saving_path is None or not os.path.exists(saving_path + '.png'):
                    print('Event wird geplottet')
                    plot_data(raw_data.T, denoised_data.T, seis_data, seis_stats, data_type, saving_path, id)
                else:
                    print('Event wurde bereits geplottet')



                # calculate CC
                cc_t_start = 688
                cc_t_end = cc_t_start + 1024
                bin_size = 11

                raw_cc = compute_moving_coherence(raw_data.T, bin_size)
                denoised_cc = compute_moving_coherence(denoised_data.T, bin_size)
                cc_gain = denoised_cc / raw_cc

                # calculate Crosscorrelation beween DAS and seismometer data
                # TODO: das hier nochmal überprüfen, ob das seine Richtigkeit hat..
                shift = 20
                raw_cc_seis = np.zeros((raw_data.shape[1], (shift * 2) + 1))
                denoised_cc_seis = np.zeros((raw_data.shape[1], (shift * 2) + 1))
                for i in range(raw_data.shape[1]):
                    raw_cc_seis[i] = correlate(seis_data[cc_t_start:cc_t_end], raw_data[cc_t_start:cc_t_end][i], shift,
                                               normalize='naive', method='fft')
                    denoised_cc_seis[i] = correlate(seis_data[cc_t_start:cc_t_end],
                                                    denoised_data[cc_t_start:cc_t_end][i], shift,
                                                    normalize='naive', method='fft')


                # mean_cc_gain und mean_cross_gain abspeichern:
                writer.writerow([id, round(cc_gain.mean(), 5), 'cross_gain', data_type, experiment])








                '''
    
        
    
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
                
                '''

