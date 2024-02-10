import re
from datetime import datetime, timedelta
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from obspy import read
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
        #data[:,i] = data[:,i] / np.abs(data[:,i]).max()
        data[:, i] = data[:,i] / np.std(data[:, i])

    return data, headers, axis

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


#experiments = os.listdir('experiments/')
experiments = ["03_accumulation_horizontal", "04_accumulation_vertical", "10_random_borehole"]
data_types = ['accumulation/0706_AJP', 'ablation/0706_RA88']

for experiment in experiments: # for every experiment

    with open('experiments/' + experiment + '/cc_evaluation_' + experiment[:2] + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['id', 'mean_cc_gain', 'mean_cross_gain', 'zone', 'model'])

        for data_type in data_types:  # for every data type

            seis_data_path = 'data/seismometer_test_data/' + data_type
            seismometer_events = os.listdir(seis_data_path)

            '''
            
            id:                 id des events, wie auch in experiments/model_name/plots zu finden ist
            mean_cc_gain:       local waveform coherence: es gibt für jedes event n (= Anzahl channel) viele gg_gain Werte.
                                hier von wird der Mittelwert berechnet und abgespeichert
            mean_cross_gain:    hier wird crosscorrelation zwischen (raw und seismometer) und (denoised und seismometer) berechnet
                                und der crosscorrelation gain berechnet. Der Mittwelwert über alle channel wird abgespeichert.
            zone:               ablation or accumulation zone
            model:              model mit welchem die denoisten Werte denoist wurde.
            '''

            # for every seismometer event
            for seismometer_event in seismometer_events:
                print("SEISMOMETER EVENT: ", seismometer_event)

                event_time = seismometer_event[-23:-15]
                event_date = seismometer_event[-34:-24]
                id = re.search(r'ID:(\d+)', seismometer_event).group(1)
                receiver = ''
                zone = ''

                if data_type[:2] == "ab":
                    receiver = seismometer_event[-14:-10]
                    zone = 'ablation'
                elif data_type[:2] == "ac":
                    receiver = seismometer_event[-12:-9]
                    zone = 'accumulation'
                else:
                    print("ERROR: No matching data type")


                # pick time window:
                t_start = datetime.strptime(event_date + ' ' + event_time + '.0', '%Y-%m-%d %H:%M:%S.%f')
                t_start = t_start - timedelta(seconds=3)
                t_end = t_start + timedelta(seconds=6)

                # load seismometer data:
                seis_stream = read(seis_data_path + '/' + seismometer_event)
                seis_data = seis_stream[0].data
                seis_stats = seis_stream[0].stats

                # perform resampling if sample frequency is 500 Hz
                if seis_stats.sampling_rate == 500:
                    seis_data = np.interp(np.arange(0, len(seis_data), 500/400), np.arange(0, len(seis_data)), seis_data)
                    seis_stats.sampling_rate = 400.0
                    seis_stats.npts = seis_data.shape[0]
                # filter and normalize seismometer data:
                seis_data = butter_bandpass_filter(seis_data, 1, 120, fs=seis_stats.sampling_rate, order=4)
                seis_data = seis_data / np.std(seis_data)

                # load raw DAS data:
                raw_folder_path = 'data/raw_DAS/0706/'
                raw_data, raw_headers, raw_axis = load_das_data(folder_path =raw_folder_path, t_start = t_start, t_end = t_end, receiver = receiver, raw = True)

                # load denoised DAS data
                denoised_folder_path = 'experiments/' + experiment + '/denoisedDAS/0706/'
                denoised_data, denoised_headers, denoised_axis = load_das_data(folder_path =denoised_folder_path, t_start = t_start, t_end = t_end, receiver = receiver, raw = False)


                '''
                Calculate CC Gain
                '''

                t_window_start = 688
                t_window_end = t_window_start + 1024
                bin_size = 11

                raw_cc = compute_moving_coherence(raw_data[t_window_start:t_window_end].T, bin_size)
                denoised_cc = compute_moving_coherence(denoised_data[t_window_start:t_window_end].T, bin_size)
                cc_gain = denoised_cc / raw_cc


                '''
                Calculate CC between DAS and Seismometer: 
                '''
                raw_cc_seis_total = []
                denoised_cc_seis_total = []

                for i in range(raw_data.shape[1]):
                    raw_cc_seis = xcorr(raw_data.T[i], seis_data)
                    denoised_cc_seis = xcorr(denoised_data.T[i], seis_data)
                    raw_cc_seis_total.append(raw_cc_seis)
                    denoised_cc_seis_total.append(denoised_cc_seis)

                cc_gain_seis = np.array(denoised_cc_seis_total) / np.array(raw_cc_seis_total)

                # Save values:
                #writer.writerow(
                #    [id, cc_gain.mean(), cc_gain_seis.mean(), zone, experiment])



