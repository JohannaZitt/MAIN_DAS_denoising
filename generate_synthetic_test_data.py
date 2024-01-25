import os
import numpy as np
from obspy import read
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from datetime import datetime, timedelta
from obspy import UTCDateTime
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5

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
        print('in if')
        data = resample(data, headers['fs'] / 400)
        headers['fs'] = 400

    # 5. bandpasfilter and normalize
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers['fs'], order=4)
        data[:,i] = data[:,i] / np.abs(data[:,i]).max()

    return data, headers, axis

def plot_das_data(data):
    channels = data.shape[0]
    alpha=0.7
    i = 0
    plt.figure(figsize=(20, 12))
    for ch in range(channels):
        plt.plot(data[ch][:] + 1.5 * i, '-k', alpha=alpha)
        i += 1
    plt.show()

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

def compute_shift(gauge = 10, slowness_max = True, fs = 400):
    if slowness_max:
        slowness = 1 / 1650
    else:
        slowness = 1 / 3900

    shift = gauge * slowness * fs
    return int(shift)


fs = 400
lowcut = 1
highcut = 120
shift = compute_shift()
n_channel = 50
SNR_values = [0, 1, 2, 3, 4]


'''
GENERATE SYNTHETIC DATA FROM DAS DATA:
'''

das_folder_path = 'data/raw_DAS/0706/'
event_times = ['19:32:35.0', '20:41:46.0', '20:18:58.0', '19:42:24.0']
ids = [0, 17, 34, 44]

for event_time in event_times:
    print(event_time)
    t_start = datetime.strptime('2020-07-06 ' + event_time, '%Y-%m-%d %H:%M:%S.%f')
    t_end = t_start + timedelta(seconds=6)
    das_data, headers, axis = load_das_data(das_folder_path, t_start, t_end, raw=True)
    das_data = das_data.T

    plot_das_data(das_data)




'''
GENERATE SYNTHETIC DATA FROM SEISMOMETERS:
'''


# 1. einlesen der Daten:
folder_path = 'data/synthetic_DAS/raw_seismometer/'
data_paths = os.listdir(folder_path)
for data_path in data_paths:
    stream = read(folder_path + '/' + data_path)
    stats = stream[0].stats
    data = stream[0].data

    # downsample data:
    if not stats['sampling_rate'] == fs:
        print('DATA NEEDS TO BE DOWNSAMPLED')

    # filter data:
    data = butter_bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs, order=4)

    # normalize data:
    data = data / np.std(data)

    for SNR in SNR_values:

        SNR = 10 ** (0.5*SNR)
        amplitude = 2 * SNR / np.abs(data).max()
        data = data * amplitude

        synthetic_data = np.zeros((n_channel, data.shape[0]))
        noise = np.random.standard_normal(synthetic_data.shape)

        for i in range(n_channel):
            synthetic_data[i] = np.roll(data, i*shift)

        synthetic_data = synthetic_data + noise

        # normalize data:
        for i in range(n_channel):
            synthetic_data[i] = synthetic_data[i] / np.std(synthetic_data[i])

        #plt.plot(synthetic_data[6])
        #plt.show()

        # savedata
        file_name = data_path.split('/')[-1][0:5] + '_SNR:' + str(round(SNR, 1)) + '.npy'
        #np.save('/home/johanna/PycharmProjects/MAIN_DAS_denoising/data/synthetic_DAS/' + file_name, synthetic_data)












