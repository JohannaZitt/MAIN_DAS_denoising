import os
import numpy as np
from obspy import read
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from datetime import datetime, timedelta
from obspy import UTCDateTime


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
GENERATE SYNTHETIC DATA FROM SEISMOMETERS:
'''


# 1. einlesen der Daten:
folder_path = '/data/synthetic_DAS/raw_seismometer/'
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


'''
GENERATE SYNTHETIC DATA FROM DAS DATA:
'''

das_folder_path = '/data/raw_DAS/0706'


# DAS data Zeitpunkte:
# 2020-07-06T19:32:41.0 ID 0
# 2020-07-06T20:41:46.0 ID 17
# 2020-07-06T20:18:58.0 ID 34
# 2020-07-06T10:42:24.0 ID 44






