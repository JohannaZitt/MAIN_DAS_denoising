from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from datetime import datetime, timedelta

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
    '''

    :param data: data, which has to be resampled
    :param ratio: resample ratio = fs_old/fs_new
    :return: resampled data as np array
    '''

    data = data.T
    # resample
    res = np.zeros((data.shape[0], int(data.shape[1]/ ratio)+1))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res.T

def plot_data(data, moveout=8):
    i = 0
    alpha = 0.8
    for ch in range(data.shape[0]):
        plt.plot(data[ch][:] + moveout * i, '-k', alpha=alpha)
        i += 1

    plt.show()


# event_data, event_time, start_channel, end_channel
event01 = ['2020/07/07', '13:24:40.5', 1320, 1450]
event02 = ['2020/07/11', '12:34:14.0', 1260, 1310]
event03 = ['2020/07/12', '06:00:36.5', 1255, 1410]
event04 = ['2020/07/12', '06:53:12.5', 1160, 1215]
event05 = ['2020/07/12', '06:53:12.5', 1215, 1350]
event06 = ['2020/07/12', '06:53:26.5', 1200, 1350]
event07 = ['2020/07/13', '00:48:35.7', 395, 440]

events = [event01, event02, event03, event04, event05, event06, event07]
n_sub = 15 # wird in DataGenerator dann auf n_sub = 11 mit zufälligem Start channel gecuttet
n_t = 1024
training_data = np.zeros((39, n_t, n_sub))
n = 0

for event in events: # for evey event

    for m in range(3): # m is the start channel of every

        # 1. load DAS data
        file_dir = 'old_old/data/training_data/raw_DAS/'
        t_start = datetime.strptime(event[0] + ' ' + event[1], '%Y/%m/%d %H:%M:%S.%f')
        t_end   = t_start + timedelta(seconds=3.5)
        data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=file_dir, convert = False)

        # 2. downsample to 12 m channel spacing
        if data.shape[1] == 4864 or data.shape[1] == 4800:
            data = data[:, m::6]
            startchannel = event[2] // 6
            endchannel = event[3] // 6
        else:
            data = data[:, m::3]
            startchannel = event[2] // 3
            endchannel = event[3] // 3
        data = data[:, startchannel:endchannel]


        # 3. set mean value to 0
        for i in range(data.shape[1]):
            mean = np.mean(data[:,i])
            data[:,i] = data[:, i] - mean

        # 4. filtering data
        lowcut_sec = 1
        highcut_sec = 120
        for i in range(data.shape[0]):
            data[i] = butter_bandpass_filter(data[i], lowcut_sec, highcut_sec, 1000, order=4)

        # 4. downsample data in time: we need sample frequency of 400 Hz
        data = resample(data, headers['fs']/400)
        data = data[:n_t, :]

        # 5. normalize data
        for i in range(data.shape[1]):
            data[:,i] /= data[:,i].std()

        # 6. concatenate all data in one dataset training_data
        n_samples = int(data.shape[1] / n_sub)
        for i in range(n_samples):
            training_data[n] = data[:, i*n_sub:(i+1)*n_sub]
            n += 1


# 7. Save data:
training_data = np.transpose(training_data, (0, 2, 1))
np.save('old_old/data/training_data/preprocessed_DAS/retraining_data.npy', training_data)



'''
IN CASE YOU WANNA PLOT DATA to double check

for j in range(training_data.shape[0]):
    plot_data = training_data[j]
    i = 0
    for ch in range(15):
        plt.plot(plot_data[ch][:] + 8 * i, '-k', alpha=0.8)
        i += 1
    plt.xlabel('Time[s]')
    plt.ylabel('Offset [m]')

    plt.show()

'''


