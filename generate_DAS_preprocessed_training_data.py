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
    """
    :param data: data to resample
    :param ratio: resample ratio = fs_old/fs_new
    :return: resampled data as np array
    """

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
event01 = ["2020/07/07", "13:24:41.5", 1320, 1450] #Ablation Zone
event02 = ["2020/07/11", "12:34:15.0", 1260, 1310] #Ablation Zone
event03 = ["2020/07/12", "06:00:37.5", 1255, 1410] #Ablation Zone
event04 = ["2020/07/12", "06:53:13.5", 1160, 1215] #Ablation Zone
event05 = ["2020/07/12", "06:53:13.5", 1215, 1350] #Ablation Zone
event06 = ["2020/07/12", "06:53:28.0", 1200, 1350] #Ablation Zone
event07 = ["2020/07/13", "00:48:36.7", 395, 440] #Ablation Zone
event08 = ["2020/07/06", "19:32:37.5", 3425, 3550] #Accumulation Zone
event09 = ["2020/07/06", "20:42:37.5", 3410, 3510] #Accumulation Zone
event10 = ["2020/07/06", "20:41:49.0", 3400, 3520] #Accumulation Zone
event11 = ["2020/07/06", "19:27:10.5", 3410, 3510] #Accumulation Zone
event12 = ["2020/07/06", "20:19:00.5", 3420, 3550] #Accumulation Zone
event13 = ["2020/07/06", "19:42:27.0", 3400, 3510] #Accumulation Zone
event14 = ["2020/07/06", "19:51:23.0", 3400, 3510] #Accumulation Zone
events = [event01, event02, event03, event04, event05, event06, event07, event08, event09, event10, event11, event12, event13, event14]
n_sub = 15
n_t = 2400
fs = 400
training_data = np.zeros((60, n_t, n_sub))
n = 0

for event in events: # for evey event

    print(event)

    for m in range(3): # m is the start channel of every

        # 1. load DAS data
        file_dir = "data/training_data/raw_DAS/"
        event_time = datetime.strptime(event[0] + " " + event[1], "%Y/%m/%d %H:%M:%S.%f")
        t_start = event_time - timedelta(seconds=3)
        t_end   = t_start + timedelta(seconds=6)
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


        # 3. downsample data in time: we need sample frequency of 400 Hz
        data = resample(data, headers["fs"] / fs)
        data = data[:n_t, :]

        # 4. filtering data
        lowcut_sec = 1
        highcut_sec = 120
        for i in range(data.shape[0]):
            data[i] = butter_bandpass_filter(data[i], lowcut_sec, highcut_sec, 400, order=4)

        # 5. set mean value to 0
        for i in range(data.shape[1]):
            mean = np.mean(data[:, i])
            data[:, i] = data[:, i] - mean

        # 6. normalize data
        for i in range(data.shape[1]):
            data[:,i] /= data[:,i].std()

        # 7. concatenate all data in one dataset training_data
        n_samples = int(data.shape[1] / n_sub)
        for i in range(n_samples):
            training_data[n] = data[:, i*n_sub:(i+1)*n_sub]
            n += 1


# 8. Save data:
training_data = np.transpose(training_data, (0, 2, 1))
print(training_data.shape)
#np.save("data/training_data/preprocessed_DAS/retraining_data.npy", training_data)



"""
PLOT SAVED DATA

for j in range(training_data.shape[0]):
    plot_data = training_data[j]
    i = 0
    for ch in range(15):
        plt.plot(plot_data[ch][:] + 10 * i, "-k", alpha=0.8)
        i += 1
    plt.show()

"""


