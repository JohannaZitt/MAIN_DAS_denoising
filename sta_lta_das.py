import numpy as np
import matplotlib.pyplot as plt
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5
from scipy.signal import butter, lfilter
from datetime import datetime, timedelta
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import plot_trigger
from obspy import Trace


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
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
    if receiver == "AKU":
        channel = 3740
    elif receiver == "AJP":
        channel = 3460
    elif receiver == "ALH":
        channel = 3842
    elif receiver == "RA82":
        channel = 1300
    elif receiver == "RA85":
        channel = 1070
    elif receiver == "RA87":
        channel = 1230
    elif receiver == "RA88":
        channel = 1615 # 1600
    else:
        print("There is no start nor end channel for receiver " + receiver + '.')

    channel = int(channel/6)
    return channel

def load_das_data(folder_path, t_start, t_end, receiver, raw):

    # 1. load data
    data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=folder_path, convert=False)
    data = data.astype("f")

    # 2. downsample data in space:
    if raw:
        if data.shape[1] == 4864 or data.shape[1] == 4800 or data.shape[1] == 4928 :
            data = data[:,::6]
        else:
            data = data[:, ::3]
        headers["dx"] = 12


    # 3. cut to size
    ch_middel = get_middel_channel(receiver)  # get start and end channel:
    data = data[:, ch_middel-40:ch_middel+40]

    if raw:
        # 4. downsample in time
        data = resample(data, headers["fs"] / 400)
        headers["fs"] = 400

    # 5. bandpasfilter and normalize
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers["fs"], order=4)
        data[:,i] = data[:,i] / np.abs(data[:,i]).max()
        #data[:, i] = data[:,i] / np.std(data[:, i])

    return data, headers, axis

def csl_method_1(data, sta, lta, n_ch):

    csl_sum = np.zeros(data.shape[0])

    for i in range(data.shape[1]):
        csl = classic_sta_lta(data[:, i], sta, lta)
        csl_sum += csl
    csl_sum /= n_ch

    return csl_sum

def csl_method_2(data, sta, lta, n_ch):
    stacked_data = np.sum(data, axis=1)

    cft = classic_sta_lta(stacked_data, sta, lta)

    return cft


"""


There are different approaches to use sta/lta for das data

1. from jousset2022.pdf volcanic events
    STA/LTA is computed for every channel along the fibre and then averaged.
    For this characteristic function the median absolute deviation is calculated.
    An event is declared when a threshold defined as the median plus three times the MDA
    is exceeded.

2. from klaasen2021.pdf (Paitz and Walter) Volcaon-Glacial-environment
    STA/LTA is computed for the stacked DAS channels (corresponding to 30-40 meters)
    The STA window length was 0.3 s, the LTA window length 60 s, the trigger value 7.5,
    and the detrigger value 1.5.
    

There are a lot of parameters to vary:
1. sta window: STA duration must be longer than a few periods of a typical expected seismic signal
               0.2-0.5 seconds in cryoseismolgy
               Since we consider multiple channels at once, sta duration can be chosen longer than considering single trace
2. lta window: LTA duration must be longer than a few periods of a typically irregular seismic noise fluctuation
               5-30 seconds in cryoseismology
3. trigger threshold: 4 worked well
4. detrgger threshold: 1 worked well
5. amount of channels: depends strongly on the extend of the icequake. using 20 channel, corresponding to 240 m cable length 


"""

"""
1. Load Data
"""
event_times = ["2020-07-27 04:25:23.0", "2020-07-27 19:43:31.0", "2020-07-27 14:52:08.0", "2020-07-27 13:43:55.0"]
event_ids = [85, 5, 8, 15]
event_categories = [2, 1, 1, 1]


event = 1
t_start = datetime.strptime(event_times[event], "%Y-%m-%d %H:%M:%S.%f") - timedelta(seconds=10)
t_end = t_start + timedelta(seconds=20)


raw_data, raw_headers, raw_axis = load_das_data(folder_path="data/raw_DAS/",
                                                t_start=t_start, t_end=t_end, receiver="ALH", raw=True)
denoised_data, denoised_headers, denoised_axis = load_das_data(folder_path="experiments/03_accumulation_horizontal/denoisedDAS/",
                                                               t_start=t_start, t_end=t_end, receiver="ALH", raw=False)
raw_data = raw_data[:, 30:50]
denoised_data = denoised_data[:, 30:50]

"""
2. Parameters
"""
fs = 400
sta = 0.4*fs
lta = 10*fs
thr_on = 4
thr_off = 1
n_ch = 20


"""
Method 1: computing classical sta_lta for every channel and stacking characteristic function
"""

raw_csl_1 = csl_method_1(raw_data, sta, lta, n_ch)
denoised_csl_1 = csl_method_1(denoised_data, sta, lta, n_ch)


"""
Method 2: Stacking Data and than calculating classical sta_lta
"""
raw_csl_2 = csl_method_2(raw_data, sta, lta, n_ch)
denoised_csl_2 = csl_method_2(denoised_data, sta, lta, n_ch)



raw_trace = Trace(data=raw_data)
denoised_trace = Trace(data=denoised_data)

plot_trigger(raw_trace, raw_csl_1, thr_on, thr_off)
plot_trigger(denoised_trace, denoised_csl_1, thr_on, thr_off)



