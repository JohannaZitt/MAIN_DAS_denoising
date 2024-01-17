import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import ndarray
from obspy import read
from scipy.signal import butter, lfilter


def resample(stream, ratio):
    """
    :param stream: stream object, which has to be resampled
    :param ratio: resample ratio = fs_old/fs_new
    :return: resampled data as np array
    """

    # convert stream to np array
    n_trc = len(stream)
    n_t = stream[0].stats.npts
    data: ndarray = np.zeros((n_trc, n_t))
    for i in range(n_trc):
        data[i] = stream[i].data[0:n_t]

    # resample
    res = np.zeros((data.shape[0], int(data.shape[1]/ ratio) + 1))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = 'band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order = 4):
    b, a = butter_bandpass(lowcut, highcut, fs, order = order)
    y = lfilter(b, a, data)
    return y


"""

Generates preprocessed training data for the model training from manually selected seismometer waveforms. 
Proprocessing:
 1. Downsampled to 400 Hz
 2. Filtered between 1-120 Hz
 3. Compute strain rate
 4. Normalize with absolute maximal value

"""

folders = ['01_ablation_horizontal', '02_ablation_vertical', '03_accumulation_vertical', '04_accumulation_horizontal', '05_stick-slip', '06_surface', '07_combined120', '09_random480']

for folder in folders:

    # Reading Data
    stream = read('data/training_data/raw_seismometer/' + folder + '/ID*.mseed')
    n_trc = len(stream)
    n_t = stream[0].stats.npts

    # downsample data to 400 Hz and convert to np array:
    fs_old = stream[0].stats.sampling_rate
    fs = 400
    if fs_old != fs:
        data = resample(stream, fs_old / fs)
    else:
        data: ndarray = np.zeros((n_trc, n_t))
        for i in range(n_trc):
            data[i] = stream[i].data[0:n_t]


    # parameters for calculating strain rate:
    gauge_length: int = 10 # in m
    swave_velocity = 1800 # in m/s
    t = gauge_length / swave_velocity
    rollouttotal: int = int(t * fs) + 1
    rollout = int(rollouttotal / 2)

    n_trc, n_t = data.shape
    for i in range(n_trc):

        # filter data
        data[i] = butter_bandpass_filter(data[i], lowcut=1, highcut=120, fs=fs, order=4)

        # compute  ground velocity into strain rate
        data[i] = np.roll(data[i], rollout) - np.roll(data[i], -rollout)
        data[i] /= gauge_length

        # scale by max amplitude
        max_amplitude = np.abs(data[i]).max()
        data[i] /= max_amplitude


    savedir = 'data/training_data/preprocessed_seismometer/'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    #np.save(savedir + folder, data)

# Compute combined800 training data set:
loaded_arrays = []
for i in range(4):
    loaded_array = np.load('data/training_data/preprocessed_seismometer/' + folders[i] + '.npy')
    loaded_arrays.append(loaded_array)
combined_array = np.vstack(loaded_arrays)
savedir = 'data/training_data/preprocessed_seismometer/'
#np.save(savedir + '08_combined480', combined_array)

'''
# Plotting the training_data waveforms
for i in range(10, 20):
    plt.plot(data[i])
    plt.savefig('test/strain_rate/pic_' + str(i))
    plt.show()
'''

