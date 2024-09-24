import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = "band")
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order = 4):
    b, a = butter_bandpass(lowcut, highcut, fs, order = order)
    y = lfilter(b, a, data)
    return y

def resample_seis(data, ratio) -> NDArray:

    """
    :param data: np array
    :param ratio: resample ratio = fs_old/fs_new
    :return: resampled data as np array
    """

    # resample
    res = np.zeros((data.shape[0], int(data.shape[1]/ ratio) + 1))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res

def resample_DAS(data, ratio) -> NDArray:
    """
    :param data: np array
    :param ratio: resample ratio = fs_old/fs_new
    :return: resampled data as np array
    """

    data = data.T
    # resample
    res = np.zeros((data.shape[0], int(data.shape[1]/ ratio)+1))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res.T