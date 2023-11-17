from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy import read
from obspy.signal.cross_correlation import correlate
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5


'''

This Script should:
Cross - Correlation Values overview max:
meaning computed the maximum correlation over 40 traces and for each trace the signal was shifted 41 times.
The Cross-correlation was computed over a time window of 1.25 sec.

'''


def resample(data, ratio):
    res = np.zeros((int(data.shape[0]/ratio)+1, data.shape[1]))
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




''' PARAMETERS FOR DAS LOADING AND PLOTTING 2020-07-10T03:54:35.0,ALH,P2,3822 '''
# time of hole data
t_start = datetime.strptime('2020/07/10 03:54:32.0', '%Y/%m/%d %H:%M:%S.%f')
t_end = datetime.strptime('2020/07/10 03:54:38.0', '%Y/%m/%d %H:%M:%S.%f')
# DAS channels to plot and calculate cc from:
ch_start = 3812
ch_end = 3842
# data points to end and start plotting and calculating cross-correlation:
n_start = 1000
n_end = 1500
# normalizing method: 'trace' or 'stream'
norm_method = 'trace'

''' PARAMETERS FOR SEISMOMETER LOADING '''
receiver = 'ALH'
event_date = '2020-07-10'
event_time = '03:54:35'
component = 'p2'

# load csv file
csv_file = np.genfromtxt('stick_slip_events_to_evaluate.csv', dtype=str, delimiter=',')

# load raw DAS data:
raw_folder_path = 'data/raw_DAS/'
raw_data, raw_headers, raw_axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=raw_folder_path,
                                                                  convert=False)
# preprocess raw DAS data and update header:
raw_data = resample(raw_data, 2.5)
raw_headers['fs'] = 400

# load denoised DAS data to plot:
denoised_folder_path = 'experiments/2023-08-24_surface_stick_slip_accumulation_13_1024/denoisedDAS/'
denoised_data, denoised_headers, denoised_axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=denoised_folder_path,
                                                                  convert=False)
# update header:
denoised_headers['fs'] = 400

# plot DAS data for controll:
raw_trc = []
trc = []
for i in range(ch_start, ch_end):
    raw_trc.append(Trace(data = raw_data[:,i],
                    header = {'location': 'rhonegletscher', 'channel': str(i), 'distance': raw_axis['dd'][i],
                              'sampling_rate': raw_headers['fs'], 'starttime': raw_headers['t0'],
                              'datatype': 'das'}))
    trc.append(Trace(data=denoised_data[:, i],
                     header={'location': 'rhonegletscher', 'channel': str(i), 'distance': raw_axis['dd'][i],
                             'sampling_rate': raw_headers['fs'], 'starttime': raw_headers['t0'],
                             'datatype': 'das'}))
raw_stream = Stream(traces = raw_trc)
stream = Stream(traces = trc)
#raw_secplt = raw_stream.plot(type = 'section', orientation = 'horizontal', size = (800, 1200), norm_method = norm_method)
#secplt = stream.plot(type = 'section', orientation = 'horizontal', size = (800, 1200), norm_method = norm_method)


# load seismometer data:
seis_folder_path = os.path.join('/home/johanna/PycharmProjects/IcequakePicking/saved_data/c0' + receiver,
                                str(event_date), component.lower())
for filename in os.listdir(seis_folder_path):
    if str(event_time) in filename:
        seis_file = filename
        break
seis_stream = read(seis_folder_path + '/' + seis_file)
seis_data = seis_stream[0].data
seis_stats = seis_stream[0].stats

plt.plot(seis_data[n_start:n_end])
#plt.show()



# load denoised DAS Data:
list_experiments = os.listdir('experiments/')
for file in list_experiments:
    print('#################################################################')
    print(file)
    denoised_folder_path = os.path.join('experiments', file, 'denoisedDAS/')
    denoised_data, denoised_headers, denoised_axis = load_das_h5.load_das_custom(t_start, t_end, input_dir=denoised_folder_path,
                                                                      convert=False)
    denoised_headers['fs'] = 400

    # compute crosscorrelation:
    max_cc = 0
    max_channel = 0
    for i in range(ch_start, ch_end):
        cc = correlate(seis_data[n_start:n_end], raw_data[n_start-20:n_end+20,i], 20, normalize='naive', method='direct')
        if max_cc < np.max(cc):
            max_cc = np.max(cc)
            max_channel = i
    #print(max_cc)
    #print(max_channel)

    # extract bin size:
    print(file[-6:-5])
    if file[-6:-5] == '9':
        bin_size = 9
    if file[-6:-5] == '3':
        bin_size = 13
    if file[-6:-5] == '1':
        bin_size = 11

    # compute cc gain
    raw_cc = compute_moving_coherence(raw_data.T[ch_start:ch_end, n_start:n_end], bin_size)
    denoised_cc = compute_moving_coherence(denoised_data.T[ch_start:ch_end, n_start:n_end], bin_size)

    cc_gain = denoised_cc/raw_cc
    cc_max = np.max(cc_gain)
    position = np.argmax(cc_gain)

    # print mac cc gain and position
    print('cc_gain: ', cc_max)
    print('cc_gain position: ', position)




