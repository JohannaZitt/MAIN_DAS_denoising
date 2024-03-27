"""


"""
import os

import numpy as np
from obspy import read
from obspy import UTCDateTime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import keras

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

    res = np.zeros((data.shape[0], int(data.shape[1] / ratio) + 1))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res

# Loading event time
event_path = "data/test_data/seismometer_denoising/borehole/"
seismometer_path = "data/test_data/seismometer_denoising/seismometer/"
events = os.listdir(event_path)
#events = events[:10]

for event in events:
    event_time = event[-23:-15]
    print(event_time)

    starttime = UTCDateTime("2020-07-27T" + event_time + ".0")
    starttime = starttime - timedelta(seconds=3)
    endtime = starttime + timedelta(seconds=8)

    # Loading corresponding data to denoise:
    stream = read(seismometer_path + "4D.RA81..EHZ.D.2020.209", starttime=starttime, endtime=endtime)
    stream += read(seismometer_path + "4D.RA82..EHZ.D.2020.209", starttime=starttime, endtime=endtime)
    stream += read(seismometer_path + "4D.RA83..EHZ.D.2020.209", starttime=starttime, endtime=endtime)
    stream += read(seismometer_path + "4D.RA84..EHZ.D.2020.209", starttime=starttime, endtime=endtime)
    stream += read(seismometer_path + "4D.RA85..EHZ.D.2020.209", starttime=starttime, endtime=endtime)
    stream += read(seismometer_path + "4D.RA86..EHZ.D.2020.209", starttime=starttime, endtime=endtime)
    stream += read(seismometer_path + "4D.RA87..EHZ.D.2020.209", starttime=starttime, endtime=endtime)
    stream += read(seismometer_path + "4D.RA88..EHZ.D.2020.209", starttime=starttime, endtime=endtime)

    stream_borehole = read(event_path + event)

    # convert stream to np.array:
    trace_data = [trace.data for trace in stream]
    data = np.vstack(trace_data)
    data_borehole = stream_borehole[0].data

    # preprocess data
    # 1. resample in time
    data = resample(data, 500/400)
    data_borehole = np.interp(np.arange(0, len(data_borehole), 2000/400), np.arange(0, len(data_borehole)), data_borehole)
    # 2. filter data
    for i in range(data.shape[0]):
        data[i]=butter_bandpass_filter(data[i], lowcut=1, highcut=120, fs=400, order=4)
        data[i] /= data[i].std()
    data_borehole = butter_bandpass_filter(data_borehole, lowcut=1, highcut=120, fs=400, order=4)
    data_borehole /= data_borehole.std()


    # Load Model:
    model_name = "11_borehole_seismometer_520"
    model_file = os.path.join("experiments", model_name, model_name + ""'.h5')
    model = keras.models.load_model(model_file)



    # Generate masks:
    N_ch = 8
    N_sub = 8
    N_t = 1024
    masks = np.ones((N_ch, N_sub, N_t, 1))
    for i in range(N_ch):
        masks[i, i] = 0


    res = np.zeros((N_ch, 3*1024))
    for n in range(3):
        # Generate eval_samples:
        sub_section = data[:, n*1024:(1+n)*1024]
        sub_section = np.expand_dims(sub_section, -1)
        eval_samples = np.zeros_like(masks)
        for i in range(N_ch):
            eval_samples[i] = sub_section

        # Denoise data
        model_output=model.predict((eval_samples, masks)) # denoised_data ist auch gleicher shape wie oben
        model_output = np.squeeze(model_output)
        seismometer_reconstructions = np.zeros((N_ch, N_t))
        for i in range(N_ch):
            seismometer_reconstructions[i] = np.sum(model_output[i], axis=0)

        res[:, n*1024:(1+n)*1024] = seismometer_reconstructions


    for i in range(res.shape[0]):
        print(i)
        res[i] = butter_bandpass_filter(res[i], 1, 120, fs=400, order=4)
        res[i] /= np.std(res[i])

    # plot raw data

    fontsize = 16
    id = event[3:5]
    data = data[:, 900:1900]
    data_borehole = data_borehole[900:1900]
    plt.figure(figsize=(15, 10))
    for i in range(data.shape[0]):
        plt.plot(data[i] + 12 * i, color="black", alpha=0.6, )
    plt.plot(data_borehole + 12 * (data.shape[0] + 1), color="red")
    plt.yticks([])
    plt.ylabel("Ground velocity [norm]", fontsize=fontsize)
    plt.xticks([0, 200, 400, 600, 800, 1000], [0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=fontsize-2)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.title("Raw: " + event, fontsize=fontsize+2)
    plt.text(1, 112, "Borehole seismometer data: ", fontsize=fontsize)
    plt.text(1, 92, "Surface seismometer data: ", fontsize=fontsize)
    plt.savefig("experiments/11_borehole_seismometer_520/pics/" + id + "_raw.png")
    plt.show()

    # plot denoised data:
    res = res[:, 900:1900]
    plt.figure(figsize=(15, 10))
    for i in range(res.shape[0]):
        plt.plot(res[i] + 12 * i, color="black", alpha=0.6)
    plt.plot(data_borehole + 12 * (res.shape[0] + 1), color="red")
    plt.yticks([])
    plt.ylabel("Ground velocity [norm]", fontsize=fontsize)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.xticks([0, 200, 400, 600, 800, 1000], [0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=fontsize-2)
    plt.title("Denoised: " + event, fontsize=fontsize+2)
    plt.text(1, 112, "Borehole seismometer data: ", fontsize=fontsize)
    plt.text(1, 92, "Surface seismometer data: ", fontsize=fontsize)
    plt.savefig("experiments/11_borehole_seismometer_520/pics/" + id + "_denoised.png")
    plt.show()
    # Save denoised data

