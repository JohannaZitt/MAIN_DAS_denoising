
import csv
import os
import re
from datetime import datetime, timedelta

import numpy as np
from obspy import read
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5

from helper_functions import butter_bandpass_filter, compute_moving_coherence, xcorr, get_middel_channel




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
    data = data[:, ch_middel-20:ch_middel+20]

    if raw:
        # 4. downsample in time
        data = resample(data, headers["fs"] / 400)
        headers["fs"] = 400

    # 5. bandpasfilter and normalize
    for i in range(data.shape[1]):
        data[:, i] = butter_bandpass_filter(data[:,i], 1, 120, fs=headers["fs"], order=4)
        #data[:,i] = data[:,i] / np.abs(data[:,i]).max()
        data[:, i] = data[:,i] / np.std(data[:, i])

    return data, headers, axis


"""

Here we calculate the Local Waveform Coherence as well as the Cross Correlation between DAS data and Co-located Seismometer
as Described in Section 4.1.1 and Section 4.3.

"""

# TODO Comment out to calculate values for all experiments
#experiments = ["01_ablation_horizontal", "02_ablation_vertical", "03_accumulation_horizontal", "04_accumulation_vertical"
#               "05_combined200", "06_combined800", "07_retrained_combined200", "08_retrained_combined800", "09_borehole_seismometer"]
experiments = ["01_ablation_horizontal"]

data_types = ["ablation", "accumulation"]

for experiment in experiments: # for every experiment

    print("#################################################################################")
    print("#################################################################################")
    print("#################################################################################")
    print("#####################" + experiment + "##################################")
    print("#################################################################################")
    print("#################################################################################")
    print("#################################################################################")

    """ Open File in which values are saved """
    with open("experiments/" + experiment + "/cc_evaluation_" + experiment[:2] + ".csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["id", "mean_cc_gain", "mean_cross_gain", "zone", "model"])

        for data_type in data_types:  # for every data type

            seis_data_path = "data/test_data/" + data_type
            seismometer_events = os.listdir(seis_data_path)

            """
            
            mean_cc_gain:       local waveform coherence:
            mean_cross_gain:    cross-correlation gain between DAS data and co-located seismometer
            
            """

            for seismometer_event in seismometer_events:
                print("SEISMOMETER EVENT: ", seismometer_event)

                """ Search for correct event """
                event_time = seismometer_event[-23:-15]
                event_date = seismometer_event[-34:-24]
                id = re.search(r"ID:(\d+)", seismometer_event).group(1)
                receiver = ""
                zone = ""

                if data_type[:2] == "ab":
                    receiver = seismometer_event[-14:-10]
                    zone = "ablation"
                elif data_type[:2] == "ac":
                    receiver = seismometer_event[-12:-9]
                    zone = "accumulation"
                else:
                    print("ERROR: No matching data type")


                """ Pick Time Window """
                t_start = datetime.strptime(event_date + " " + event_time + ".0", "%Y-%m-%d %H:%M:%S.%f")
                t_start = t_start - timedelta(seconds=3)
                t_end = t_start + timedelta(seconds=6)

                """ Load Seismometer DAta """
                seis_stream = read(seis_data_path + "/" + seismometer_event)
                seis_data = seis_stream[0].data
                seis_stats = seis_stream[0].stats

                """ Resampling """
                if seis_stats.sampling_rate == 500:
                    seis_data = np.interp(np.arange(0, len(seis_data), 500/400), np.arange(0, len(seis_data)), seis_data)
                    seis_stats.sampling_rate = 400.0
                    seis_stats.npts = seis_data.shape[0]
                """ Filter and Normalize DAta """
                seis_data = butter_bandpass_filter(seis_data, 1, 120, fs=seis_stats.sampling_rate, order=4)
                seis_data = seis_data / np.std(seis_data)

                """ Load raw DAS data """
                raw_folder_path = "data/raw_DAS/"
                raw_data, raw_headers, raw_axis = load_das_data(folder_path =raw_folder_path, t_start = t_start, t_end = t_end, receiver = receiver, raw = True)

                """ Load Denoised DAS data """
                denoised_folder_path = "experiments/" + experiment + "/denoisedDAS/"
                denoised_data, denoised_headers, denoised_axis = load_das_data(folder_path =denoised_folder_path, t_start = t_start, t_end = t_end, receiver = receiver, raw = False)


                """
                Calculate CC Gain
                """
                t_window_start = 688
                t_window_end = t_window_start + 1024
                bin_size = 11

                raw_cc = compute_moving_coherence(raw_data[t_window_start:t_window_end].T, bin_size)
                denoised_cc = compute_moving_coherence(denoised_data[t_window_start:t_window_end].T, bin_size)
                cc_gain = denoised_cc / raw_cc


                """
                Calculate CC between DAS and Seismometer: 
                """
                raw_cc_seis_total = []
                denoised_cc_seis_total = []
                raw_data = raw_data[:, 17:23]
                denoised_data = denoised_data[:, 17:23]
                print("Shape:", raw_data.shape)
                for i in range(raw_data.shape[1]):
                    raw_cc_seis = xcorr(raw_data.T[i], seis_data)
                    denoised_cc_seis = xcorr(denoised_data.T[i], seis_data)
                    raw_cc_seis_total.append(raw_cc_seis)
                    denoised_cc_seis_total.append(denoised_cc_seis)
                cc_gain_seis = np.array(denoised_cc_seis_total) / np.array(raw_cc_seis_total)


                """ Save Values: """
                writer.writerow(
                    [id, cc_gain.mean(), cc_gain_seis.max(), zone, experiment])



