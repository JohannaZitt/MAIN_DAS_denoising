import matplotlib.pyplot as plt
from obspy import read
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta



# DAS data Zeitpunkte:
# 2020-07-06T19:32:41.0 ID 0
# 2020-07-06T20:41:46.0 ID 17
# 2020-07-06T20:18:58.0 ID 34
# 2020-07-06T10:42:24.0 ID 44


# 1. einlesen der Daten:
data_path = '/home/johanna/PycharmProjects/MAIN_DAS_denoising/data/synthetic_DAS/raw_seismometer/'
for waveform in os.listdir(data_path):
    icequake = read(data_path + '/' + waveform)
    plt.plot(icequake[0].data)
    plt.show()
#print(icequake[0].data)
