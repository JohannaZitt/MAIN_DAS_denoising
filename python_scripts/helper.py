import os
import re
import os

from obspy import UTCDateTime
from webdav3.client import Client
import numpy as np

'''
eight_path = os.path.join("../experiments", '08_combined480', "plots", "ablation", "0706_RA88")
nine_path = os.path.join("../experiments", '09_random480', "plots", "accumulation", "0706_AJP")

categories = ["1_raw:visible_denoised:better_visible",
              "2_raw:not_visible_denoised:visible",
              "3_raw:not_visible:denoised:not_visible",
              "5_miscellaneous"]

eight_ids = []
for category in categories:

    events = os.listdir(eight_path + "/" + category)
    for event in events:
        id = int(re.search(r'ID:(\d+)', event).group(1))
        eight_ids.append(id)

nine_ids = []
for category in categories:

    events_nine = os.listdir(nine_path + "/" + category)
    for event in events_nine:
        id = int(re.search(r'ID:(\d+)', event).group(1))
        nine_ids.append(id)


eight_ids.sort()
nine_ids.sort()
print(eight_ids)
print(nine_ids)
print(len(eight_ids))
print(len(nine_ids))
for i in eight_ids:
    if not i in nine_ids:
        print('Hier: ', i)
        


path = "../old/raw_seismometer/"
folders = ["ablation_horizontal_surface", "ablation_vertical_surface", "accumulation_horizontal_surface", "accumulation_vertical_surface"]


with open("../surface_event_times.txt", "w") as file:

    for folder in folders:
        event_times = os.listdir(path + folder)
        for event_time in event_times:
            file.write(str(event_time) + "\n")
            
'''

""" Download DAS test

# Access to Nextcloud Server via generated App Password
options = {
    'webdav_hostname': 'https://cloud.scadsai.uni-leipzig.de/remote.php/dav/files/jz76tevi/',
    'webdav_login': 'jz76tevi',
    'webdav_password': 'AyJ5r-qpbxe-doxPc-sMKss-9KJ48'
}
client = Client(options)

file = "rhone1khz_UTC_20200804_001220.542.h5"

remote_path = 'environment-earth/Projects/Rhonegletscher/Data/no_backup/DAS_2020/20200804/' + file
local_path = "helper_folder/" + file

client.download(remote_path=remote_path , local_path=local_path)
"""

