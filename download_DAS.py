import os

from obspy import UTCDateTime
from webdav3.client import Client
import numpy as np


'''

This script downloads DAS 2020 files from the Next Cloud Server according to time, you specify in a csv file. 
The time needs to be in the format of YYYY-MM-DDThh:mm:ss.m (e.g. 2020-07-12T03:29:50.2) and needs be the first column
of the csv file.
The time you note should be the starting time of the section you wanna look at.
The files downloaded from Nextcloud are 30 seconds long. 
In case that you specify a time at the edge of a file (e.g. second 29 and the file ends at second 30), two files 
are downloaded and stored. 
The variable "range" defines the maximum range you want to look at in seconds. If you set range = 6 it is ensured that 
from the time you specify in the csv file, you always have a 6 second window you can look at. 


In order to not use your own password from the NextCloud Server in this Skript, generate a app password in Nextcloud 
and use it instead. Here is, how you do it: 

1. Log in to your Nextcloud account
2. Go to your profile settings and find the "Security" section.
3. Scroll down to "Devices & Sessions"
4. Choose your App name (e.g. "Access WebDAV Client") and click on "Create new app password"
5. Use this password in this script


'''

def get_event_times (event_folder_path):

    file_names = os.listdir(event_folder_path)

    event_times = []
    for file_name in file_names:
        date = file_name[-29:-19]
        time = file_name[-18:-10]
        event_time = date + 'T' + time + '.0'
        event_times.append(event_time)

    return event_times



def set_file_name (event_time, client, remote_path, range):

    remote_files = client.list(remote_path)

    seconds = int(event_time[-2:])
    minutes = int(event_time[-4:-2])

    # files with corresponding hh:mm
    possible_files = [filename for filename in remote_files if event_time[-6:-2] == filename[23:27]]

    # check whether files exist. When not return empty list
    return_files = []
    if not len(possible_files):
        return return_files

    file_sec1_int = int(possible_files[0][27:29])
    file_sec2_int = int(possible_files[1][27:29])
    if file_sec1_int > file_sec2_int:
        print('ATTENTION!! file_sec1 > file_sec2')

    case = 0
    # Find folder for the event
    if file_sec1_int <= seconds < file_sec2_int:
        print('case1')
        return_files.append(possible_files[0])
        case = 1
    if file_sec2_int <= seconds:
        print('case2')
        return_files.append(possible_files[1])
        case = 2
    if seconds < file_sec1_int:
        print('case3')
        minutes -= 1
        if 0 <= minutes <= 9:
            return_file = possible_files[1][:25] + '0' + str(minutes) + possible_files[1][27:]
            return_files.append(return_file)
        else:
            return_file = possible_files[1][:25] + str(minutes) + possible_files[1][27:]
            return_files.append(return_file)
        case = 3


    # Find folder for event + range
    if len(return_files):
        if case == 1 and seconds + range > file_sec2_int:
            print('1')
            return_files.append(possible_files[1])
        if case == 2 and (seconds + range) % 60 < range:
            print('2')
            minutes += 1
            if 0 <= minutes <= 9:
                return_file = possible_files[0][:25] + '0' + str(minutes) + possible_files[0][27:]
                return_files.append(return_file)
            else:
                return_file = possible_files[0][:25] + str(minutes) + possible_files[0][27:]
                return_files.append(return_file)
        if case == 3 and not (seconds + range) % 60 < file_sec1_int:
            print('3')
            return_files.append(possible_files[0])

    return return_files


# Get Event times from seismometer data:
event_folder_path = 'data/test_data/stick-slip_ablation'

events = get_event_times(event_folder_path)
print(events)

# Access to Nextcloud Server via generated App Password
options = {
    'webdav_hostname': 'https://cloud.scadsai.uni-leipzig.de/remote.php/dav/files/jz76tevi/',
    'webdav_login': 'jz76tevi',
    'webdav_password': 'AyJ5r-qpbxe-doxPc-sMKss-9KJ48'
}
client = Client(options)

# Set range in seconds
range = 6

# Set local path, where the data should be stored
local_path = 'data/raw_DAS/'

for event in events:

    print('___________________Event: ', event, ' _____________________________')

    # Set folder and file according to nextcloud structure
    event_time = UTCDateTime(event)
    folder = event_time.strftime("%Y%m%d")
    folder2 = event_time.strftime("%Y%m%d") + '_2'

    # Set remote and local path:
    remote_path = 'environment-earth/Projects/Rhonegletscher/Data/DAS_2020/' + folder
    remote_path2 = 'environment-earth/Projects/Rhonegletscher/Data/DAS_2020/' + folder2


    # Set Files we want to download
    files = set_file_name(event_time=event_time.strftime("%H%M%S"), client=client, remote_path=remote_path, range=range)

    # In case we didn't find the file in the first folder (YYYYMMDD), search for it in the second folder (YYYYMMDD_2)
    files2 = []
    if not len(files) and client.check(remote_path2):
        files2 = set_file_name(event_time=event_time.strftime("%H%M%S"), client=client, remote_path=remote_path2, range=range)

    # Download Data
    if len(files):
        print('download file ', files)
        for file in files:
            client.download(remote_path=remote_path + '/' + file, local_path=local_path + file)
    if len(files2):
        print('download file2 ', files2)
        for file in files2:
            client.download(remote_path=remote_path2 + '/' + file, local_path=local_path + file)
    if not len(files) and not len(files2):
        print("No data for that event time available :'(")

    
