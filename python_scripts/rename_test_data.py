import os


folder_path = '../data/raw_DAS/surface_ablation/'
files = os.listdir(folder_path)
#files = ['urface_ablationrhone1khz_UTC_20200723_113908.575.h5']

for file in files:
    new_filename = file[15:]
    #os.rename(folder_path + file, folder_path + new_filename)


'''
folder_path = '../data/test_data/stick-slip_accumulation/'
files = os.listdir(folder_path)
#files = ['ID:1_2020-07-06_10:27:21_c0ALH_p2.mseed']



for i, file in enumerate(files):
    component = file[-8:-6]
    receiver = file[-12:-9]
    time = file[-34:-15]
    new_filename = 'stick-slip_' + receiver + '_' + time + '_' + component + '.mseed'
    #print(new_filename)
    os.rename(folder_path + file, folder_path + new_filename)
'''
