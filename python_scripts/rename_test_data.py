import os
'''
folder = '../data/test_data/surface_accumulation'
files = os.listdir(folder)
#files = ['stick-slip_AJP_2020-07-07_04:33:19_p2.mseed']

for file in files:
    new_file_name = file[:32] + '_' + file[-8:]
    #print(new_file_name)
    #os.rename(folder + '/' + file, folder + '/' + new_file_name)


models_path = '../experiments'
models = os.listdir(models_path)
print(models)

raw_DAS_path = '../data/raw_DAS'
data_types = os.listdir(raw_DAS_path)
print(data_types)
#print(data_types)
#data_types = ['stick-slip_ablation', 'stick-slip_accumulation', 'surface_accumulation']

for model in models:

    # for every raw data folder
    for data_type in data_types:

        raw_das_folder_path = os.path.join(raw_DAS_path, data_type)
        saving_path = os.path.join('../experiments', model, 'denoisedDAS', data_type) # Hier fliege ich immer raus, sobald
        model_file = os.path.join('../experiments', model, model + '.h5')
        print('hi')


'''
'''
import os


folder_path = '../data/raw_DAS/surface_ablation/'
files = os.listdir(folder_path)
#files = ['urface_ablationrhone1khz_UTC_20200723_113908.575.h5']

for file in files:
    new_filename = file[15:]
    #os.rename(folder_path + file, folder_path + new_filename)

'''
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
