import h5py

with h5py.File('cc_gain.h5', 'r') as hf:
    for experiment in hf.keys():
        print(print("Experiment: ", experiment))
        experiment_group = hf[experiment]

        for data_type in experiment_group.keys():
            print('\tData Type: ', data_type)
            data_type_group = experiment_group[data_type]

            for seis_event in data_type_group.keys():
                print('\t\tSeismometer Event: ', seis_event)
                seis_event_group = data_type_group[seis_event]
                #data_cc = seis_event_group['data_cc'][:]
                #print('\t\tData CC: ', data_cc)
                data_cc_seis = seis_event_group['data_cc_seis'][:]
                print('\t\tData CC Seis: ', data_cc_seis)