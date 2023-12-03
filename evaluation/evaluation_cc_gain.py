import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File('cc_gain.h5', 'r') as hf:

    #experiments = ['01_ablation_horizontal']

    #for experiment in experiments:
    for experiment in hf.keys():
        experiment_group = hf[experiment]

        plt.figure(figsize=(14, 10))

        for data_type in experiment_group.keys():
            print('\tData Type: ', data_type)
            data_type_group = experiment_group[data_type]

            mean_cc_values = []

            for seis_event in data_type_group.keys():
                seis_event_group = data_type_group[seis_event]
                data_cc = seis_event_group['data_cc'][:]
                mean_cc = np.mean(data_cc)
                mean_cc_values.append(mean_cc)
                mean_cc_values = mean_cc_values[:40]
                #print(data_cc.shape)
                #data_cc_seis = seis_event_group['data_cc_seis'][:]
                #print('\t\tData CC Seis: ', data_cc_seis)


            ids = np.arange(1, 41)


            # Plot erstellen

            plt.scatter(ids, mean_cc_values, label = data_type)
        plt.xlabel('# Icequakes')
        plt.ylabel('CC gain [-]')
        plt.ylim(0, 5)
        plt.title('Experiment: ' + experiment)
        plt.grid(True)
        plt.legend()
        plt.show()
