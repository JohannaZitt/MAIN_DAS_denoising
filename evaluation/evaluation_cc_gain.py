import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File('1_cc_gain.h5', 'r') as hf:

    #experiments = ['01_ablation_horizontal']

    #for experiment in experiments:
    for experiment in hf.keys():
        experiment_group = hf[experiment]

        plt.figure(figsize=(14, 10))

        for data_type in experiment_group.keys():
            data_type_group = experiment_group[data_type]

            cc_values = []

            for seis_event in data_type_group.keys():
                seis_event_group = data_type_group[seis_event]
                data_cc = seis_event_group['data_cc'][:]
                mean_cc = np.mean(data_cc)
                sd_cc = np.std(data_cc)
                values = [mean_cc, sd_cc]
                cc_values.append(values)

                #print(data_cc.shape)
                #data_cc_seis = seis_event_group['data_cc_seis'][:]
                #print('\t\tData CC Seis: ', data_cc_seis)

            cc_values = sorted(cc_values, key=lambda x: x[0])
            cc_values = cc_values[:40]
            cc_means = [item[0] for item in cc_values]
            cc_stds = [item[1] for item in cc_values]
            ids = np.arange(1, 41)

            print(np.shape(cc_stds))
            print(np.shape(cc_means))

            # Plot erstellen

            plt.scatter(ids, cc_means, label = data_type)
            plt.axhline(y = cc_means + cc_stds, color='green', linestyle='--')
            plt.axhline(y = cc_means - cc_stds, color='green', linestyle='--')

        plt.xlabel('# Icequakes')
        plt.ylabel('CC gain [-]')
        plt.ylim(0, 5)
        plt.title('Experiment: ' + experiment)
        plt.grid(True)
        plt.legend()
        plt.show()
