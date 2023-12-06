import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

def compute_data_cc_seis_gain(raw, denoised):
    max_mean = -1
    trace_index = -1
    for i in range(denoised.shape[0]):
        # calculate which trace should be considered
        denoised_mean = np.mean(denoised[i])
        if max_mean < denoised_mean:
            max_mean = denoised_mean
            trace_index = i

    objective_trace_denoised = denoised[trace_index]
    objective_trace_raw = raw[trace_index]

    max_value_denoised = np.max(objective_trace_denoised) + 1
    max_value_index = np.argmax(objective_trace_denoised)
    max_value_raw = objective_trace_raw[max_value_index] + 1

    gain = max_value_denoised / max_value_raw

    return gain

with h5py.File('cc_gain_naive.h5', 'r') as hf:



    '''
    ###############################################################
    #########  SINGLE PLOTS, EACH FOR ONE MODEL  ##################
    ###############################################################
    
    for experiment in hf.keys():
        experiment_group = hf[experiment]

        plt.figure(figsize=(14, 10))

        for data_type in experiment_group.keys():
            data_type_group = experiment_group[data_type]

            means_cc = []

            for seis_event in data_type_group.keys():
                seis_event_group = data_type_group[seis_event]
                data_cc = seis_event_group['data_cc'][:]
                mean_cc = np.mean(data_cc)
                means_cc.append(mean_cc)

                #print(data_cc.shape)
                #data_cc_seis = seis_event_group['data_cc_seis'][:]
                #print('\t\tData CC Seis: ', data_cc_seis)

            means_cc = sorted(means_cc)
            means_cc = means_cc[:40]

            ids = np.arange(1, 41)

            # Plot erstellen
            plt.scatter(ids, means_cc, label = data_type)

        # Font:
        s_font = 15
        m_font = 18
        l_font = 21
        plt.xlabel('# Icequakes', size=s_font)
        plt.ylabel('CC gain [-]', size=s_font)
        plt.ylim(0, 5)
        plt.title('Experiment: ' + experiment, size=l_font)
        plt.grid(True)
        plt.legend(loc='upper left', fontsize=s_font)

        # save figure
        plt.show()
        #plt.savefig('plots/plot_for_every_model/' + experiment)
    '''

    '''
    ###############################################################
    ###########  ONE PLOT FOR ALL MODELS  #########################
    ###############################################################

    fig, axes = plt.subplots(3, 3, figsize=(25, 18))

    for i, experiment in enumerate(hf.keys()):
        experiment_group = hf[experiment]

        row = i // 3
        col = i % 3

        ax = axes[row, col]

        for data_type in experiment_group.keys():
            data_type_group = experiment_group[data_type]
            means_cc = []

            for seis_event in data_type_group.keys():
                seis_event_group = data_type_group[seis_event]
                data_cc = seis_event_group['data_cc'][:]
                mean_cc = np.mean(data_cc)
                means_cc.append(mean_cc)

            means_cc = sorted(means_cc)
            means_cc = means_cc[:40]

            ids = np.arange(1, 41)

            # Plot
            ax.scatter(ids, means_cc, label=data_type)

        ax.set_xlabel('# Icequakes', size=15)
        ax.set_ylabel('CC gain [-]', size=15)
        ax.set_ylim(0, 5)
        ax.set_title('Experiment: ' + experiment, size=18)
        ax.grid(True)
        ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    # Save plot
    #plt.show()
    plt.savefig('plots/model_plot')
    '''

    '''
    ###############################################################
    ###########  SINGLE PLOTS FOR DATA TYPES  #####################
    ###############################################################
    
    experiments = ['01_ablation_horizontal', '02_ablation_vertical', '03_accumulation_vertical', '04_accumulation_horizontal', '05_stick-slip', '06_surface', '07_combined120', '08_combined480', '09_random480']
    data_types = ['stick-slip_ablation', 'surface_ablation', 'stick-slip_accumulation', 'surface_accumulation']
    #data_types = data_types[0:1]

    for data_type in data_types:

        plt.figure(figsize=(14, 10))

        for i, experiment in enumerate(experiments):
            experiment_group = hf[experiment]

            means_cc = []
            for seis_event in experiment_group[data_type].keys():
                seis_event_group = experiment_group[data_type][seis_event]
                data_cc = seis_event_group['data_cc'][:]
                mean_cc = np.mean(data_cc)
                means_cc.append(mean_cc)

            means_cc = means_cc[:40]

            if i == 0:
                index = [i for i, _ in sorted(enumerate(means_cc), key=lambda x: x[1])]
            means_cc = [means_cc[j] for j in index]


            ids = np.arange(1, 41)

            # Plot
            plt.scatter(ids, means_cc, label=experiment)

        plt.xlabel('# Icequakes', size=15)
        plt.ylabel('CC gain [-]', size=15)
        plt.ylim(0, 5)
        plt.title('Data Type: ' + data_type, size=18)
        plt.grid(True)
        plt.legend(loc='upper left', fontsize=12)
        plt.show()
    '''

    '''
    ###############################################################
    ###########  ONE PLOT FOR ALL DATA TYPES  #####################
    ###############################################################
    '''

    #experiments = ['01_ablation_horizontal', '02_ablation_vertical', '03_accumulation_vertical', '04_accumulation_horizontal', '05_stick-slip', '06_surface', '07_combined120', '08_combined480', '09_random480']

    #experiments = ['01_ablation_horizontal', '02_ablation_vertical', '03_accumulation_vertical', '04_accumulation_horizontal']
    #experiments = ['05_stick-slip', '06_surface']
    #experiments = ['07_combined120', '08_combined480', '09_random480']
    #experiments = ['02_ablation_vertical', '07_combined120', '08_combined480']
    #experiments = ['02_ablation_vertical', '09_random480']
    experiments = ['03_accumulation_vertical', '07_combined120', '08_combined480']


    data_types = ['stick-slip_ablation', 'surface_ablation', 'stick-slip_accumulation', 'surface_accumulation']

    fig, axes = plt.subplots(2, 2, figsize=(25, 18))

    for ax, data_type in zip(axes.flatten(), data_types):
        for i, experiment in enumerate(experiments):
            experiment_group = hf[experiment]

            means_cc = []
            for seis_event in experiment_group[data_type].keys():
                seis_event_group = experiment_group[data_type][seis_event]
                data_cc = seis_event_group['data_cc'][:]
                mean_cc = np.mean(data_cc)
                means_cc.append(mean_cc)

            means_cc = means_cc[:40]

            if i == 0:
                index = [i for i, _ in sorted(enumerate(means_cc), key=lambda x: x[1])]
            means_cc = [means_cc[j] for j in index]

            ids = np.arange(1, 41)

            # Plot
            ax.scatter(ids, means_cc, label=experiment)
            ax.set_xlabel('# Icequakes', size=15)
            ax.set_ylabel('CC gain [-]', size=15)
            ax.set_ylim(0, 5)
            ax.set_title('Data Type: ' + data_type, size=18)
            ax.grid(True)
            ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    #plt.show()
    plt.savefig('plots/plot_for_every_data_type/' + 'accumulation_vs_random_model')


    '''
    ###############################################################
    #########  SEIS!!! SINGLE PLOTS, EACH FOR ONE MODEL  ##########
    ###############################################################

    for experiment in hf.keys():
        experiment_group = hf[experiment]

        plt.figure(figsize=(14, 10))

        for data_type in experiment_group.keys():
            data_type_group = experiment_group[data_type]

            cc_seis_gain = []

            for seis_event in data_type_group.keys():
                seis_event_group = data_type_group[seis_event]

                data_cc_seis_raw = seis_event_group['data_cc_seis_raw'][:]
                data_cc_seis_denoised = seis_event_group['data_cc_seis_denoised'][:]

                data_cc_seis_gain = compute_data_cc_seis_gain(data_cc_seis_raw, data_cc_seis_denoised)

                cc_seis_gain.append(data_cc_seis_gain)

            ids = np.arange(1, 41)
            cc_seis_gain = sorted(cc_seis_gain)
            cc_seis_gain = cc_seis_gain[:40]

            # Plot erstellen
            plt.scatter(ids, cc_seis_gain, label=data_type)

        # Font:
        s_font = 15
        m_font = 18
        l_font = 21
        plt.xlabel('# Icequakes', size=s_font)
        plt.ylabel('CC gain [-]', size=s_font)
        plt.ylim(0.75, 1.5)
        plt.title('Experiment: ' + experiment, size=l_font)
        plt.grid(True)
        plt.legend(loc='upper left', fontsize=s_font)

        # save figure
        #plt.show()
        plt.savefig('plots/seis_plot_for_every_model/seis_' + experiment)
    '''

    '''
    ###############################################################
    ###########  SEIS ONE PLOT FOR ALL MODELS  ####################
    ###############################################################
    

    fig, axes = plt.subplots(3, 3, figsize=(25, 18))

    for i, experiment in enumerate(hf.keys()):
        experiment_group = hf[experiment]

        row = i // 3
        col = i % 3

        ax = axes[row, col]

        for data_type in experiment_group.keys():
            data_type_group = experiment_group[data_type]
            cc_seis_gain = []

            for seis_event in data_type_group.keys():
                seis_event_group = data_type_group[seis_event]
                data_cc_seis_raw = seis_event_group['data_cc_seis_raw'][:]
                data_cc_seis_denoised = seis_event_group['data_cc_seis_denoised'][:]

                data_cc_seis_gain = compute_data_cc_seis_gain(data_cc_seis_raw, data_cc_seis_denoised)

                cc_seis_gain.append(data_cc_seis_gain)

            cc_seis_gain = sorted(cc_seis_gain)
            cc_seis_gain = cc_seis_gain[:40]
            ids = np.arange(1, 41)

            # Plot
            ax.scatter(ids, cc_seis_gain, label=data_type)
            #plt.ylim(0.75, 1.5)

        ax.set_ylim(0.75, 1.5)
        ax.set_xlabel('# Icequakes', size=15)
        ax.set_ylabel('CC gain [-]', size=15)
        ax.set_title('Experiment: ' + experiment, size=18)
        ax.grid(True)
        ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    # Save plot
    #plt.show()
    plt.savefig('plots/seis_plot_for_every_model/seis_model_plot')
    '''


    '''
    ###############################################################
    ###########  SEIS ONE PLOT FOR ALL DATA TYPES  ################
    ###############################################################
    

    #experiments = ['01_ablation_horizontal', '02_ablation_vertical', '03_accumulation_vertical', '04_accumulation_horizontal', '05_stick-slip', '06_surface', '07_combined120', '08_combined480', '09_random480']

    #experiments = ['01_ablation_horizontal', '02_ablation_vertical', '03_accumulation_vertical', '04_accumulation_horizontal']
    #experiments = ['05_stick-slip', '06_surface']
    experiments = ['07_combined120', '08_combined480', '09_random480']


    data_types = ['stick-slip_ablation', 'surface_ablation', 'stick-slip_accumulation', 'surface_accumulation']

    fig, axes = plt.subplots(2, 2, figsize=(25, 18))

    for ax, data_type in zip(axes.flatten(), data_types):
        for i, experiment in enumerate(experiments):
            experiment_group = hf[experiment]

            cc_seis_gain = []
            for seis_event in experiment_group[data_type].keys():
                seis_event_group = experiment_group[data_type][seis_event]
                data_cc_seis_raw = seis_event_group['data_cc_seis_raw'][:]
                data_cc_seis_denoised = seis_event_group['data_cc_seis_denoised'][:]

                data_cc_seis_gain = compute_data_cc_seis_gain(data_cc_seis_raw, data_cc_seis_denoised)

                cc_seis_gain.append(data_cc_seis_gain)

            cc_seis_gain = cc_seis_gain[:40]

            if i == 0:
                index = [i for i, _ in sorted(enumerate(cc_seis_gain), key=lambda x: x[1])]
            cc_seis_gain = [cc_seis_gain[j] for j in index]

            ids = np.arange(1, 41)

            # Plot
            ax.scatter(ids, cc_seis_gain, label=experiment)
            ax.set_xlabel('# Icequakes', size=15)
            ax.set_ylabel('CC gain [-]', size=15)
            ax.set_ylim(0.75, 2.2)
            ax.set_title('Data Type: ' + data_type, size=18)
            ax.grid(True)
            ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    #plt.show()
    plt.savefig('plots/seis_plot_for_every_data_type/' + 'random')
    '''

