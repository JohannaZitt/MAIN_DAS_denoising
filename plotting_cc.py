import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#experiments = os.listdir("experiments")
experiments = ["01_ablation_horizontal", "02_ablation_vertical", "03_accumulation_horizontal", "04_accumulation_vertical",
               "05_combined200", "06_combined800", "07_retrained_combined200", "08_retrained_combined800",
               "09_borehole_seismometer"]
values_ablation_cc_gain = {}
values_ablation_cc_gain_seis = {}
values_accumulation_cc_gain = {}
values_accumulation_cc_gain_seis = {}

'''

Extract Data:

'''
for experiment in experiments:

    csv_path = os.path.join("experiments", experiment, "cc_evaluation_" + experiment[0:2] + ".csv")

    with open(csv_path, "r") as file:

        csv_reader = csv.DictReader(file)

        ablation_cc_gain = []
        ablation_cc_gain_seis = []
        accumulation_cc_gain = []
        accumulation_cc_gain_seis = []

        for row in csv_reader:

            if int(row["id"]) <= 179 and row["zone"] == "ablation":
                ablation_cc_gain.append(float(row["mean_cc_gain"]))
                ablation_cc_gain_seis.append(float(row["mean_cross_gain"]))
            if int(row["id"]) <= 126 and row["zone"] == "accumulation":
                accumulation_cc_gain.append(float(row["mean_cc_gain"]))
                accumulation_cc_gain_seis.append(float(row["mean_cross_gain"]))

    values_ablation_cc_gain[experiment[3:]] = np.mean(ablation_cc_gain)
    values_ablation_cc_gain_seis[experiment[3:]] = np.mean(ablation_cc_gain_seis)
    values_accumulation_cc_gain[experiment[3:]] = np.mean(accumulation_cc_gain)
    values_accumulation_cc_gain_seis[experiment[3:]] = np.mean(accumulation_cc_gain_seis)

'''

Plot - cc_seis and local cc seperated

'''

datatypes = ["local", "seis"]
for datatype in datatypes:

    # Data:
    labels = ["Ablation\nHorizontal", "Ablation\nVertical", "Accumulation\nHorizontal", "Accumulation\nVertical"
              , "Combined 200", "Combined 800", "Retrained\nCombined 200", "Retrained\nCombined 800", "Borehole"]

    if datatype == "local":
        acc_cc_gain = list(values_accumulation_cc_gain.values())
        abl_cc_gain = list(values_ablation_cc_gain.values())
        label = "Local Waveform Coherence Gain []"
    else:
        acc_cc_gain = list(values_accumulation_cc_gain_seis.values())
        abl_cc_gain = list(values_ablation_cc_gain_seis.values())
        label = "Cross Correlation Seismometer Gain []"

    # Params:
    col_acc = "#008B93"
    col_abl = "#DF575F"
    width = 0.2  # Breite der Balken
    x = np.arange(len(labels))  # x-Koordinaten für die Balken
    fontsize = 12
    gain = 2 #fontsize gain

    plt.figure(figsize=(13, 6))
    plt.grid(axis="y", zorder = 0)
    plt.bar(x - 0.5 * width, abl_cc_gain, width=width, label="Ablation", color=col_abl, zorder=3)
    plt.bar(x + 0.5 * width, acc_cc_gain, width=width, label="Accumulation", color=col_acc, zorder=3)

    plt.axhline(y=1, color="blue", zorder=4)
    # Achsenbeschriftungen und Titel hinzufügen
    plt.ylabel(label, fontsize=fontsize+gain)
    plt.yticks((0.5, 1, 1.5, 2, 2.5), (0.5, 1.0, 1.5, 2.0, 2.5), fontsize=fontsize)
    plt.legend(fontsize=fontsize+gain, frameon=False, ncol=1, loc="upper left")
    plt.xticks(x, labels, rotation=20, fontsize=fontsize+1)

    print(label)
    print("Labels", labels)
    print("Accumulation: ", acc_cc_gain)
    print("Ablation: ", abl_cc_gain)



    # Diagramm anzeigen

    plt.tight_layout()
    plt.show()
    #plt.savefig("plots/cc/" + datatype + ".png", dpi=250)







'''

Pot Data - all in one




# Data:
labels = ["Ablation\nHorizontal", "Ablation\nVertical", "Accumulation\nHorizontal", "Accumulation\nVertical"
          , "Combined 120", "Combined 480", "Random 480"]
acc_cc_gain = list(values_accumulation_cc_gain.values())
acc_cc_gain_seis = list(values_accumulation_cc_gain_seis.values())
abl_cc_gain = list(values_ablation_cc_gain.values())
abl_cc_gain_seis = list(values_ablation_cc_gain_seis.values())

# Params:
col_acc_local = "#008B93"
col_acc_seis = "#9DE1E3"
col_abl_local = "#DF575F"
col_abl_seis = "#FFA09E"
width = 0.14  # Breite der Balken
x = np.arange(len(labels))  # x-Koordinaten für die Balken
fontsize = 12

plt.figure(figsize=(11, 6))
plt.grid(axis='y', zorder = 0)
plt.bar(x - 1.5 * width, acc_cc_gain, width=width, label='Acc Local', color = col_acc_local, zorder = 3)
plt.bar(x - 0.5 * width, acc_cc_gain_seis, width=width, label='Acc Seis', color = col_acc_seis, zorder = 3)
plt.bar(x + 0.5 * width, abl_cc_gain, width=width, label='Abl Local', color = col_abl_local, zorder = 3)
plt.bar(x + 1.5 * width, abl_cc_gain_seis, width=width, label='Abl Seis', color = col_abl_seis, zorder = 3)
plt.axhline(y=1, color = "blue", zorder=4)
# Achsenbeschriftungen und Titel hinzufügen
plt.ylabel('Gain []', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(x, labels, rotation=15, fontsize=fontsize)

custom_lines = [mpatches.Patch(facecolor=col_acc_local, edgecolor='black'),
                mpatches.Patch(facecolor=col_acc_seis, edgecolor='black'),
                mpatches.Patch(facecolor=col_abl_local, edgecolor='black'),
                mpatches.Patch(facecolor=col_abl_seis, edgecolor='black')
                ]
#plt.legend(custom_lines, ['Accumulation Local', 'Accumulation Seis', 'Ablation Local', 'Ablation Seis'], fontsize=fontsize,
#           loc='upper left', ncol=4, frameon = True, bbox_to_anchor=(0, 0.1), facecolor = 'white')

plt.legend(custom_lines, ['Accumulation Local', 'Accumulation Seis', 'Ablation Local', 'Ablation Seis'], fontsize=fontsize,
           loc='upper left', ncol=4, frameon = False, bbox_to_anchor=(0, 1.1))

# Diagramm anzeigen
#plt.show()
plt.savefig('plots/cc/all_in_one.png')
'''





