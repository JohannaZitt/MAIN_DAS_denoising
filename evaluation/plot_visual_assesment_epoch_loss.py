import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import json




'''

1. Visual Assesment 

'''

models = ['Ablation\nHorizontal', 'Ablation\nVertical', 'Accumulation\nVertical', ' Accumulation\nHorizontal', 'Stick-slip',
          'Surface', 'Combined120', 'Combined480', 'Random480']
data_types = ['stick-slip ablation', 'stick-slip accumulation', 'surface accumulation', 'surface ablation']
stick_slip_ablation = [4/49, 2/49, 6/49, 8/49, 6/49, 4/49, 4.5/49, 5.5/49, 2.5/49]
stick_slip_accumulation = [4/41, 1.5/41, 8.5/41, 10.5/41, 8/41, 5/41, 5/41, 5/41, 4.5/41]
surface_ablation = [2/44, 1.5/44, 4/44, 5.5/44, 3.5/44, 4/44, 2/44, 2/44, 2.5/44]
surface_accumulation = [3/45, 1/45, 4/45, 3/45, 3.5/45, 1/45, 1.5/45, 2/45, 2/45]
# bereits in rohen visible icequakes:
raw_visible_quakes = [1/49, 18/41, 0, 36/45]
visual_assesment_data = np.column_stack((stick_slip_ablation, stick_slip_accumulation, surface_accumulation, surface_ablation))
visual_assesment_data_small = np.column_stack((stick_slip_ablation, stick_slip_accumulation))

# plotting:
bar_width = 0.28
fontsize = 12

# Positionen der Balken
bar_positions = np.arange(len(models))

# Farben für die Balken
#blue = '#06a237'
#green = '#064ca2'
#light_blue = '#08e74f'
#light_green = '#4998f8'

colors = ['#66B2FF', '#0000FF', '#009900', '#66CC00']  # Beispiel: Rote, grüne, blaue und gelbe Balken


fontsize = 16

''' 1a  All - only category 2 

# Plot des Balkendiagramms für jede Kategorie und jeden Wert
plt.figure(figsize=(10, 6))

for i in range(4):
    plt.bar(bar_positions + i * bar_width, visual_assesment_data[:, i], width=bar_width, color=colors[i], label=data_types[i], zorder=2)

# Beschriftungen und Titel hinzufügen
plt.ylabel('increase in events', size=fontsize)
plt.xticks(bar_positions + 0.3, models, rotation=15, fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.grid(axis='y', zorder=0)  # Gitterlinien für die y-Achse anzeigen
plt.tight_layout()
#plt.savefig('plots/visual_assesnent.png')
plt.show()
'''

''' 1b All - category 1 and 2 
plt.figure(figsize=(10, 6))

# Plot der Basislinie
blue = '#06a237'
green = '#064ca2'
light_blue = '#08e74f'
light_green = '#4998f8'
colors = [blue, green, blue, green]
colors_light = [light_blue, light_green, light_blue, light_green]
data_types_raw = ['raw stick-slip ablation', 'raw stick-slip accumulation', 'raw surface ablation', 'raw surface accumulation']

# Plot der anderen Daten
for i in range(4):
    if data_types[i]=='stick-slip ablation' or data_types[i] == 'stick-slip accumulation':
        plt.bar(bar_positions + i * bar_width, np.full(len(bar_positions), raw_visible_quakes[i]), width=bar_width,
                color=colors_light[i], hatch='//', zorder=3)
        plt.bar(bar_positions + i * bar_width, visual_assesment_data[:, i] + raw_visible_quakes[i], width=bar_width,
                color=colors[i], hatch='\\\\', zorder=2)
    else:
        plt.bar(bar_positions + i * bar_width, np.full(len(bar_positions), raw_visible_quakes[i]), width=bar_width,
                color=colors_light[i], zorder=3)
        plt.bar(bar_positions + i * bar_width, visual_assesment_data[:, i] + raw_visible_quakes[i], width=bar_width,
                color=colors[i], zorder=2)

# Custom Legend:
custom_lines = [mpatches.Patch(facecolor='none', hatch='//', edgecolor='black'),
                mpatches.Patch(facecolor='none', edgecolor='black'),
                mpatches.Patch(facecolor=light_green, edgecolor='black'),
                mpatches.Patch(facecolor=green, edgecolor='black'),
                mpatches.Patch(facecolor=light_blue, edgecolor='black'),
                mpatches.Patch(facecolor=blue, edgecolor='black')]
plt.legend(custom_lines, ['stick-slip', 'surface', 'accumulation-raw', 'accumulation', 'ablation-raw', 'ablation'], fontsize=fontsize)

# Beschriftungen und Titel hinzufügen
plt.ylabel('increase in events', size=fontsize)
plt.xticks(bar_positions + 0.3, models, rotation=15, fontsize=fontsize)
plt.grid(axis='y', zorder=0)  # Gitterlinien für die y-Achse anzeigen
plt.tight_layout()
#plt.savefig('plots/visual_assesnent_1_and_2.png')
plt.show()
'''


''' 1c All - category 1 and 2 vertauscht'''
plt.figure(figsize=(16.2, 6))

# Plot der Basislinie
col_acc = '#008B93'#'#507EB9'#'#67A2E7' # '#9CB3FF'
col_acc_light = '#9DE1E3'#'#A7D7FF' #'#AAE0FF' # '#DCE9FC'

col_abl = '#DF575F' #'#CC6B33'#'#7CD45F' # '#9CFFA9'
col_abl_light = '#FFA09E' #'#FFB351'#'#A0F880' #'#D3F2D7'

colors = [col_abl, col_acc, col_abl, col_acc]
colors_light = [col_abl_light, col_acc_light, col_abl_light, col_acc_light]
bar_positions = np.arange(1, 15, 1.7)
print(bar_positions)
alpha = 0.03
for i in range(4):
    if data_types[i]=='stick-slip ablation' or data_types[i] == 'stick-slip accumulation':
        plt.bar(bar_positions + i * bar_width, np.full(len(bar_positions), visual_assesment_data[:, i]), width=bar_width - alpha,
                color=colors[i], hatch='///',  linewidth = 0, edgecolor = colors_light[i], zorder=3) # edgecolor = 'grey',
        plt.bar(bar_positions + i * bar_width, visual_assesment_data[:, i] + raw_visible_quakes[i], width=bar_width - alpha,
                color=colors_light[i], hatch='///', linewidth = 0, edgecolor = colors[i], zorder=2)
    else:
        plt.bar(bar_positions + i * bar_width, np.full(len(bar_positions), visual_assesment_data[:, i]), width=bar_width- alpha,
                color=colors[i], zorder=3)
        plt.bar(bar_positions + i * bar_width, visual_assesment_data[:, i] + raw_visible_quakes[i], width=bar_width- alpha,
                color=colors_light[i], zorder=2)

# Custom Legend:
custom_lines = [mpatches.Patch(facecolor='none', hatch='//', edgecolor='black'),
                mpatches.Patch(facecolor='none', edgecolor='black'),
                mpatches.Patch(facecolor=col_acc, edgecolor='black'),
                mpatches.Patch(facecolor=col_acc_light, edgecolor='black'),
                mpatches.Patch(facecolor=col_abl, edgecolor='black'),
                mpatches.Patch(facecolor=col_abl_light, edgecolor='black')
                ]
plt.legend(custom_lines, ['Stick-slip', 'Surface', 'Accumulation', 'Accumulation raw', 'Ablation', 'Ablation raw'], fontsize=fontsize, loc='upper left', ncol=6, frameon = False, bbox_to_anchor=(0, 1.15))

# Beschriftungen und Titel hinzufügen
plt.ylabel('Detection [%]', size=fontsize)
plt.xticks(bar_positions + 0.3, models, rotation=15, fontsize=fontsize)
#plt.yscale('log')
plt.yticks([0.2, 0.4, 0.6, 0.8], [20, 40, 60, 80], fontsize=fontsize-4)
plt.grid(axis='y', zorder=0)  # Gitterlinien für die y-Achse anzeigen
plt.tight_layout()
plt.savefig('plots/visual_assesnent_1_and_2_vertauscht.png')
plt.show()



''' 1c. Visual Assesment ONLY STICK-SLIP 


# Farben für die Balken
colors = ['#66B2FF', '#0000FF', '#009900', '#66CC00']  # Beispiel: Rote, grüne, blaue und gelbe Balken

# Plot des Balkendiagramms für jede Kategorie und jeden Wert
plt.figure(figsize=(10, 6))

for i in range(2):
    plt.bar(bar_positions + i * bar_width, visual_assesment_data_small[:, i], width=bar_width, color=colors[i], label=data_types[i], zorder=2)

# Beschriftungen und Titel hinzufügen
plt.ylabel('increase in stick-slip events', size=fontsize)
plt.xticks(bar_positions + 0.1, models, rotation=15, fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.grid(axis='y', zorder=0)  # Gitterlinien für die y-Achse anzeigen
plt.tight_layout()
# plt.savefig('plots/visual_assesnent_small.png')
# plt.show()
'''

'''
2. Epoch Loss


train_last = [0.0612, 0.0621, 0.0637, 0.0634, 0.0661, 0.063, 0.0638, 0.0629, 0.0624]
validation_last = [0.0612, 0.0637, 0.0643, 0.0654, 0.0648, 0.0615, 0.0623, 0.0644, 0.0634]
train_mean = [0.0621, 0.063, 0.0644, 0.0653, 0.0661, 0.0634, 0.0652, 0.0632, 0.0626]
validation_mean = [0.0621, 0.0648, 0.0647, 0.0652, 0.0669, 0.0619, 0.0638, 0.0639, 0.0629]
epoch_loss_data = np.column_stack((train_last, validation_last, train_mean, validation_mean))

bar_width = 0.2

# Positionen der Balken
bar_positions = np.arange(len(models))

# Farben für die Balken
colors = ['r', 'g', 'b', 'y']  # Beispiel: Rote, grüne, blaue und gelbe Balken

# Plot des Balkendiagramms für jede Kategorie und jeden Wert
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.bar(bar_positions + i * bar_width, epoch_loss_data[:, i], width=bar_width, color=colors[i], label=data_types[i])

# Beschriftungen und Titel hinzufügen
plt.xlabel('training data type')
plt.ylabel('epoch loss')
plt.ylim([0.06, 0.07])
#plt.title('')
plt.xticks(bar_positions + 0.3, models, rotation=15) #  rotation=45, ha='left'
plt.legend()

plt.grid(axis='y')  # Gitterlinien für die y-Achse anzeigen
plt.tight_layout()
#plt.savefig('plots/epoch_loss.png')
plt.show()
'''

'''
3. CC = Local Waveform coherence


with open('cc_experiment_means.json', 'r') as file:
    data = json.load(file)

# Konvertiere Daten in ein 9x4 Numpy-Array
cc_data = np.array([[data[key][sub_key] for sub_key in data[key]] for key in data])

bar_width = 0.2

# Positionen der Balken
bar_positions = np.arange(len(models))

# Farben für die Balken
colors = ['r', 'g', 'b', 'y']  # Beispiel: Rote, grüne, blaue und gelbe Balken

# Plot des Balkendiagramms für jede Kategorie und jeden Wert
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.bar(bar_positions + i * bar_width, cc_data[:, i], width=bar_width, color=colors[i], label=data_types[i])

# Beschriftungen und Titel hinzufügen
plt.xlabel('training data type')
plt.ylabel('cc gain [-]')
plt.xticks(bar_positions + 0.3, models, rotation=15) #  rotation=45, ha='left'
plt.legend()

plt.grid(axis='y')  # Gitterlinien für die y-Achse anzeigen
plt.tight_layout()
#plt.savefig('plots/cc_gain_loss.png')
plt.show()
'''

'''
4. CC = Local Waveform coherence: accumulation surface and stick-slip zusammenfassen und ablation surface und stick-slip zusammengfasst
'''

with open('cc_experiment_means.json', 'r') as file:
    data = json.load(file)

key_order = ['stick-slip_ablation', 'stick-slip_accumulation', 'surface_ablation', 'surface_accumulation']

res_abl = []
res_acc = []


for key in data:
    vals = [data[key][subkey] for subkey in key_order]
    abl = (vals[0] + vals[2]) / 2  # Mittelwert für Ablation
    acc = (vals[1] + vals[3]) / 2  # Mittelwert für Akkumulation
    res_abl.append(abl)
    res_acc.append(acc)

cc_data_small = np.column_stack([res_acc, res_abl])

bar_width = 0.2
# Positionen der Balken
bar_positions = np.arange(len(models))
# Farben für die Balken
colors = [col_acc, col_abl]
# Labels für Legende
data_types_small = ['Accumulation zone', 'Ablation zone']

# Plot des Balkendiagramms für jede Kategorie und jeden Wert
plt.figure(figsize=(16.2, 6))
for i in range(2):
    plt.bar(bar_positions + i * bar_width, cc_data_small[:, i], width=bar_width-alpha, color=colors[i], label=data_types_small[i], zorder=i+2)

# Beschriftungen und Titel hinzufügen
plt.ylabel('CC gain []', size=fontsize)
plt.axhline(y=1, color = 'black', zorder=5, )
plt.grid(axis='y', zorder=0)
plt.xticks(bar_positions + 0.1, models, rotation=15, size=fontsize) #  rotation=45, ha='left'
plt.legend(fontsize=fontsize, loc='upper left', ncol=6, frameon = False, bbox_to_anchor=(0, 1.15))


plt.tight_layout()
plt.savefig('plots/cc_gain_loss_acc_vs_ab.png')
plt.show()
