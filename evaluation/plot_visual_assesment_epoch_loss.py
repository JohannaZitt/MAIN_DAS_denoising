import numpy as np
import matplotlib.pyplot as plt
import json




'''

1. Visual Assesment ALL
'''

models = ['ablation\nhorizontal', 'ablation\nvertical', 'accumulation\nvertical', ' accumulation\nhorizontal', 'stick-slip',
          'surface', 'combined120', 'combined480', 'random480']
data_types = ['stick-slip ablation', 'stick-slip accumulation', 'surface ablation', 'surface accumulation']
stick_slip_ablation = [4/49, 2/49, 6/49, 8/49, 6/49, 4/49, 4.5/49, 5.5/49, 2.5/49]
stick_slip_accumulation = [4/41, 1.5/41, 8.5/41, 10.5/41, 8/41, 5/41, 5/41, 5/41, 4.5/41]
surface_ablation = [2/44, 1.5/44, 4/44, 5.5/44, 3.5/44, 4/44, 2/44, 2/44, 2.5/44]
surface_accumulation = [3/45, 1/45, 4/45, 3/45, 3.5/45, 1/45, 1.5/45, 2/45, 2/45]
# bereits in rohen visible icequakes:
raw_visible_quakes = [1/49, 18/41, 0, 36/45]
visual_assesment_data = np.column_stack((stick_slip_ablation, stick_slip_accumulation, surface_ablation, surface_accumulation))

# plotting:
bar_width = 0.2

# Positionen der Balken
bar_positions = np.arange(len(models))

# Farben für die Balken
colors = ['#66B2FF', '#0000FF', '#009900', '#66CC00']  # Beispiel: Rote, grüne, blaue und gelbe Balken

# Plot des Balkendiagramms für jede Kategorie und jeden Wert
plt.figure(figsize=(10, 6))
fontsize = 12
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

1. Visual Assesment ONLY STICK-SLIP


models = ['ablation\nhorizontal', 'ablation\nvertical', 'accumulation\nvertical', ' accumulation\nhorizontal', 'stick-slip',
          'surface', 'combined120', 'combined480', 'random480']
data_types = ['ablation', 'accumulation']
stick_slip_ablation = [4/49, 2/49, 6/49, 8/49, 6/49, 4/49, 4.5/49, 5.5/49, 2.5/49]
stick_slip_accumulation = [4/41, 1.5/41, 8.5/41, 10.5/41, 8/41, 5/41, 5/41, 5/41, 4.5/41]
surface_ablation = [2/44, 1.5/44, 4/44, 5.5/44, 3.5/44, 4/44, 2/44, 2/44, 2.5/44]
surface_accumulation = [3/45, 1/45, 4/45, 3/45, 3.5/45, 1/45, 1.5/45, 2/45, 2/45]
visual_assesment_data_small = np.column_stack((stick_slip_ablation, stick_slip_accumulation))

# plotting:
bar_width = 0.2

# Positionen der Balken
bar_positions = np.arange(len(models))

# Farben für die Balken
colors = ['#66B2FF', '#0000FF', '#009900', '#66CC00']  # Beispiel: Rote, grüne, blaue und gelbe Balken

# Plot des Balkendiagramms für jede Kategorie und jeden Wert
plt.figure(figsize=(10, 6))
fontsize = 12
for i in range(2):
    plt.bar(bar_positions + i * bar_width, visual_assesment_data_small[:, i], width=bar_width, color=colors[i], label=data_types[i], zorder=2)

# Beschriftungen und Titel hinzufügen
plt.ylabel('increase in stick-slip events', size=fontsize)
plt.xticks(bar_positions + 0.1, models, rotation=15, fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.grid(axis='y', zorder=0)  # Gitterlinien für die y-Achse anzeigen
plt.tight_layout()
# plt.savefig('plots/visual_assesnent_small.png')
plt.show()
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
#plt.show()
'''
'''
4. CC = Local Waveform coherence: accumulation surface and stick-slip zusammenfassen und ablation surface und stick-slip zusammengfasst




with open('cc_experiment_means.json', 'r') as file:
    data = json.load(file)

key_order = ['stick-slip_ablation', 'stick-slip_accumulation', 'surface_ablation', 'surface_accumulation']

res_abl = []
res_acc = []
fontsize = 13

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
colors = ['#3399FF', '#0000CC']
# Labels für Legende
data_types_small = ['accumulation zone', 'ablation zone']

# Plot des Balkendiagramms für jede Kategorie und jeden Wert
plt.figure(figsize=(10, 6))
for i in range(2):
    plt.bar(bar_positions + i * bar_width, cc_data_small[:, i], width=bar_width, color=colors[i], label=data_types_small[i], zorder=i+2)

# Beschriftungen und Titel hinzufügen
plt.ylabel('cc gain []', size=fontsize)
plt.axhline(y=1, color = 'black', zorder=1)
plt.grid(axis='y', zorder=0)
plt.xticks(bar_positions + 0.1, models, rotation=15, size=fontsize) #  rotation=45, ha='left'
plt.legend(fontsize=fontsize)


plt.tight_layout()
#plt.savefig('plots/cc_gain_loss_acc_vs_ab.png')
plt.show()
'''