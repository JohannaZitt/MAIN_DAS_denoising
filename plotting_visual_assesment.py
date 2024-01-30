import os
import matplotlib.pyplot as plt
import numpy as np

experiments = os.listdir("experiments")
experiments.sort()
values_acc = {}
values_abl = {}


for experiment in experiments:

    ablation_path = os.path.join("experiments", experiment, "plots", "ablation", "0706_RA88")
    accumulation_path = os.path.join("experiments", experiment, "plots", "accumulation", "0706_AJP")

    categories = ["1_raw:visible_denoised:better_visible",
                  "2_raw:not_visible_denoised:visible",
                  "3_raw:not_visible:denoised:not_visible"]

    values_category_acc = {}
    values_category_abl = {}

    for category in categories:

        category_size_acc = len(os.listdir(os.path.join(accumulation_path, category)))
        category_size_abl = len(os.listdir(os.path.join(ablation_path, category)))
        values_category_acc[category] = category_size_acc
        values_category_abl[category] = category_size_abl

    values_acc[experiment] = values_category_acc
    values_abl[experiment] = values_category_abl

print("Accumulation: ", values_acc)
print("Ablation: ", values_abl)

'''

Plotting Data - Accumulation



# Extrahiere die Kategorien und Werte
categories = ["Ablation\nHorizontal", "Ablation\nVertical", "Accumulation\nHorizontal", "Accumulation\nVertical"
              , "Combined 120", "Combined 480", "Random 480"]
raw_visible = [data['1_raw:visible_denoised:better_visible'] for data in values_acc.values()]
denoised_successfull = [data['2_raw:not_visible_denoised:visible'] for data in values_acc.values()]
not_visible = [data['3_raw:not_visible:denoised:not_visible'] for data in values_acc.values()]

# Plot
bar_width = 0.5
col_acc = "#008B93"
bar_positions = np.arange(len(categories))
fontsize = 12

plt.figure(figsize=(10, 6))
plt.bar(bar_positions, denoised_successfull, bar_width, label='denoised successfully', color=col_acc, alpha = 1)
plt.bar(bar_positions, not_visible, bar_width, bottom= np.array(denoised_successfull), label='not visible on raw nor denoised', color=col_acc, alpha = 0.5)
plt.bar(bar_positions, raw_visible, bar_width, bottom=np.array(denoised_successfull) + np.array(not_visible), label='visible on raw data', color=col_acc, alpha = 0.3)
plt.ylabel('# Events', fontsize=fontsize+1)
plt.yticks(fontsize=fontsize)
plt.xticks(bar_positions, categories, fontsize=fontsize, rotation=15)
plt.legend(fontsize=fontsize, frameon = False, ncol = 3, loc='upper left', bbox_to_anchor=(0.017, 1.1))

plt.tight_layout()
plt.savefig("plots/visual_assesment/accumulation.png")
#plt.show()

'''


'''

Plotting Data - Accumulation



# Extrahiere die Kategorien und Werte
categories = ["Ablation\nHorizontal", "Ablation\nVertical", "Accumulation\nHorizontal", "Accumulation\nVertical"
              , "Combined 120", "Combined 480", "Random 480"]
raw_visible = [data['1_raw:visible_denoised:better_visible'] for data in values_abl.values()]
denoised_successfull = [data['2_raw:not_visible_denoised:visible'] for data in values_abl.values()]
not_visible = [data['3_raw:not_visible:denoised:not_visible'] for data in values_abl.values()]

# Plot
bar_width = 0.5
col_abl = "#DF575F"
bar_positions = np.arange(len(categories))
fontsize = 12

plt.figure(figsize=(10, 6))
#plt.grid(axis='y', zorder = 0)
plt.bar(bar_positions, denoised_successfull, bar_width, label='denoised successfully', color=col_abl, alpha = 1, zorder=3)
plt.bar(bar_positions, not_visible, bar_width, bottom= np.array(denoised_successfull), label='not visible on raw nor denoised', color=col_abl, alpha = 0.5, zorder=3)
plt.bar(bar_positions, raw_visible, bar_width, bottom=np.array(denoised_successfull) + np.array(not_visible), label='visible on raw data', color=col_abl, alpha = 0.3, zorder=3)
plt.ylabel('# Events', fontsize=fontsize+1)
plt.yticks(fontsize=fontsize)
plt.xticks(bar_positions, categories, fontsize=fontsize, rotation=15)
plt.legend(fontsize=fontsize, frameon = False, ncol = 3, loc='upper left', bbox_to_anchor=(0.017, 1.1))

plt.tight_layout()
plt.savefig("plots/visual_assesment/ablation.png")
#plt.show()
'''

'''

Combined Plot:

'''

# Die Daten für die beiden Plots
categories = ["Ablation\nHorizontal", "Ablation\nVertical", "Accumulation\nHorizontal", "Accumulation\nVertical",
              "Combined 120", "Combined 480", "Random 480"]

raw_visible_acc = [data['1_raw:visible_denoised:better_visible'] for data in values_acc.values()]
denoised_successfull_acc = [data['2_raw:not_visible_denoised:visible'] for data in values_acc.values()]
not_visible_acc = [data['3_raw:not_visible:denoised:not_visible'] for data in values_acc.values()]

raw_visible_abl = [data['1_raw:visible_denoised:better_visible'] for data in values_abl.values()]
denoised_successfull_abl = [data['2_raw:not_visible_denoised:visible'] for data in values_abl.values()]
not_visible_abl = [data['3_raw:not_visible:denoised:not_visible'] for data in values_abl.values()]

# Breite der Balken
col_acc = "#008B93"
col_abl = "#DF575F"
fontsize=12
bar_width = 0.25
bar_gap = 0.05
bar_positions_acc = np.arange(len(categories))
bar_positions_abl = bar_positions_acc + bar_width + bar_gap # Verschiebe die Positionen für Ablation

# Erster Plot (Accumulation)
plt.figure(figsize=(10, 6))

plt.bar(bar_positions_acc, denoised_successfull_acc, bar_width, label='denoised successfully', color=col_acc, alpha=1)
plt.bar(bar_positions_acc, not_visible_acc, bar_width, bottom=np.array(denoised_successfull_acc),
        label='not visible on raw nor denoised', color=col_acc, alpha=0.5)
plt.bar(bar_positions_acc, raw_visible_acc, bar_width,
        bottom=np.array(denoised_successfull_acc) + np.array(not_visible_acc), label='visible on raw data', color=col_acc,
        alpha=0.3)

# Zweiter Plot (Ablation)
plt.bar(bar_positions_abl, denoised_successfull_abl, bar_width, label='denoised successfully', color=col_abl, alpha=1)
plt.bar(bar_positions_abl, not_visible_abl, bar_width, bottom=np.array(denoised_successfull_abl),
        label='not visible on raw nor denoised', color=col_abl, alpha=0.5)
plt.bar(bar_positions_abl, raw_visible_abl, bar_width,
        bottom=np.array(denoised_successfull_abl) + np.array(not_visible_abl), label='visible on raw data', color=col_abl,
        alpha=0.3)

plt.ylabel('# Events', fontsize=fontsize + 1)
plt.yticks(fontsize=fontsize)
plt.xticks(bar_positions_acc + bar_width / 2 + bar_gap / 2, categories, fontsize=fontsize, rotation=15)
plt.legend(fontsize=fontsize, frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0.017, 1.1))

# Den gesamten Plot speichern
plt.tight_layout()
#plt.savefig("plots/visual_assesment/combined_plot.png")

# Den Plot anzeigen
plt.show()