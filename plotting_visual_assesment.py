import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


'''

What TODO, to make it even preatier:
1. single and combined plots with skip in y-axis
2. mpatches mit zwei Farben in Legend bei combined plots

here you can find sample code, where you can skip a part of the y-axis: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html

ABLATION:
events worth looking at:
ablation:   0, 142, 175 (weil gut denoiset), 
            128 (unterschiedlich gut denoised)
            36, 53, 56, 61, 95, 99 (unterschiedliche Kategorien bei unterschiedlichen modellen)


events, die vielleicht auch in 2 sein könnten:
model 1 oder 2: 1, 23, 29, 36, 47, 48, 62, 74, 79, 83, 104, 161, 170, 171
model 3: 18, 23, 62, 74, 83, 173
model 4: 20, 48, 56, 61, 152
model 7: 101
model 8: 13, 15, 27, 56, 104, 129, 152
model 9: 19, 65, 83, 89, 104, 147 161





ACCUMULATION:
events worth looking at:
accumulation:   90, 106 (weil gut denoised)
                41, 61, 67, 76, 116  (andere Kategorien)


events die vielleicht auch kategorie 2 sein könnten:
model 1: - 
model 2: 14, 61
model 3: 36, 37, 72, 73
model 4: 58, 73
model 7: 41, 56
model 8: 58, 67, 72, 88, 116
model 9: 36, 58, 112, 116

'''



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
#plt.savefig("plots/visual_assesment/accumulation.png")
plt.show()

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

Combined Plot: 6 Legend




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


# Custom Legend:
custom_lines = [mpatches.Patch(facecolor=col_abl, alpha=1, edgecolor='black'),
                mpatches.Patch(facecolor=col_acc, alpha=1, edgecolor='black'),
                mpatches.Patch(facecolor=col_abl, alpha=0.5, edgecolor='black'),
                mpatches.Patch(facecolor=col_acc, alpha=0.5, edgecolor='black'),
                mpatches.Patch(facecolor=col_abl, alpha=0.3, edgecolor='black'),
                mpatches.Patch(facecolor=col_acc, alpha=0.3, edgecolor='black')
                ]


plt.legend(custom_lines, ["Ablation: Denoising Successfull", "Accumulation: Denoising Successfull", "Denoising Unsucssesfull", "Denoising Unsucssesfull", "Visible on Raw", "Visible on Raw"], fontsize=fontsize-1, frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0.028, 1.15))

# Den gesamten Plot speichern
plt.tight_layout()
plt.savefig("plots/visual_assesment/combined_plot:6legend.png", bbox_inches="tight", pad_inches = 0.1)

# Den Plot anzeigen
# plt.show()
'''

'''

Combined Plot: 5 Legend




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
bar_positions_abl = np.arange(len(categories))
bar_positions_acc = bar_positions_abl + bar_width + bar_gap # Verschiebe die Positionen für Ablation

# Erster Plot (Accumulation)
fig, ax = plt.subplots(figsize=(10, 6))
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
plt.xticks(bar_positions_abl + bar_width / 2 + bar_gap / 2, categories, fontsize=fontsize, rotation=15)


# Custom Legend:
custom_lines = [mpatches.Patch(facecolor=col_abl, alpha=1, edgecolor='black'),
                mpatches.Patch(facecolor=col_acc, alpha=1, edgecolor='black'),
                mpatches.Patch(facecolor="black", alpha=1, edgecolor='black'),
                mpatches.Patch(facecolor="black", alpha=0.3, edgecolor='black'),
                mpatches.Patch(facecolor="black", alpha=0.1, edgecolor='black')
                ]

legend1 = ax.legend(custom_lines[0:2], ["Ablation                     ", "Accumulation"],
                    fontsize=fontsize, frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0.028, 1.15))
legend2 = ax.legend(custom_lines[2:5], ["Denoising Sucssesfull", "Denoising Unsucssesfull", "Visible on Raw", "Visible on Raw"],
                    fontsize=fontsize, frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0.028, 1.1))

ax.add_artist(legend1)
ax.add_artist(legend2)

# Den gesamten Plot speichern
plt.tight_layout()
plt.savefig("plots/visual_assesment/combined_plot:5legend.png", bbox_inches="tight", pad_inches = 0.8)

# Den Plot anzeigen
# plt.show()

'''

