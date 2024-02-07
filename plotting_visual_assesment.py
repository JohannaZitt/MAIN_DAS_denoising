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

Combined Plot: 6 Legend


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
col_acc1 = "#008B93"
col_acc2 = "#7FC4C8"
col_acc3 = "#CCE7E9"
col_abl1 = "#DF575F"
col_abl2 = "#EEAAAE"
col_abl3 = "#F5CCCE"
zorder = 3
fontsize=12
bar_width = 0.25
bar_gap = 0.05
bar_positions_abl = np.arange(len(categories))
bar_positions_acc = bar_positions_abl + bar_width + bar_gap # Verschiebe die Positionen für Ablation

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={
                           'width_ratios': [1],
                           'height_ratios': [1, 6]})
fig.subplots_adjust(hspace=0.05)

''' Erster Oberer Subplot'''
# Accumulation Data:
ax1.bar(bar_positions_acc, denoised_successfull_acc, bar_width, label='denoised successfully', color=col_acc1, zorder=zorder)
ax1.bar(bar_positions_acc, not_visible_acc, bar_width, bottom=np.array(denoised_successfull_acc),
        label='not visible on raw nor denoised', color=col_acc2, zorder=zorder)
ax1.bar(bar_positions_acc, raw_visible_acc, bar_width,
        bottom=np.array(denoised_successfull_acc) + np.array(not_visible_acc), label='visible on raw data', color=col_acc3, zorder=zorder)

# Ablation Data:
ax1.bar(bar_positions_abl, denoised_successfull_abl, bar_width, label='denoised successfully', color=col_abl1, zorder=zorder)
ax1.bar(bar_positions_abl, not_visible_abl, bar_width, bottom=np.array(denoised_successfull_abl),
        label='not visible on raw nor denoised', color=col_abl2, zorder=zorder)
ax1.bar(bar_positions_abl, raw_visible_abl, bar_width,
        bottom=np.array(denoised_successfull_abl) + np.array(not_visible_abl), label='visible on raw data', color=col_abl3, zorder=zorder)

''' Zweiter Unterer Subplot'''
# Accumulation Data:
ax2.bar(bar_positions_acc, denoised_successfull_acc, bar_width, label='denoised successfully', color=col_acc1, zorder=zorder)
ax2.bar(bar_positions_acc, not_visible_acc, bar_width, bottom=np.array(denoised_successfull_acc),
        label='not visible on raw nor denoised', color=col_acc2, zorder=zorder)
ax2.bar(bar_positions_acc, raw_visible_acc, bar_width,
        bottom=np.array(denoised_successfull_acc) + np.array(not_visible_acc), label='visible on raw data', color=col_acc3, zorder=zorder)

# Ablation Data
ax2.bar(bar_positions_abl, denoised_successfull_abl, bar_width, label='denoised successfully', color=col_abl1, zorder=zorder)
ax2.bar(bar_positions_abl, not_visible_abl, bar_width, bottom=np.array(denoised_successfull_abl),
        label='not visible on raw nor denoised', color=col_abl2, zorder=zorder)
ax2.bar(bar_positions_abl, raw_visible_abl, bar_width,
        bottom=np.array(denoised_successfull_abl) + np.array(not_visible_abl), label='visible on raw data', color=col_abl3, zorder=zorder)

''' Ticks and Spines:'''
ax1.set_ylim(113, 121)
ax1.set_yticks([115, 120], [115, 120], fontsize=fontsize)
ax1.spines.bottom.set_visible(False)
ax1.xaxis.set_visible(False)
ax1.grid(axis="y", zorder=0)

ax2.set_ylim(0, 37)
ax2.xaxis.tick_bottom()
ax2.spines.top.set_visible(False)
ax2.set_ylabel('# Events', y=0.51, ha = 'left',  fontsize=fontsize)
ax2.grid(axis="y", zorder=0)

plt.yticks(fontsize=fontsize)
plt.xticks(bar_positions_abl + bar_width / 2 + bar_gap / 2, categories, fontsize=fontsize, rotation=15)


''' Generate Separation Lines: '''
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


# Custom Legend:
custom_lines = [mpatches.Patch(facecolor=col_abl1, alpha=1, edgecolor='black'),
                mpatches.Patch(facecolor=col_acc1, alpha=1, edgecolor='black'),
                mpatches.Patch(facecolor="black", alpha=1, edgecolor='black'),
                mpatches.Patch(facecolor="black", alpha=0.3, edgecolor='black'),
                mpatches.Patch(facecolor="black", alpha=0.1, edgecolor='black')
                ]

legend1 = ax1.legend(custom_lines[0:2], ["Ablation                     ", "Accumulation"],
                    fontsize=fontsize, frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0.028, 1.5), handlelength=5)
legend2 = ax1.legend(custom_lines[2:5], ["Denoising Sucssesfull", "Denoising Unsucssesfull", "Visible on Raw", "Visible on Raw"],
                    fontsize=fontsize, frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0.028, 1.8))

ax1.add_artist(legend1)
ax1.add_artist(legend2)

plt.tight_layout()

# Den Plot anzeigen oder speichern
plt.show()
#plt.savefig("plots/visual_assesment/combined_plot:6legend.png", bbox_inches="tight", pad_inches = 0.1)



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

