import os
import re



eight_path = os.path.join("experiments", '08_combined480', "plots", "ablation", "0706_RA88")
nine_path = os.path.join("experiments", '09_random480', "plots", "accumulation", "0706_AJP")

categories = ["1_raw:visible_denoised:better_visible",
              "2_raw:not_visible_denoised:visible",
              "3_raw:not_visible:denoised:not_visible",
              "5_miscellaneous"]

eight_ids = []
for category in categories:

    events = os.listdir(eight_path + "/" + category)
    for event in events:
        id = int(re.search(r'ID:(\d+)', event).group(1))
        eight_ids.append(id)

nine_ids = []
for category in categories:

    events_nine = os.listdir(nine_path + "/" + category)
    for event in events_nine:
        id = int(re.search(r'ID:(\d+)', event).group(1))
        nine_ids.append(id)


eight_ids.sort()
nine_ids.sort()
print(eight_ids)
print(nine_ids)
print(len(eight_ids))
print(len(nine_ids))
for i in eight_ids:
    if not i in nine_ids:
        print('Hier: ', i)




