import os
import shutil
import tkinter as tk
from tkinter import simpledialog, filedialog
from PIL import Image

def list_png_images(folder_path):
    png_images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            pass
        else:
            png_images.append(file_path)
    return png_images


# Dialog zum Auswählen des Bildordners
image_folder = tk.filedialog.askdirectory(title="Welchen Bildordner willst du kategorisieren?")
image_paths = list_png_images(image_folder)

# Dialog zur Eingabe der Kategorien
categories = ["1_raw:visible_denoised:better_visible",
              "2_raw:not_visible_denoised:visible",
              "3_raw:not_visible:denoised:not_visible",
              "4_raw:visible_denoised:not_visible",
              "5_miscellaneous"]
#while True:
#    category = simpledialog.askstring("Kategorie hinzufügen", "Geben Sie eine Kategorie ein (Leer lassen, um fortzufahren):")
#    if category:
#        categories.append(category)
#    else:
#        break

# Erstellen der Ordner Struktur:
for category in categories:
    os.makedirs(os.path.join(image_folder, category), exist_ok=True)


for image_path in image_paths:

    # open image
    image = Image.open(image_path)
    image.show()
    image_name = image_path.split('/')[-1]

    # ask for category to put the imate
    while True:
        cat = simpledialog.askstring(title="Kategorie abfrage",
                                    prompt="In welche der Kategorien soll das Bild geschoben werden? \n"
                                           "1_raw:visible_denoised:better_visible \n"
                                           "2_raw:not_visible_denoised:visible \n"
                                           "3_raw:not_visible:denoised:not_visible \n"
                                           "4_raw:visible_denoised:not_visible \n"
                                           "5_miscellaneous")

        if not cat in ['1', '2', '3', '4', '5']:
            print('Diese Kategorie existiert nicht. ')
            print('Bitte nutze eine dieser Kategorien: 1, 2, 3, 4, 5')

        else:
            break

    image.close()

    # save picture in categories
    if cat == '1':
        shutil.move(image_path, os.path.join(image_folder, categories[0], image_name))
    elif cat == '2':
        shutil.move(image_path, os.path.join(image_folder, categories[1], image_name))
    elif cat == '3':
        shutil.move(image_path, os.path.join(image_folder, categories[2], image_name))
    elif cat == '4':
        shutil.move(image_path, os.path.join(image_folder, categories[3], image_name))
    elif cat == '5':
        shutil.move(image_path, os.path.join(image_folder, categories[4], image_name))






