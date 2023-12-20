import os
import shutil
import tkinter as tk
from tkinter import simpledialog, filedialog
from PIL import Image


# Dialog zum Auswählen des Bildordners
image_folder = tk.filedialog.askdirectory(title="Welchen Bildordner willst du kategorisieren?")
images = os.listdir(image_folder)

# Dialog zur Eingabe der Kategorien
categories = []
while True:
    category = simpledialog.askstring("Kategorie hinzufügen", "Geben Sie eine Kategorie ein (Leer lassen, um fortzufahren):")
    if category:
        categories.append(category)
    else:
        break

# Erstellen der Ordner Struktur:
for category in categories:
    os.makedirs(os.path.join(image_folder, category), exist_ok=True)


for image in images:

    # open image
    image_path = os.path.join(image_folder, image)
    img = Image.open(image_path)
    img.show()

    # ask for category to put the imate
    while True:
        cat = simpledialog.askstring(title="Kategorie abfrage",
                                    prompt="In welche der Kategorien soll das Bild geschoben werden?")
        if not cat in categories:
            print('Diese Kategorie existiert nicht. ')
            print('Bitte nutze eine dieser Kategorien: ', categories)

        else:
            break

    # save picture in categories
    shutil.move(image_path, os.path.join(image_folder, cat, image))



