import numpy as np
import os

# Die Zeilen sind: Wall time, Step, Value
# Wall time = elapsed real time != CPU or GPU time
# Wall time is the actual time taken from the start of a computer program to the end
# CPU time measures only the time during which the processor is actively working on a certain task or process

# Value = epoch loss

# Beim Mitagessen ist rausgekommen, dass:
#    1. unübliche evaluierung
#    2. NICHT das Minimum nehmen, könnte zufälliger ausreißer sein, der mit der sonstigen Performance nichts zu tun hat
#    3. Wenn man loss mit anderen metriken in Verbindung bringen will, sollte man gleichen Zeitpunkt nehmen, wie die
#       anderen Metriken genutzt werden -> also am Ende des Trainings. Beispielsweise letzter validation loss
#    4. Sollte validation loss bevorzugen, gegenüber training loss -> da Netzwerk hier auf noch nie gesehenen Daten getestet wird.
#    5. mittleren loss ab epoche x ist auch sinnvoll zu betrachten
# FAZIT: es werden zwei Werte erhoben
#    1. letzter validation epoch loss
#    2. mittel über die letzen n Werte
#


experiments = os.listdir('../experiments')
#experiments = ['01_ablation_horizontal', '02_ablation_vertical', '03_accumulation_vertical', '04_accumulation_horizontal']


output = [['model_name', 'train_last', 'validation_last', 'train_mean', 'validation_mean', 'train_min', 'validation_min']]


with open('epoch_loss.txt', 'a') as file:

    file.write(str(output)+'\n')

    for experiment in experiments:
        file_path = os.path.join('../experiments', experiment, 'logs/')
        validation_file_name = experiment[:2] + '_epoch_loss_validation.csv'
        train_file_name = experiment[:2] + '_epoch_loss_train.csv'
        validation_loss = np.loadtxt(file_path + validation_file_name, delimiter=',', skiprows=1)
        train_loss = np.loadtxt(file_path + train_file_name, delimiter=',', skiprows=1)

        validation_min = round(np.min(validation_loss[:,2]),4)
        validation_mean = round(np.mean(validation_loss[500:,2]),4)
        validation_last = round(validation_loss[-1,2],4)

        train_min = round(np.min(train_loss[:, 2]),4)
        train_mean = round(np.mean(train_loss[500:, 2]),4)
        train_last = round(train_loss[-1, 2],4)

        calculated_values = [experiment, train_last, validation_last, train_mean, validation_mean, train_min, validation_min]

        file.write(str(calculated_values)+'\n')









