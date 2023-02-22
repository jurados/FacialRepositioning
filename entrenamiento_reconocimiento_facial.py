import os
import numpy as np
import cv2 as cv


# Cargar archivos
lista_data = os.listdir('./data/')

ids, rostros_data = [], []
id = 0

for file in lista_data:
    ruta_completa = './data/' + file
    for archivo in os.listdir(ruta_completa):
        ids.append(id)
        # Otra forma de convertir a escala de grises es seleccionar el valor de 0
        rostros_data.append(cv.imread(filename=ruta_completa+'/'+archivo,flags=0))
    id += 1
# Entrenamiento
entrenamiento_modelo = cv.face.EigenFaceRecognizer_create()

print('Iniciando el entrenamiento ...')
entrenamiento_modelo.train(rostros_data,np.array(ids))

entrenamiento_modelo.write('Entrenamiento_EFR.xml')
print('Entrenamienot terminado ...')
