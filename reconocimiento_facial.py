# El objetivo es el reconocimiento de rostros, evitando los objetos que generan ruidos
# Obtieniendo los puntos claves para el reconociemitno
# Entrenando mucho mejor el modelo y así obtener buenas resuestas

# OpenCV ya tiene una repositorio de todos los objetos que son ruidos
# https://github.com/opencv/opencv

# El que mas interesa es haarcascade_frontalface_default.xml

import cv2 as cv
import os

#A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, 
# displaying Matplotlib images, sorting contours, detecting edges, and much more easier with OpenCV.
import imutils

modelo = 'photos'
ruta_completa = './data/'+modelo

if not os.path.exists(ruta_completa):
    os.makedirs(name=ruta_completa)
    
# Cargar el ruido
ruidos = cv.CascadeClassifier('entrenamiento_opencv/data/haarcascades/haarcascade_frontalface_default.xml')

# Caputar imagenes desde la camara
#camara = cv.VideoCapture(0)

# Capturar imagenes desde un video
camara = cv.VideoCapture('assets/vids/ElonMusk.mp4')

id = 0
while True:
    respuesta , captura = camara.read()

    if respuesta == False: break

    captura = imutils.resize(captura,width=640)

    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura = captura.copy()

    # scaleFactor: 1.X donde X representa el porcentaje
    #
    cara = ruidos.detectMultiScale(image=grises, scaleFactor=1.4, minNeighbors= 3)

    # recorrido de todo el video
    for (x,y,e1,e2) in cara:
        cv.rectangle(img=captura, pt1=(x,y), pt2=(x+e1,y+e2), color=(255,0,0), thickness=2)
        rostro_capturado = idcaptura[y:y+e2,x:x+e1]
        rostro_capturado = cv.resize(src=rostro_capturado, dsize=[160,160], interpolation=cv.INTER_CUBIC)
        cv.imwrite(filename=ruta_completa+f'/imagen_{id}.jpg',img=rostro_capturado)

        id += 1

    cv.imshow('Resultados Rostro',mat=captura)

    if id == 100:
        break

camara.release()
cv.destroyAllWindows()