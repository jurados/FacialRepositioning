# !/usr/bin/env python
# coding: utf-8
# última modificación: Feb 22 / 2023

import cv2 as cv
import os

lista_data = os.listdir('./data')

entrenamiento_efc = cv.face.EigenFaceRecognizer_create()
entrenamiento_efc.read('Entrenamiento_EFR.xml')

ruidos = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

camara = cv.VideoCapture(0)

while True:
    _, captura = camara.read()
    grises = cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura = grises.copy()
    cara = ruidos.detectMultiScale(grises,1.3,5)

    for (x,y,e1,e2) in cara:
        rostro_capturado = idcaptura[y:y+e2,x:x+e1]
        rostro_capturado = cv.resize(src=rostro_capturado, dsize=[160,160], interpolation=cv.INTER_CUBIC)

        resultado = entrenamiento_efc.predict(rostro_capturado)
        if resultado[1] <8200:
            cv.putText(img=captura,text=f'{lista_data[resultado[0]]}',org=(x,y-5),fontFace=1,fontScale=1.3,color=(255,0,0),thickness=1)
            cv.rectangle(img=captura, pt1=(x,y), pt2=(x+e1,y+e2), color=(0,255,0), thickness=2)
        else:
            cv.putText(img=captura,text='No encontrado',org=(x,y-5),fontFace=1,fontScale=1.3,color=(255,0,0),thickness=1)
            cv.rectangle(img=captura, pt1=(x,y), pt2=(x+e1,y+e2), color=(0,255,0), thickness=2)
        
    cv.imshow("Resultados",captura)

    if cv.waitKey(1) == ord('q'): break

camara.release()
cv.destroyAllWindows()