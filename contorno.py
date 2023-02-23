# !/usr/bin/env python
# coding: utf-8
# última modificación: Feb 22 / 2023

# Umbralizacion: Diferencia el espacio donde se encuentra el objeto y el mismo objeto
# Segmentacion: Reconocer la imagen como una figura geometrica a traves de su contonro

import cv2 as cv

imagen = cv.imread(filename='assets/img/contorno.jpg')
grises = cv.cvtColor(src=imagen,code=cv.COLOR_BGR2GRAY)
# Devuelve dos valores
# 100 van avariando dependiendo de la calidad de la iagen
# THRESH_BINARY separa las imagenes exactamente entre los colores blanco y negro
_, umbral = cv.threshold(src=grises, thresh=100, maxval=255, type=cv.THRESH_BINARY)

# Se debe invertir los colores para encontrar contornos
# El objeto debe ser blanco y el fondo debe ser negra
# Se usa el metodo más simple para no consumir toda la memoria
contorno, jerarquia = cv.findContours(image=umbral, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)

# Si se quieren todos los contornos se elecciona -1. En caso de que se quiera un número finito se selecciona 
# contourIdx = 1, 2, ..., n
# cv.drawContours(imagen,contorno,1,(255,0,0),3)
cv.drawContours(image=imagen, contours=contorno, contourIdx=-1, color=(255,0,0), thickness=3)

cv.imshow('Imagen original',imagen)
cv.imshow('Imagen grises', grises) 
cv.imshow('Imagen Umbral', umbral)

# Para una iamgen estatica es suficienite
# el valor por defecto es 1, pero eso es para mantener activo videos camara web
cv.waitKey(0)

# Destruir todas las ventanas abeirtas
cv.destroyAllWindows()

# Encontrar los contornos, Debe estar umbralizada la imagen es decir, ya debe de tener una sepración en blanco inegro
# la iamgen del contorno
# Tiene dos metodos Aprox none se pone el contorno y aprox simple solo selecciona pocos puntos y 