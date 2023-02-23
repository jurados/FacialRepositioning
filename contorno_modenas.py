# !/usr/bin/env python
# coding: utf-8
# última modificación: Feb 22 / 2023

# Objetivo: Contar la cantidad de monedas que se encuentra en una imagen.

import cv2 as cv
import numpy as np

valor_gauss  = 3
valor_kernel = 3

# Cargar imagen
#original = cv.imread('img/monedas.jpg')
original = cv.imread(filename='./assets/imgs/monedas_colombia.jpg')

# Convertir a escala de grises
gris = cv.cvtColor(src=original, code=cv.COLOR_BGR2GRAY) 

# Se usara el desenfoque gaussiano, disminuyendo la cantidad
# de píxeles por imagen con la idea de "suavizarla". Usando una matriz auxiliar
# de un tamaño de fijo para fijar un solo valor.
gauss = cv.GaussianBlur(src=gris,ksize=(valor_gauss,valor_gauss), sigmaX=0)

# Eliminar los ruidos remanentes en una imagen a través de la función 
# "Canny" buscando los bordes.
canny = cv.Canny(image=gauss, threshold1=60, threshold2=100)

kernel = np.ones(shape=(valor_kernel,valor_kernel))
# La idea es usar enteros teniendo en cuenta incluso 8 decimales

# Cambiar la forma de la imagen
# Teniendo en cuenta las caraceteristicas de las monedas, la mayor parte
# del ruido se encuentre dentro de ella, por esta razón usamos la morfología
# "Close"
cierre = cv.morphologyEx(src=canny, op=cv.MORPH_CLOSE, kernel=kernel)

contornos, jerarquia = cv.findContours(image=cierre.copy(), mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
print("Monedas encontradas {}".format(len(contornos)))
cv.drawContours(image=original, contours=contornos, contoursIdx=-1, color=(0,0,255),thickness=2)

# Mostrar resultados
cv.imshow('Grises',gris)
cv.imshow('Gauss',gauss)
cv.imshow('Canny',canny)
cv.imshow('Resultado',original)

cv.waitKey(0)