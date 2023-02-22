# Objetivo: Contar la cantidad de monedas que se encuentra en la iamgen

import cv2 as cv
import numpy as np

valor_gauss = 3
valor_kernel = 3

# Cargar imagen
#original = cv.imread('img/monedas.jpg')
original = cv.imread('img/monedas_colombia.jpg')

# Convertir a escala de grises
gris = cv.cvtColor(original, cv.COLOR_BGR2GRAY) 

# Se usara el desenfoque gaussiano, disminuyendo la cantidad
# de píxeles por imagen con la idea de "suavizarla". Usando una matriz auxiliar
# de un tamaño de fijo para fijar un solo valor.
gauss = cv.GaussianBlur(src=gris,ksize=(valor_gauss,valor_gauss), sigmaX=0)

# Eliminar los ruidos remanentes en una imagen a través de la función 
# "Canny" buscando los bordes.
canny = cv.Canny(gauss,60,100)

#kernel = np.ones(shape=(valor_kernel,valor_kernel),np.uint8)
kernel = np.ones(shape=(valor_kernel,valor_kernel))
# La idea es usar enteros teniendo en cuenta incluso 8 decimales

# Cambiar la forma de la imagen
# Teneindo en cuenta las caraceteristicas de las monedas, la mayor parte
# del ruido se encuentre dentro de ella, por esta razón usamos la morfología
# "Close"
cierre = cv.morphologyEx(canny,cv.MORPH_CLOSE,kernel)

contornos, jerarquia = cv.findContours(cierre.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
print("Monedas encontradas {}".format(len(contornos)))
cv.drawContours(original, contornos, -1, (0,0,255),2)

# Mostrar resultados
cv.imshow('Grises',gris)
cv.imshow('Gauss',gauss)
cv.imshow('Canny',canny)
cv.imshow('Resultado',original)

cv.waitKey(0)