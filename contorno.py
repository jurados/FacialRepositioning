#Umbralizacion: Diferencia el espacio donde se encuentra el objeto y el mismo objeto
#Segmentacion: Reconcoer la imagen como una figura geometrica a traves de su contonro

import cv2 as cv

imagen = cv.imread(filename='img/contorno.jpg')
grises = cv.cvtColor(src=imagen,code=cv.COLOR_BGR2GRAY)
# Devuelve dos valores
# 100 van avariando dependiendo de la calidad de la iagen
_, umbral = cv.threshold(grises,100,255,cv.THRESH_BINARY)

# Se debe invertir los colores para encontrar contornos
contorno, jerarquia = cv.findContours(umbral,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

# Si se quieren todos los colores se elecciona -1.
#cv.drawContours(imagen,contorno,1,(255,0,0),3)
cv.drawContours(imagen,contorno,-1,(255,0,0),3)

cv.imshow('Imagen original',imagen)
cv.imshow('Imagen grises', grises) 
cv.imshow('Imagen Umbral', umbral)

# Para una iamgen estatica es suficienite
# el valor por defecto es 1, pero eso es para mantener activo videos camara web
cv.waitKey(0)

# Destruir todas las ventanas abeirtas
cv.destroyAllWindows()
#cv2.cvDestroyAllWindows()

# Encontrar los contornos, Debe estar umbralizada la imagen es decir, ya debe de tener una sepración en blanco inegro
# la iamgen del contorno
# Tiene dos metodos Aprox none se pone el contorno y aprox simple solo selecciona pocos puntos y 