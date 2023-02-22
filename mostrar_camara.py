import cv2 as cv

# Esta función permite utilizar la camara del pc
# si se selecciona el valor de 0 se usa la camara por defecto
# de la laptop.
captura_video = cv.VideoCapture(0)

if not captura_video.isOpened():
    print('No se encontró ninguna cámara')
    exit()

# Mostrar continuamente la cámara
while True:
    # Comenzar la captura de video
    tipo_camara, camara = captura_video.read()
    grises = cv.cvtColor(camara, cv.COLOR_BGR2GRAY)
    #cv.imshow("En Vivo", camara)
    
    # Mostrar la camara en escala de grises
    cv.imshow("En Vivo", grises)

    # Esto es útil para salir de la captura del video
    if cv.waitKey(1) == ord("q"):
        break

captura_video.release()
cv.destroyAllWindows()