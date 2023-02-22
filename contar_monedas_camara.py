import cv2 as cv
import numpy as np

def ordernar_puntos(puntos):
    """
    Objetivo: Definir los puntos guías para generar el contorno
    alrededor de la figura (moendas).
    -----
    Params:
        puntos: 
    ----

    """

    n_puntos = np.concateante(puntos[0],puntos[1],puntos[2],puntos[3]).tolist()
    # Ordenar los puntos horizontales y verticales
    y_order = sorted(n_puntos,key=lambda n_puntos:n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order,key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x1_order = sorted(x2_order,key=lambda x2_order: x2_order[0])

    return [x1_order[0],x2_order[0],x2_order[1]]

def alineamiento(imagen, ancho, alto):
    """
    Objetivo: Permite mantener alineada la imagen apesar de que se rote
    la cámara.
    """

    imagen_alineada = None
    grises = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
    tipo_umbral, umbral = cv.threshold(grises,150,255,cv.THRESH_BINARY)
    cv.imshow("Umbral",umbral)

    # Aplicar esto desde el punto 0.
    contorno, jerarquia = cv.findContours(umbral, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    contorno = sorted(contorno, key=cv.contourArea,reverse=True)
    for c in contorno:
        # La siguiente función permite encontrar un área más cercana a la forma real del objeto
        # estudiado.
        epsilon = 0.01*cv.arcLength(curve=c, closed=True)
        approx = cv.approxPolyDP(curve=c, epsilon=epsilon, closed=True)
        if len(approx) == 4:
            puntos = ordernar_puntos(approx)
            puntos_1 = np.float32(puntos)
            puntos_2 = np.float32([[0,0],[ancho,0],[0,alto],[ancho,alto]])

            # Parte fija de la observacion
            M = cv.getPerspectiveTransform(puntos_1,puntos_2)
            imagen_alineada = cv.warpPerspective(src=imagen, M=M, dsize=(ancho, alto))

    return imagen_alineada

captura_video = cv.VideoCapture(0)
while True:
    tipo_camara, camara = captura_video.read()
    if tipo_camara == False:
        break
    # relacion de aspecto = ancho/alto 
    # alto_pixesles o ancho_píxeles = ancho_pixles / ra o alt_pixeles * ra
    
    imagen_A6 = alineamiento(imagen=camara,ancho=480,alto=677)
    if imagen_A6 is not None:
        puntos = []
        imagen_gris = cv.cvtColor(imagen_A6,cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(imagen_gris,(5,5),1)
        _, umbral_2 = cv.threshold(blur,0,255,cv.THRESH_OTSU+cv.THRESH_BINARY_INV)
        cv.imshow("Umbral",umbral_2)
        contorno_2, jerarquia_2 = cv.findContours(umbral_2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]
        cv.drawContours(imagen_A6,contorno_2,-1,(255,0,0),2)
        
        # Depende de la cantidad de monedas
        suma_1 = 0.0
        suma_2 = 0.0

        for c in contorno_2:
            area = cv.contourArea(c)

            # Busca el centro de "masa" o "centroide" de la iamgen
            momentos = cv.moments(array=c)
            # "m00" momento estatico
            if momentos["m00"] == 0:
                momentos["m00"] = 1.0
            # Momentos de movimientos "m10" y "m01"
            x = int(momentos["m10"]/momentos["m00"])
            y = int(momentos["m01"]/momentos["m00"])

            # Diametro de monedas colombianas [mm]
            dmoneda_500  = 23.70 
            dmoneda_1000 = 26.70
            area_moneda = lambda d: np.pi*(d/2*10)**2 * 480 # Valor en píxeles
            if area < area_moneda(dmoneda_1000)+100 and area > area_moneda(dmoneda_1000) - 100:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(img=imagen_A6,text="COP 1000",org=(x,y),fontFace=font,fontScale=0.74,color="red")
                suma_1 += 0.2

            if area < area_moneda(dmoneda_500)+100 and area > area_moneda(dmoneda_500) - 100:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(img=imagen_A6,text="COP 500",org=(x,y),fontFace=font,fontScale=0.74,color="red")
                suma_1 += 0.2

        total = suma_1 + suma_2
        print("Sumatoria total es: ",round(total,2))
        cv.imshow("Imagen A6", imagen_A6)
        cv.imshow("Camara",camara)
    if cv.waitKey(1) == ord("s"):
        break

captura_video.realeased()
cv.destroyAllWindows()


