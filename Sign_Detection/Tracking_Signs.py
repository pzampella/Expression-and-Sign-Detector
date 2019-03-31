import cv2
import numpy as np
import os
import shutil
from numpy import *
from scipy.stats import mode
import dlib
from skimage import exposure
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
import platform
sc = StandardScaler()

##########################################FUNCIONES#####################################################################
def detectar_gesto(Imagen):
    bajo = np.array([0, 48, 80], dtype="uint8")                                                                         # Definir rango minimo de tez de piel
    alto = np.array([20, 255, 255], dtype="uint8")                                                                      # Definir rango maximo de tez de piel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))                                                     # Generar matriz de Kernel
    ImgHSV = cv2.cvtColor(Imagen, cv2.COLOR_BGR2HSV)                                                                # Pasar imagen a HSV
    Mascara = cv2.inRange(ImgHSV, bajo, alto)                                                                       # Generar mascara
    Mascara = cv2.erode(Mascara, kernel, iterations=1)                                                              # Erosionar mascara
    Mascara = cv2.dilate(Mascara, kernel, iterations=2)                                                             # Dilatar mascara
    Binario = cv2.threshold(Mascara, 60, 255, cv2.THRESH_BINARY)[1]                                                 # Pasar mascara a binario
    contours, hierarchy = cv2.findContours(Binario, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                      # Extraer contorno
    max_area = 0
    idx = 0
    for i in range(len(contours)):                                                                                  # Seleccionar el contorno mas grande
        contorno = contours[i]
        area = cv2.contourArea(contorno)
        if (area > max_area):
            max_area = area
            idx = i
        # fin if
    # fin for
    dedos = ""
    if len(contours) > 0:
        contorno = contours[idx]
        hull = cv2.convexHull(contorno, returnPoints=False)
        defects = cv2.convexityDefects(contorno, hull)
        if cv2.contourArea(cv2.convexHull(contorno, hull)) < 1.1*cv2.contourArea(contorno):
            dedos = 0
        else:
            dedos = 1
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contorno[s][0])
                end = tuple(contorno[e][0])
                far = tuple(contorno[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi/2:
                    # cv2.line(Imagen, start, end, [0, 255, 0], 2)
                    # cv2.circle(Imagen, far, 5, [0, 0, 255], -1)
                    dedos += 1
                # fin if
            # fin for
        # fin if
    return dedos

########################################################################################################################
def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames
########################################################################################################################
def contraste(imagen):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    contraste = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return contraste
########################################################################################################################
def leer_letra(numero):
    switcher = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "K",
        10: "L",
        11: "M",
        12: "N",
        13: "O",
        14: "P",
        15: "Q",
        16: "R",
        17: "S",
        18: "T",
        19: "U",
        20: "V",
        21: "W",
        22: "X",
        23: "Y"
    }
    return switcher.get(numero, lambda: "Invalid number")
########################################################################################################################
def main():
        #Cargar archivos
    sistema = platform.system()
    if sistema == 'Linux':
        path_signs = "/home/pzampella/NimSet/CNN/Hands/Keras/CNN_87_1534855126/"
        json_file = open(path_signs + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        classifier_hands = model_from_json(loaded_model_json)
        classifier_hands.load_weights(path_signs + '/model.h5')
        clasificador_rostro = cv2.CascadeClassifier("/home/pzampella/opencv-3.4.1/data/lbpcascades/lbpcascade_frontalface_improved.xml")       # Cargar el clasificador de rostros
        #clasificador_mano = cv2.CascadeClassifier("/home/pzampella/opencv-3.4.1/data/haarcascades/haarcascade_hands.xml")    # Cargar el clasificador de_manos
    if sistema == 'Windows':
        path_signs = r'CNN'
        json_file = open(path_signs + r'\model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        classifier_hands = model_from_json(loaded_model_json)
        classifier_hands.load_weights(path_signs + '\model.h5')
        clasificador_rostro = cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml")       # Cargar el clasificador de rostros
        #clasificador_mano = cv2.CascadeClassifier("/home/pzampella/opencv-3.4.1/data/haarcascades/haarcascade_hands.xml")    # Cargar el clasificador de_manos
         
    # load weights into new model
    classifier_hands.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    print("Loaded model hands from disk")

    Path_imagenes = "/home/pzampella/Imagenes/"
    if os.path.exists(Path_imagenes):
        shutil.rmtree(Path_imagenes)
    if not os.path.exists(Path_imagenes):
        os.makedirs(Path_imagenes)
    min_area = 3000                                                                                                     # Minima area detectable (en pixeles)
    piel_min = 0.5                                                                                                      # Porcentaje de piel minimo detectable
    bajo = np.array([0, 48, 80], dtype="uint8")                                                                         # Definir rango minimo de tez de piel
    alto = np.array([20, 255, 255], dtype="uint8")                                                                      # Definir rango maximo de tez de piel
    trackers = cv2.MultiTracker_create()                                                                                       # Inicializar tracker

    bboxs = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]                                                                  # Declarar bounding box

    print("\n Set the maximum number of hands to be detected. The greater the number, the lower the performance.\n")
    max_hands = int(input("Maximum number of hands: "))
    type(max_hands)
    
    Webcam = cv2.VideoCapture(0)                                                                                        # Declarar variable de captura de video y almacenar un frame

    if Webcam.isOpened() == False:                                                                                      # Revisar si hay acceso a la webcam:
        print("Error: Camara no disponible\n\n")                                                                         # Si no, mostrar error
        os.system("pause")
        return                                                                                                          # Terminar programa y salir
    # fin if

    FrameLeido, ImgOriginal = Webcam.read()

    if not FrameLeido or ImgOriginal is None:                                                                           # Si el frame no fue leido exitosamente:
        print("Error: no se ha podido leer la imagen de la camara\n")                                                    # Mostrar error
        os.system("pause")
        return
    # fin if

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) # Generar matriz de Kernel

    ImgHSV = cv2.cvtColor(ImgOriginal, cv2.COLOR_BGR2HSV) # Pasar imagen a HSV
    Mascara = cv2.inRange(ImgHSV, bajo, alto) # Generar mascara
    #Mascara = cv2.erode(Mascara, kernel, iterations=2) # Erosionar mascara
    Mascara = cv2.dilate(Mascara, kernel, iterations=3) # Dilatar mascara
    Borroso = cv2.GaussianBlur(Mascara, (3, 3), 0) # Difuminar mascara
    Imagen = cv2.threshold(Borroso, 60, 255, cv2.THRESH_BINARY)[1] # Pasar mascara a binario
    contours, hierarchy = cv2.findContours(Imagen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Extraer contornos
    areas = [cv2.contourArea(c) for c in contours] 
    contador = 0
    for i in range(0, len(areas)): # Contar el numero de areas mayores que el minimo
        if areas[i] > min_area:
            contador += 1
        # fin if
    # fin for
    maximo = min(contador, max_hands) # Definir el numero de areas (maximo 3, manos y cara)

    for i in range(0, maximo): # Para cada area
        areas = [cv2.contourArea(c) for c in contours] # Extraer areas
        max_index = np.argmax(areas) #Hallar el area de mayor tamano
        idx = contours[max_index]
        x, y, w, h = cv2.boundingRect(idx) # Crear el bounding box que encierra la mayor area
        bboxs[i] = (x, y, w, h)
        contours.pop(max_index) # Eliminar la mayor area
    #fin for

    for i in range(0, maximo):
        new_tracker = cv2.TrackerMIL_create()
        _ = trackers.add(new_tracker, Imagen, bboxs[i]) # Inicializar trackers
    # fin for
    contador_viejo = maximo
    reset_tracker = 0
    cuenta = 0
    texto = ""
    cara_cuenta = 0
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while cv2.waitKey(1) != 27 and Webcam.isOpened(): # Ejecutar mientras la tecla ESC no sea presionada
        cuenta += 1
        FrameLeido, ImgOriginal = Webcam.read() # Leer el proximo frame
        if not FrameLeido or ImgOriginal is None: # Si el frame no fue leido exitosamente:
            print("Error: no se ha podido leer la imagen de la camara\n") # Mostrar error
            os.system("pause")
            break # Terminar programa y salir
        # fin if

        ImgHSV = cv2.cvtColor(ImgOriginal, cv2.COLOR_BGR2HSV) # Pasar imagen a HSV
        Mascara = cv2.inRange(ImgHSV, bajo, alto) # Generar mascara
        #Mascara = cv2.erode(Mascara, kernel, iterations=1) # Erosionar mascara
        Mascara = cv2.dilate(Mascara, kernel, iterations=3) # Dilatar mascara
        Borroso = cv2.GaussianBlur(Mascara, (3, 3), 0) # Difuminar mascara
        Imagen = cv2.threshold(Borroso, 60, 255, cv2.THRESH_BINARY)[1] # Pasar mascara a binario
        contours, _ = cv2.findContours(Imagen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Extraer contornos
        areas = [cv2.contourArea(c) for c in contours]
        contador = 0
        for i in range(0, len(areas)): # Contar el numero de areas mayores que el minimo
            if areas[i] > min_area:
                contador += 1
            # fin if
        # fin for
        ###############################################
        maximo = min(contador, max_hands) # Definir el numero de areas (maximo 3, manos y cara)
        if contador_viejo != maximo or reset_tracker == 1: # Si cambia el numero de objetos o se pierde el tracking de alguno, reconfigurar tracking
            trackers.clear()
            trackers = cv2.MultiTracker_create() # Reiniciar tracking
            for i in range(0, maximo): # Para cada area
                areas = [cv2.contourArea(c) for c in contours] # Extraer areas
                max_index = np.argmax(areas) # Hallar el area de mayor tamano
                idx = contours[max_index]
                x, y, w, h = cv2.boundingRect(idx) # Crear el bounding box que encierra la mayor area
                bbox = (x, y, w, h)
                contours.pop(max_index) # Eliminar la mayor area
                new_tracker = cv2.TrackerMIL_create()
                _ = trackers.add(new_tracker, Imagen, bbox) # Redefinir el tracker
            # fin for
            reset_tracker = 0
        # fin if
        tracking, bboxs = trackers.update(Imagen) # Actualizar tracker
        contours, _ = cv2.findContours(Imagen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        puntos = []
        puntos_medios = ""
        for bbox in bboxs:
            if tracking:
                # Tracking exitoso
                p = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
                puntos_medios = puntos_medios + str(p[0]) + "," + str(p[1])+" "
                for i in range(0, len(areas)):
                    if cv2.pointPolygonTest(contours[i], p, False) == 1.0:
                            _, _, bbox[2], bbox[3] = cv2.boundingRect(contours[i]) # Actualizar alto y ancho del bounding box
                     # fin if
                # fin for
                x_ = int(bbox[0])
                y_ = int(bbox[1])
                w_ = int(bbox[2])
                h_ = int(bbox[3])
                if h>w:
                    alpha = int((h_-w_)/2)
                    p1 = (x_-alpha, y_)
                    p2 = (x_+w_+alpha, y_+h_)
                else:
                    alpha = int((w_-h_)/2)
                    p1 = (x_, y_-alpha)
                    p2 = (x_+w_, y_+h_+alpha)
                tamano1 = len(ImgOriginal)
                tamano2 = len(ImgOriginal[0])
                if p1[0]<0:
                    p1=(0, p1[1])
                if p1[1]<0:
                    p1=(p1[0], 0)
                if p2[0]>tamano1-1:
                    p2=(tamano1-1, p2[1])
                if p2[1]>tamano2-1:
                    p2=(p2[0], tamano2-1)
                revisar = ImgOriginal[p1[1]:p2[1], p1[0]:p2[0]]
                if len(revisar[0]) != 0:
                    faces = len(clasificador_rostro.detectMultiScale(cv2.cvtColor(revisar, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5))
                else:
                    faces = -1
                if faces == 0:
                    puntos.append([p1[1], p2[1], p1[0], p2[0]])            # Calcular puntos del bounding box
                    cv2.rectangle(ImgOriginal, p1, p2, (200, 0, 0), 2, 1) # Dibujar el nuevo bounding box
                #end if
            else:
                # Tracking fallido
                cv2.putText(ImgOriginal, "Fallo durante rastreo", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)
                print("Fallo durante rastreo")
            # fin if
        # fin for
        #texto = texto + puntos_medios + "|"
        contador_viejo = maximo
        for i in range(0, len(puntos)):
            piel = 0
            if puntos[i][0] != 0 or puntos[i][1] != 0 or puntos[i][2] != 0 or puntos[i][3] != 0: # Si el cuadro uno tiene imagen
                Cuadro = Imagen[puntos[i][0]:puntos[i][1], puntos[i][2]:puntos[i][3]] # Recortar imagen
                unos = list(Cuadro[0]).count(1)
                total = len(Cuadro) * len(Cuadro[0])
                if total == 0:
                    piel = 0
                else:
                    piel = unos / total # Calcular el porcentaje de piel detectada
            # fin if
            if piel < piel_min:
                reset_tracker = 1
            # fin if
            if (puntos[i][1]-puntos[i][0]>5) and (puntos[i][3]-puntos[i][2]>5):
                Aux = ImgOriginal[puntos[i][0]:puntos[i][1], puntos[i][2]:puntos[i][3]]
                Mask = Imagen[puntos[i][0]:puntos[i][1], puntos[i][2]:puntos[i][3]]
                Mask2 = cv2.morphologyEx(Mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) )
                Aux = cv2.bitwise_and(Aux,Aux,mask = Mask2)
                Aux2 = cv2.cvtColor(Aux, cv2.COLOR_RGB2GRAY)
                Aux2[Aux2 == 0] = 128
                contrast = exposure.equalize_hist(Aux2) * 255
                Aux3 = sc.fit_transform(cv2.resize(contrast, (28, 28)))
                Aux4 = Aux3.reshape(1, 28, 28, 1)
                    # plt.imshow(Aux)
                    # plt.show()
                letra = np.argmax(classifier_hands.predict(Aux4))
                cv2.putText(ImgOriginal, leer_letra(letra), (puntos[i][2]+5, puntos[i][0]+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    # fin if
                #cv2.namedWindow("Uno", cv2.WINDOW_NORMAL)
                #cv2.imshow("Uno", Aux) # Dibujar el cuadro
                #cv2.imwrite(Path_imagenes + str(i) + "_" + str(0) + ".jpg", ImgOriginal[puntos[i][0]:puntos[i][1], puntos[i][2]:puntos[i][3]])
                #cv2.imwrite(Path_imagenes + str(i) + "_" + str(cuenta) + ".jpg", ImgOriginal[puntos[i][0]:puntos[i][1], puntos[i][2]:puntos[i][3]])  # Guardar imagen
            #fin if
            #cv2.namedWindow("Video mascara", cv2.WINDOW_NORMAL)
            #cv2.imshow("Video mascara", Aux2)
        # fin for
        cv2.namedWindow("Video original", cv2.WINDOW_NORMAL)                                                            # Crear ventana para mostrar frame original
        cv2.imshow("Video original", ImgOriginal)                                                                       # Mostrar frame original
    # fin while
    archivo = open(Path_imagenes+"\Coordenadas.txt", "w")
    archivo.write(texto)
    archivo.close()
    return

########################################################################################################################
if __name__ == "__main__":
    main()
