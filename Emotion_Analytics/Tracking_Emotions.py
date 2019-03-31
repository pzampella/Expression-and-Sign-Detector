import cv2
import numpy as np
import os
import shutil
import math
from numpy import *
from scipy.stats import mode
import dlib
from skimage import exposure
from PIL import Image
from skimage import io
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
def leer_emocion(numero):
    switcher = {
        0: "Angry",
        1: "Neutral",
        2: "Disgust",
        3: "Fear",
        4: "Happy",
        5: "Sad",
        6: "Surprise"
    }
    return switcher.get(numero, lambda: "Invalid number")
########################################################################################################################
def main():
    #Cargar archivos
    sistema = platform.system()
    if sistema == 'Linux':
        model_path1 = r'/home/pzampella/NimSet/CNN/Face/Keras/CNN_79_1534936535'
        json_file = open(model_path1 + r'/model.json', 'r')
        #Fin cargar archivos
        loaded_model_json = json_file.read()
        json_file.close()
        classifier_face = model_from_json(loaded_model_json)
        classifier_face.load_weights(model_path1 + r'/model.h5')
        classifier_face.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        clasificador_rostro = cv2.CascadeClassifier(r"/home/pzampella/opencv-3.4.1/data/lbpcascades/lbpcascade_frontalface_improved.xml")
    if sistema == 'Windows':
        #model_path1 = 'CNN'
        json_file = open(r'CNN\model.json', 'r')
        #Fin cargar archivos
        loaded_model_json = json_file.read()
        json_file.close()
        classifier_face = model_from_json(loaded_model_json)
        classifier_face.load_weights(r'CNN\model.h5')
        classifier_face.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        clasificador_rostro = cv2.CascadeClassifier(r"lbpcascade_frontalface_improved.xml")       # Cargar el clasificador de rostros
    # load weights into new model
    
    print("Loaded model face from disk")

    min_area = 3000                                                                                                     # Minima area detectable (en pixeles)
    piel_min = 0.5                                                                                                      # Porcentaje de piel minimo detectable
    bajo = np.array([0, 48, 80], dtype="uint8")                                                                         # Definir rango minimo de tez de piel
    alto = np.array([20, 255, 255], dtype="uint8")                                                                      # Definir rango maximo de tez de piel

    bboxs = []                                                                  										# Declarar bounding box

    print("\n Set the maximum number of faces to be detected. The greater the number, the lower the performance.\n")
    max_faces = int(input("Maximum number of faces: "))
    type(max_faces)

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
    maximo = min(contador, max_faces)                                                                                           # Definir el numero de areas (maximo 3, manos y cara)

    for i in range(0, maximo):                                                                                          # Para cada area
        areas = [cv2.contourArea(c) for c in contours]                                                                  # Extraer areas
        max_index = np.argmax(areas)                                                                                    #Hallar el area de mayor tamano
        idx = contours[max_index]
        x, y, w, h = cv2.boundingRect(idx)                                                                              # Crear el bounding box que encierra la mayor area
        bboxs.append((x, y, w, h))
        contours.pop(max_index)                                                                                         # Eliminar la mayor area
    #fin for
    trackers = cv2.MultiTracker_create()   # Inicializar tracker
    for i in range(0, maximo):
        new_tracker = cv2.TrackerMIL_create()
        #_ = new_tracker.init(Imagen, bboxs[i])
        _ = trackers.add(new_tracker, Imagen, bboxs[i])                                                     # Inicializar trackers
    # fin for
    contador_viejo = maximo
    reset_tracker = 0                                                                                                         # Porcentaje de piel en cuadro detectado
    cuenta = 0
    cara_cuenta = 0
    emocion = [-1]
    while cv2.waitKey(1) != 27 and Webcam.isOpened():                                                                   # Ejecutar mientras la tecla ESC no sea presionada
        cuenta += 1
        FrameLeido, ImgOriginal = Webcam.read()                                                                         # Leer el proximo frame
        if not FrameLeido or ImgOriginal is None:                                                                       # Si el frame no fue leido exitosamente:
            print("Error: no se ha podido leer la imagen de la camara\n")                                               # Mostrar error
            os.system("pause")
            break                                                                                                       # Terminar programa y salir
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
        maximo = min(contador, max_faces) # Definir el numero de areas (maximo 3, manos y cara)

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
                #_ = new_tracker.init(Imagen, bbox)
                _ = trackers.add(new_tracker, Imagen, bbox) # Redefinir el tracker
            # fin for
            reset_tracker = 0
        # fin if
        tracking, bboxs = trackers.update(Imagen) # Actualizar tracker
        contours, _ = cv2.findContours(Imagen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        puntos = []
        counter = 0
        puntos_medios = ""
        for bbox in bboxs:
            if tracking:
                # Tracking exitoso
                p = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
                puntos_medios = puntos_medios + str(p[0]) + "," + str(p[1])+" "
                for i in range(0, len(areas)):
                    if cv2.pointPolygonTest(contours[i], p, False) == 1.0:
                            _, _, bbox[2], bbox[3] = cv2.boundingRect(contours[i])                                          # Actualizar alto y ancho del bounding box
                     # fin if
                # fin for
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                Aux = ImgOriginal[p1[1]:p2[1], p1[0]:p2[0]]
                faces = len(clasificador_rostro.detectMultiScale(cv2.cvtColor(Aux, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5))
                if faces==1:
                    puntos.append([p1[1], p2[1], p1[0], p2[0]])                                                          # Calcular puntos del bounding box
                    cv2.rectangle(ImgOriginal, p1, p2, (200, 0, 0), 2, 1)                                                   # Dibujar el nuevo bounding box
            else:
                # Tracking fallido
                cv2.putText(ImgOriginal, "Fallo durante rastreo", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)
                print("Fallo durante rastreo")
            # fin if
        # fin for
        contador_viejo = maximo
        for i in range(0,len(puntos)):
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
            Aux = ImgOriginal[puntos[i][0]:puntos[i][1], puntos[i][2]:puntos[i][3]]
            if cara_cuenta == 0:
                detected_face = detect_faces(Aux)
                if len(detected_face) > 0:
                    if True:
                        face = np.array(Image.fromarray(Aux).crop(detected_face[0]))
                        Aux2 = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                        contrast = exposure.equalize_hist(Aux2) * 255
                        Aux3 = sc.fit_transform(cv2.resize(contrast, (32, 32)))
                        archivo = Aux3.reshape(1, 32, 32, 1)
                    else:
                        Aux = ImgOriginal[puntos[i][0]:puntos[i][1], puntos[i][2]:puntos[i][3]]
                        Mask = Imagen[puntos[i][0]:puntos[i][1], puntos[i][2]:puntos[i][3]]
                        Mask2 = cv2.morphologyEx(Mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) )
                        Aux = cv2.bitwise_and(Aux,Aux,mask = Mask2)
                        Aux2 = cv2.cvtColor(Aux, cv2.COLOR_RGB2GRAY)
                        Aux2[Aux2 == 0] = 128
                        contrast = exposure.equalize_hist(Aux2) * 255
                        Aux3 = sc.fit_transform(cv2.resize(contrast, (32, 32)))
                        archivo = Aux3.reshape(1, 32, 32, 1)
                    emocionA = np.argmax(classifier_face.predict(archivo))
                    emocion = mode([emocionA, emocionA])
            cv2.putText(Aux, leer_emocion(emocion[0][0]), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            # fin if
            cv2.namedWindow("Video original", cv2.WINDOW_NORMAL) # Crear ventana para mostrar frame original
        cv2.imshow("Video original", ImgOriginal) # Mostrar frame original                                                                                    
        bboxs = []
        #if cara_cuenta==5 or len(puntos)==0:
        #    cara_cuenta = 0
        #else:
        #	cara_cuenta += 1
    # fin while
    return

########################################################################################################################
if __name__ == "__main__":
    main()
