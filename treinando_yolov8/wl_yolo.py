import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

#tensorFlow versão 2.9.1

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(max_num_hands=1)

model = YOLO("runs/classify/train/weights/last.pt")

def predictor(test_image):
       return model(test_image)

def process_image(img):
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar o filtro de Canny
    edges = cv2.Canny(gray, 100, 200)
    # Converter as bordas detectadas para uma máscara de 3 canais
    mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Remover o fundo
    img_no_bg = cv2.bitwise_and(img, mask)
    blurimg = cv2.blur(img, (3, 3) )
    return img


while True:
    success, img = cap.read()
    frameRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape

    if handsPoints != None:
        for hand in handsPoints:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(img, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)

            try:
                imgCrop = img[y_min-60:y_max+60,x_min-60:x_max+60]
                imgCrop = cv2.resize(imgCrop, (64, 64))
                imgCrop = process_image(imgCrop)
                cv2.imshow("crop", imgCrop)
                prediction = predictor(imgCrop)
                cv2.putText(img,prediction[0].names[prediction[0].probs.top1],(x_min-50,y_min-65),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)
            except:
                continue

    cv2.imshow('Imagem',img)
    cv2.waitKey(1)
