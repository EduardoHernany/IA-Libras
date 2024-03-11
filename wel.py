import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import tensorflow as tf

#tensorFlow versão 2.9.1

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(max_num_hands=1)

classes = ['A', 'B', 'C' ,'D', 
'E' 'F', 'G','I', 'L''M', 'N',  'O', 'P', 'Q', 'R', 
 'S', 'T', 'U', 'V', 'W', 'Y']

model = load_model('model_epoch_48_98.6_final.h5')
data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)

def predictor(test_image):
       test_image = tf.keras.utils.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = model.predict(test_image)
       return result


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
                imgCrop = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
                cv2.imshow("crop", imgCrop)
                prediction = predictor(imgCrop)
                indexVal = np.argmax(prediction)
                print(prediction)
                cv2.putText(img,classes[indexVal],(x_min-50,y_min-65),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)

            except:
                continue

    cv2.imshow('Imagem',img)
    cv2.waitKey(1)


