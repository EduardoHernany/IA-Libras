import cv2
import mediapipe as mp
import numpy as np
import time
from keras.models import load_model

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1)
classes = ['A', 'B', 'C', 'D', 'E']
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

word = ""
gesture_start_time = None
gesture_duration = 5  # Duração do contador para captura de gestos

while True:
    success, img = cap.read()
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape

    if handsPoints:
        for hand in handsPoints:
            # Determinar os limites da mão
            x_max, y_max = 0, 0
            x_min, y_min = w, h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max, y_max = max(x, x_max), max(y, y_max)
                x_min, y_min = min(x, x_min), min(y, y_min)

            cv2.rectangle(img, (x_min - 50, y_min - 50), (x_max + 50, y_max + 50), (0, 255, 0), 2)

            if gesture_start_time is None:
                gesture_start_time = time.time()

            time_elapsed = time.time() - gesture_start_time
            time_left = gesture_duration - time_elapsed
            if time_left < 0:
                time_left = 0

            cv2.putText(img, f"Tempo: {time_left:.2f}s", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            if time_elapsed >= gesture_duration:
                try:
                    imgCrop = img[y_min - 50:y_max + 50, x_min - 50:x_max + 50]
                    imgCrop = cv2.resize(imgCrop, (224, 224))
                    imgArray = np.asarray(imgCrop)
                    normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                    data[0] = normalized_image_array
                    prediction = model.predict(data)
                    indexVal = np.argmax(prediction)

                    word += classes[indexVal]
                    gesture_start_time = None  # Reinicia o contador

                except Exception as e:
                    print(e)

    else:
        gesture_start_time = None  # Reinicia o contador se não houver mãos detectadas

    cv2.putText(img, word, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Imagem', img)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Pressione 'q' para sair
        break
    elif key == 81:  # Seta esquerda para apagar
        word = word[:-1]
    elif key == 83:  # Seta direita para enviar
        print("Palavra enviada:", word)
        word = ""

cap.release()
cv2.destroyAllWindows()