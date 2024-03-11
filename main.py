import cv2
import mediapipe as mp
import numpy as np
import time
from keras.models import load_model

# Carregar o modelo treinado com imagens de 64x64 processadas por Canny
model = load_model('sign_language_model_processed_64x64.h5')

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1)
classes = ["A",  "B",  "C",  "D",  "E",  "F",  "G",  "I",  "L",  "M",  "N",  "O",  "P",  "Q",  "R",  "S",  "T",  "U",  "V",  "W", "Y"]

data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)  # Atualizado para 64x64

word = ""
gesture_start_time = None
gesture_duration = 5  # Duração do contador para captura de gestos

while True:
    success, img = cap.read()
    if not success:
        print("Falha ao capturar imagem.")
        break
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Aplicar Canny para destacar as bordas (em vez de aplicar depois do recorte)
            imgCanny = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
            imgCanny = cv2.cvtColor(imgCanny, cv2.COLOR_GRAY2BGR)  # Converte de volta para BGR
            
            # Utiliza a mesma lógica de detecção e recorte de mãos
            # Mas aplicamos o algoritmo de Canny antes do recorte
            h, w, _ = img.shape
            x_min, x_max, y_min, y_max = w, 0, h, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, x_max, y_min, y_max = min(x, x_min), max(x, x_max), min(y, y_min), max(y, y_max)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Recorta a imagem Canny em vez da original
            hand_img = imgCanny[y_min:y_max, x_min:x_max]
            
            # Redimensiona a imagem para o tamanho esperado pelo modelo (64x64)
            hand_img_resized = cv2.resize(hand_img, (64, 64))
            
            # Prepara a imagem para a predição
            img_array = np.asarray(hand_img_resized)
            normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1  # Normaliza a imagem
            data[0] = normalized_image_array
            
            # Realiza a predição
            prediction = model.predict(data)
            class_id = np.argmax(prediction)
            
            # Atualiza a palavra baseada no tempo decorrido
            if gesture_start_time is None or time.time() - gesture_start_time >= gesture_duration:
                word += classes[class_id]
                gesture_start_time = time.time()  # Reinicia o contador
    
    # Mostra o texto acumulado na tela
    cv2.putText(img, word, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Imagem Processada', img)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Sair
        break
    elif key == 81:  # Seta esquerda para apagar
        word = word[:-1]
    elif key == 83:  # Seta direita para enviar
        print("Palavra enviada:", word)
        word = ""  # Reinicia a palavra

cap.release()
cv2.destroyAllWindows()
