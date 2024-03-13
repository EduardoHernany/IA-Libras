from distutils import config
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ReduceLROnPlateau

import datetime

# Caminho para o diretório com as imagens processadas
dataset_dir = "Letras"
dataset_validation_dir = "test"
FILE_NAME="cnn_model_libras"

def getDateStr():
        return str('{date:%Y%m%d_%H%M}').format(date=datetime.datetime.now())


# Parâmetros de pré-processamento
image_size = (64, 64)  # Tamanho das imagens para o treinamento
batch_size = 64  # Número de amostras por lote

# Preparando os dados
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalização das imagens,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.25 # 25% dos dados para validação
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Modo "categorical" para classificação multiclasse,
    shuffle=False,
)

validation_generator = train_datagen.flow_from_directory(
    dataset_validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
)

model_constructors = {
            "DenseNet201": tf.keras.applications.DenseNet201,
        }

base_model = model_constructors["DenseNet201"](weights='imagenet', include_top=False, input_shape=(64,64,3))
for layer in base_model.layers:
    layer.trainable=False


print(len(train_generator.class_indices))
def build(width, height, channels, classes):

        inputShape = (height, width, channels)

        model = Sequential()
        model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = inputShape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(filters = 32, kernel_size = (3,3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(filters = 64, kernel_size = (3,3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))

        model.add(Flatten())
        model.add(Dense(254, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation = 'softmax'))
        return model

# Compilando o modelo
#model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



model  = build(64, 64, 3, len(train_generator.class_indices))
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
model.compile(optimizer=SGD(0.01),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])

# Treinamento do modelo
classifier = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    shuffle = False,
    verbose=2,
    epochs=50,
    callbacks=[early]
)

# Salvar o modelo treinado
print("[INFO] Salvando modelo treinado ...")

file_date = getDateStr()
model.save('../models/'+FILE_NAME+file_date+'.h5')

EPOCHS = len(classifier.history["loss"])



print('[INFO] Summary: ')
model.summary()


print("\n[INFO] Avaliando a CNN...")
score = model.evaluate_generator(generator=validation_generator, steps=(train_generator.n // validation_generator.batch_size), verbose=1)
print('[INFO] Accuracy: %.2f%%' % (score[1]*100), '| Loss: %.5f' % (score[0]))

print("[INFO] Sumarizando loss e accuracy para os datasets 'train' e 'test'")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,EPOCHS), classifier.history["loss"], label="train_loss")
plt.plot(np.arange(0,EPOCHS), classifier.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,EPOCHS), classifier.history["acc"], label="train_acc")
plt.plot(np.arange(0,EPOCHS), classifier.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('../models/graphics/'+FILE_NAME+file_date+'.png', bbox_inches='tight')

print('[INFO] Gerando imagem do modelo de camadas da CNN')
plot_model(model, to_file='../models/image/'+FILE_NAME+file_date+'.png', show_shapes = True)

print('\n\n')