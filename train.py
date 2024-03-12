import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Caminho para o diretório com as imagens processadas
dataset_dir = "/home/eduardo/Documentos/IA/Letras_Processed"
dataset_validation_dir = "/home/eduardo/Documentos/IA/Letras_Processed"

# Parâmetros de pré-processamento
image_size = (64, 64)  # Tamanho das imagens para o treinamento
batch_size = 64  # Número de amostras por lote

# Preparando os dados
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalização das imagens
    validation_split=0.2  # 20% dos dados para validação
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Modo "categorical" para classificação multiclasse
    subset='training'  # Dados de treinamento
)

validation_generator = train_datagen.flow_from_directory(
    dataset_validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Dados de validação
)


# Definição do modelo simplificado para imagens 64x64
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compilando o modelo
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=50
)

# Salvar o modelo treinado
model.save('sign_language_model_processed_64x64.h5')