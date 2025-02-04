from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten ,Dense, Input
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
import os

DATA_SETS_DIR = os.path.join(Path(__file__).resolve().parent.parent,'fruits_dataset/fruits-360_dataset_100x100/fruits-360')

train_path = os.path.join(DATA_SETS_DIR,'Training')
test_path = os.path.join(DATA_SETS_DIR,'Test')

model = Sequential([
    Input(shape=(100, 100, 3)),  # ✅ input_shape yerine Input() kullanılıyor
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(1024),
    Activation('relu'),
    Dropout(0.5),

    Dense(141),  # output = class sayısı
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer= 'rmsprop',
              metrics=["accuracy"])

batch_size = 32 # her iterasyonda 32 tane resim train edilecek

model.summary()