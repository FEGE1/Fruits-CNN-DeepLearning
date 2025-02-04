from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten ,Dense, Input
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
import os
from datetime import datetime
import time
import json
import codecs

DATA_SETS_DIR = os.path.join(Path(__file__).resolve().parent.parent,'fruits_dataset/fruits-360_dataset_100x100/fruits-360')

train_path = os.path.join(DATA_SETS_DIR,'Training')
test_path = os.path.join(DATA_SETS_DIR,'Test')

model = Sequential([
    Input(shape=(100, 100, 3)),
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

# Data Generation - Train - Test
Train_Datagen = ImageDataGenerator(rescale= 1./255, # rgb 0-255 arasında
                   shear_range= 0.3, #rastgele resmi sağa veya sola çevirme
                   horizontal_flip= True,
                   zoom_range= 0.3)

Test_Datagen = ImageDataGenerator(rescale= 1./255)



Train_Generator = Train_Datagen.flow_from_directory(train_path,                 
                                                    target_size= (100, 100),
                                                    batch_size= batch_size,
                                                    color_mode= 'rgb',
                                                    class_mode= 'categorical') # Belli bir kalıba uygunsa classları ve içindekileri otomatik okuma

Test_Generator = Train_Datagen.flow_from_directory(test_path,                 
                                                    target_size= (100, 100),
                                                    batch_size= batch_size,
                                                    color_mode= 'rgb',
                                                    class_mode= 'categorical')

# HIST
hist = model.fit(
    Train_Generator,
    steps_per_epoch= 1600 // batch_size, # Yukarıda generator kısmında kaç tane resim üreticeğinin belirtmemiştik, 1600 tane resim lazım olarak belirledik 
    epochs= 100,
    validation_data= Test_Generator,
    validation_steps= 800 // batch_size
)

time.sleep(1)

# Save Model
date = datetime.today()
date = str(date.day)+"-"+str(date.month)+"-"+str(date.year)+"-"+str(date.hour)+":"+str(date.minute)+":"+str(date.second)

model.save_weights('model_saves/model-{}.weights.h5'.format(date))

# Save Accuracy - Loss with Json
with open("model_saves/model-{}.json".format(date),"w") as f:
    json.dump(hist.history, f)

print(hist.history.keys())
plt.plot(hist.history['loss'], label = 'Train Loss')
plt.plot(hist.history['val_loss'], label = 'Validation Loss')
plt.figure()
plt.plot(hist.history['accuracy'], label = 'Train acc')
plt.plot(hist.history['val_accuracy'], label = 'Validation acc')
plt.legend()
plt.show()