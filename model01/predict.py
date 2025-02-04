from keras._tf_keras.keras.models import load_model, Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten ,Dense, Input
import numpy as np
from keras._tf_keras.keras.preprocessing import image
import matplotlib.pyplot as plt
import json
import os

# Modeli Yükleyin
model_path = 'model_saves/model-4-2-2025-19:3:22.weights.h5' 
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

# Modelin ağırlıklarını yükleyin
model.load_weights(model_path)

# Sınıf Etiketlerini Yükleyin (JSON dosyasından)
labels_json_path = "model_saves/class_labels.json"  # Modelin eğitildiği JSON dosyasını buraya koyun
with open(labels_json_path, "r") as f:
    labels = json.load(f)['train_labels']  # Burada doğru etiketleri almanız gerekebilir (eğitimde kullanılan etiketler)

# Resmi Yükleyin ve Hazırlayın
img_path = 'predict_img/predict.jpg'  # Tahmin etmek istediğiniz fotoğrafın yolu
img = image.load_img(img_path, target_size=(100, 100))  # Resmi 100x100 boyutlarına getirin
img_array = image.img_to_array(img)  # Resmi numpy array'ine çevirin
img_array = np.expand_dims(img_array, axis=0)  # Batch dimension ekleyin (model bunu bekliyor)
img_array /= 255.0  # Normalize edin (0-1 arası)

pred = model.predict(img_array)  # Resmi tahmin için modele gönderin

# Tahmin edilen sınıfı ve etiketini yazdırın
predicted_class = np.argmax(pred, axis=1)[0]  # Tahmin edilen sınıf numarası
print(f"Tahmin Edilen Sınıf Numarası: {predicted_class}")  # Bu satır hata ayıklamak için

# Etiketleri yazdırın
if str(predicted_class) in labels:
    predicted_label = labels[str(predicted_class)]  # JSON'dan sınıf numarasına karşılık gelen etiket
    print(f"Tahmin Edilen Meyve: {predicted_label}")
else:
    print(f"Etiket bulunamadı! Sınıf Numarası: {predicted_class}")

# Resmi Görüntüle
plt.imshow(img)
plt.title(f"Predicted: {predicted_label}")
plt.show()
