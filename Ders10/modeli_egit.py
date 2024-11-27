import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# MNIST veri setini yükle
(x_egitim, y_egitim), (x_test, y_test) = mnist.load_data()

# Verileri normalize et
x_egitim, x_test = x_egitim / 255.0, x_test / 255.0

# Etiketleri kategorik hale getir
y_egitim = to_categorical(y_egitim, 10)
y_test = to_categorical(y_test, 10)

# Modeli oluştur
model = Sequential(
    [
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

# Modeli derle
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Modeli eğit
model.fit(x_egitim, y_egitim, epochs=5, validation_data=(x_test, y_test))

# Modeli kaydet
model.save("model.h5")
