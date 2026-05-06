import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os



# Stap 1: Data voorbereiden en inladen
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normeren van de pixelwaarden van 0-255 naar 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Stap 2: Model bouwen
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Afvlakken van de inputafbeeldingen
    Dense(128, activation='relu'),  # Verborgen laag met 128 neuronen en ReLU-activatie
    Dense(10, activation='softmax') # Uitvoerlaag met 10 neuronen (één voor elk cijfer) en softmax-activatie
])

# Stap 3: Model compileren
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Stap 4: Model trainen
model.fit(X_train, y_train, epochs=5)

# Stap 5: Model evalueren
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
