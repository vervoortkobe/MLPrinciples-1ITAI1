import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
import numpy as np

# 1. Data inladen:
# Data voorbereiden en inladen
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Data verkennen
print("Shape van de dataset:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)
print("Eerste 5 labels van y_train:")
print(y_train[:5])
print("Unieke pixelwaarden in X_train:")
print(np.unique(X_train))
print("Unieke labels in y_train:")
print(np.unique(y_train))
print()

# 3. Data voorbereiden
# Normaliseren van de pixelwaarden van 0-255 naar 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# 4. Bouw een model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

# 5. Model compileren
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Model trainen
model.fit(X_train, y_train, epochs=5)

# 7. Model evalueren
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)