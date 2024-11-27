import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape the images
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

image = cv2.imread('digit.png', cv2.IMREAD_GRAYSCALE)

# Resize the image to 28x28
image = cv2.resize(image, (28, 28))

# Invert the colors
image = cv2.bitwise_not(image)

# Normalize the image
image = image.astype('float32') / 255

# Reshape the image
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=-1)

# Predict the digit
prediction = np.argmax(model.predict(image))

print("Predicted Digit:", prediction)