import tensorflow as tf
from tensorflow import keras

import numpy as np

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

print(f"Predicting {test_images.shape[0]} test images")
raw_preds = model.predict(test_images)
preds = np.argmax(raw_preds, axis=1)

correct_predictions = np.equal(test_labels, preds)
accuracy = np.sum(correct_predictions) / correct_predictions.shape[0]
print(f"Test set accuracy: {accuracy:2f}")