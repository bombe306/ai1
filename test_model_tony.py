import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

# === [ Step 6 ] ===
# Show MNIST graphic number
def show_mnist_graphic_number(img):
    img = np.array(img, dtype='float')
    pixels = img.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

# Load the saved model
new_model = tf.keras.models.load_model('mnist_trained.keras')

# Show the model architecture
new_model.summary()

# Make a prediction on an image

img1 = x_train[1]
print(new_model.predict(np.reshape(img1, newshape=(1, 28, 28))))

# Show the image
show_mnist_graphic_number(img1)

img1 = x_train[6]
print(new_model.predict(np.reshape(img1, newshape=(1, 28, 28))))

# Show the image
show_mnist_graphic_number(img1)

img1 = x_train[3513]
print(new_model.predict(np.reshape(img1, newshape=(1, 28, 28))))

# Show the image
show_mnist_graphic_number(img1)

img1 = x_train[10123]
print(new_model.predict(np.reshape(img1, newshape=(1, 28, 28))))

# Show the image
show_mnist_graphic_number(img1)

img1 = x_train[43213]
print(new_model.predict(np.reshape(img1, newshape=(1, 28, 28))))

# Show the image
show_mnist_graphic_number(img1)