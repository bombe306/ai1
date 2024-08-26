import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# The following line improves formatting when outputting NumPy arrays.
np.set_printoptions(linewidth=200)

# === [ Step 1 ] ===
# Import data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# === [ Step 2 ] ===
# Data cleaning - To make the training easier. Easier = less time and resources
# Normalization is the task of scaling all values in the same level between 0 and 1.
# We divide by 255 because the color spectrum is between 0 and 255.
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

# === [ Step 3 ] ===
def plot_curve(epochs, hist, list_of_metrics):
    """Plot a curve of one or more classification metrics vs. epoch."""
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Value')

    for m in list_of_metrics:
        # Show accuracy as 0 to 100% for better reading
        if 'acc' in m:
            x = hist[m] * 100
        else:
            x = hist[m]

        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()
    plt.savefig(list_of_metrics[0] + '.png')
    plt.show()

# === [ Step 4 ] ===
def create_model(my_learning_rate):
    """Create and compile a deep neural net."""

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_features, train_label, epochs, batch_size, validation_split):
    """Train the model by feeding it data."""
    history = model.fit(x=train_features, y=train_label, batch_size=batch_size, epochs=epochs, shuffle=True,
                        validation_split=validation_split)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

# === [ Step 5 ] ===
learning_rate = 0.01
epochs = 250
batch_size = 32
validation_split = 0.2

my_model = create_model(learning_rate)
epochs, hist = train_model(my_model, x_train_normalized, y_train, epochs, batch_size, validation_split)

plot_curve(epochs, hist, ['accuracy', 'val_accuracy'])
plot_curve(epochs, hist, ['loss', 'val_loss'])

print('\n Evaluate the new model against the train set:')
my_model.evaluate(x=x_train_normalized, y=y_train, batch_size=batch_size)

print('\n Evaluate the new model against the test set:')
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

my_model.save('mnist_trained.keras')