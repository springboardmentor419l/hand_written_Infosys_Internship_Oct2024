from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def build_mlp():
    # Build a simple MLP model
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))  # Flatten input image to 1D vector
    model.add(Dense(128, activation="relu"))  # Hidden layer with 128 neurons
    model.add(Dense(64, activation="relu"))   # Hidden layer with 64 neurons
    model.add(Dense(10, activation="softmax")) # Output layer with 10 classes (digits 0-9)
    return model

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape data for the MLP model (flatten images to 1D vectors)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Compile and train the model
model = build_mlp()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
