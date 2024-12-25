from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def build_cnn():
    # Build a simple CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))  # Conv layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling to reduce spatial size
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))  # Second convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Second max pooling layer
    model.add(Flatten())  # Flatten the output from convolution layers
    model.add(Dense(128, activation="relu"))  # Fully connected layer
    model.add(Dense(10, activation="softmax")) # Output layer (10 classes)
    return model

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape data for the CNN model (add channel dimension for grayscale images)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Compile and train the model
model = build_cnn()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
