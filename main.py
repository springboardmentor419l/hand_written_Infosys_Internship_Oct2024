from mlp_model import build_mlp
from cnn_model import build_cnn
from lenet_model import build_lenet
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def train_models():
    # Train MLP model
    mlp_model = build_mlp()
    mlp_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
    
    # Train CNN model
    cnn_model = build_cnn()
    cnn_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
    
    # Train LeNet model
    lenet_model = build_lenet()
    lenet_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

if __name__ == "__main__":
    train_models()
