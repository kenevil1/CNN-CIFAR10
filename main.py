from keras import datasets
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPool2D, Activation, Dense, Flatten, Dropout
from keras.backend.tensorflow_backend import set_session
from tensorflow import Session, ConfigProto
import numpy as np
import cv2

CATEGORIES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
IMG_SIZE = 32

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(Session(config=config))

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

def load_model(model):
    # load json and create model
    json_file = open('modelweights.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("modelweights.h5")

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("modelweights.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modelweights.h5")

def format_data(img):
    img = img / 255
    img = np.array(img)
    img = img.reshape(-1, 32, 32, 3)
    return img

def model_predict(img):
    test = cv2.resize(cv2.imread(img), (IMG_SIZE, IMG_SIZE))
    test = format_data(test)

    print(CATEGORIES[np.argmax(model.predict(test))])

x_train = format_data(x_train)

model = Sequential([
    Conv2D(64, (3,3), input_shape=x_train.shape[1:]),
    MaxPool2D((2,2)),
    Activation('relu'),
    Dropout(0.25),

    Conv2D(32, (3,3)),
    MaxPool2D((2,2)),
    Activation('relu'),
    Dropout(0.25),
    Flatten(),

    Dense(10),
    Activation('softmax'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def fit_model():
    model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.1)

def evaluate_model(x_test):
    x_test = format_data(x_test)
    print(model.evaluate(x_test, y_test, batch_size=32))

### Main functions for model
fit_model()
save_model(model)
#load_model(model)
#evaluate_model(x_test)
model_predict("TestImages/toucan.jpg")
