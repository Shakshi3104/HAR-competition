import numpy as np
import random

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

import dataset

from tensoract.applications.vgg16 import VGG16
from tensoract.applications.mobilenet import MobileNet

from tfxtend.keras.callbacks import ConfusionMatrixLogger, FMeasureLogger
from tfxtend.keras.callbacks.benckmark import PerformanceLogger

from sklearn.metrics import accuracy_score
from tfxtend.metrics import confusion_error_matrix

CLASSES = 6
window_size = 512
stride = 512
batch = 256
epochs = 100

SEED = 0

if __name__ == "__main__":
    print(tf.__version__)
    print(tf.test.gpu_device_name())

    # Set random seed
    np.random.seed(SEED)
    random.seed(SEED)

    # Set data path
    paths = ["./HASC_Apple_100/配布用/dataset_{}".format(i) for i in range(4)]

    data = []
    target = []

    # Load data
    for path in paths:
        hasc = dataset.HASC(path)
        data_, target_, _ = hasc.load(window_size, stride)

        data.append(data_.reshape(-1, window_size*3, 1))
        target.append(target_)

    # train test
    x_train = np.concatenate(data[:3])
    y_train = np.concatenate(target[:3])
    x_test = data[-1]
    y_test = target[-1]

    y_train = to_categorical(y_train, num_classes=CLASSES)
    y_test_ = to_categorical(y_test, num_classes=CLASSES)

    print(x_train.shape)
    print(y_test_.shape)

    # Load model
    model = VGG16(include_top=True, weights=None, input_shape=x_train.shape[1:])
    # model = MobileNet(include_top=True, weights=None, input_shape=x_train.shape[1:])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Callbacks
    cf_callback = ConfusionMatrixLogger(model, x_test, y_test,
                                        label_list=hasc.label_list)
    fm_callback = FMeasureLogger(model, x_test, y_test,
                                 label_list=hasc.label_list)

    bench_callback = PerformanceLogger()

    # Training
    stack = model.fit(x=x_train, y=y_train, batch_size=batch, epochs=epochs,
                      validation_data=(x_test, y_test_),
                      verbose=1, callbacks=[cf_callback, fm_callback, bench_callback])

    predict = model(x_test)
    predict = np.argmax(predict, axis=1)

    accuracy = accuracy_score(y_test, predict)
    print("{}%".format(accuracy * 100))

    cf = confusion_error_matrix(predict, y_test, target_names=["stay", "walk", "jog", "skip", "stUp", "stDown"])
    print(cf)


