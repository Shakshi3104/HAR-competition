import os
import numpy as np

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

import dataset

from tensoract.applications.vgg16 import VGG16

from tfxtend.keras.callbacks import ConfusionMatrixLogger, FMeasureLogger
from tfxtend.keras.callbacks.benckmark import PerformanceLogger

# from models import PoolingNet

CLASSES = 6
window_size = 512
stride = 512
batch = 256
epochs = 100

if __name__ == "__main__":
    print(tf.__version__)
    print(tf.test.gpu_device_name())

    # Set data path
    path = "./HASC_Apple_100/配布用/dataset_0"

    # Load data
    hasc = dataset.HASC(path)
    data, target, subject = hasc.load(window_size, stride)

    # Reshape
    data = data.reshape(-1, window_size*3, 1)
    target_ = to_categorical(target, num_classes=CLASSES)
    print(data.shape)
    print(target_.shape)

    # train-test split
    train_person = hasc.person_list[:30]
    train_index = [i for i, x in enumerate(subject) if x in train_person]
    test_index = [i for i, x in enumerate(subject) if x not in train_person]

    x_train, y_train = data[train_index], target_[train_index]
    x_test, y_test_, y_test = data[test_index], target_[test_index], target[test_index]
    print(x_train.shape)
    print(y_test_.shape)

    # Load model
    model = VGG16(include_top=True, weights=None, input_shape=x_train.shape[1:])
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


