import numpy as np
import random

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from tensoract.applications.vgg16 import VGG16
from tensoract.applications.mobilenet import MobileNet
from tensoract.applications.mobilenet_v2 import MobileNetV2
from tensoract.applications.efficientnet import EfficientNetB0
from tensoract.applications.pyramidnet import PyramidNet18
from tensoract.applications.resnet import ResNet18

from tfxtend.keras.callbacks import ConfusionMatrixLogger, FMeasureLogger
from tfxtend.benckmark import PerformanceLogger

import dataset
import utils

from models import VGG16_GAP

CLASSES = 6

batch = 128
epochs = 100

SEED = 0


if __name__ == "__main__":
    print(tf.__version__)
    print(tf.test.gpu_device_name())

    # Set random seed
    np.random.seed(SEED)
    random.seed(SEED)

    # Load dataset
    x_train, y_train, x_test, y_test = dataset.load_hasc()
    print(x_train.shape)
    print(y_test.shape)

    y_train = to_categorical(y_train, num_classes=CLASSES)
    y_test_ = to_categorical(y_test, num_classes=CLASSES)

    print(x_train.shape)
    print(y_test_.shape)

    # Load model
    models = {
        "VGG16": VGG16(include_top=True, weights=None, input_shape=x_train.shape[1:]),
        "VGG16-GAP": VGG16_GAP(include_top=True, weights=None, input_shape=x_train.shape[1:]),
        # "MobileNet": MobileNet(include_top=True, weights=None, input_shape=x_train.shape[1:]),
        # "MobileNetV2": MobileNetV2(include_top=True, weights=None, input_shape=x_train.shape[1:]),
        # "EfficientNet B0": EfficientNetB0(include_top=True, weights=None, input_shape=x_train.shape[1:]),
        "ResNet 18": ResNet18(include_top=True, weights=None, input_shape=x_train.shape[1:]),
        "PyramidNet 18": PyramidNet18(include_top=True, weights=None, input_shape=x_train.shape[1:]),
    }

    for model_name, model in models.items():
        print(model_name)
        print("-" * 40)
        model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        # Callbacks
        label_list = ["stay", "walk", "jog", "skip", "stUp", "stDown"]
        cf_callback = ConfusionMatrixLogger(model, x_test, y_test,
                                            label_list=label_list)
        fm_callback = FMeasureLogger(model, x_test, y_test,
                                     label_list=label_list)

        bench_callback = PerformanceLogger()

        # training
        stack = model.fit(x=x_train, y=y_train, batch_size=batch, epochs=epochs,
                          validation_data=(x_test, y_test_),
                          verbose=1, callbacks=[cf_callback, fm_callback, bench_callback])

        predict = model(x_test)
        predict = np.argmax(predict, axis=1)

        # calc metrics
        utils.calc_metrics(y_test, predict)

        # test for competition
        utils.test(model)

        print("-" * 40)
