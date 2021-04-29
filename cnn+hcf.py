import numpy as np
import pandas as pd
import scipy.stats as sp
import random

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.pipeline import FeatureUnion

from pathlib import Path

from tfxtend.keras.callbacks import ConfusionMatrixLogger, FMeasureLogger
from tfxtend.benckmark import PerformanceLogger

import dataset
import utils
import features
from models import VGG16_GAP_HCF

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

    # feature extraction
    extractors = [
        ('min', features.Feature(np.amin)),
        ('max', features.Feature(np.amax)),
        ('mean', features.Feature(np.mean)),
        ('std', features.Feature(np.std)),
        ('first_quartiles', features.Feature(features.first_quartiles)),
        ('median', features.Feature(np.median)),
        ('third_quartiles', features.Feature(features.third_quartiles)),
        ('iqr', features.Feature(sp.iqr)),
        ('corrcoef', features.Feature(features.corrcoef)),
        ('abs_corrcoef', features.Feature(features.abs_corrcoef)),
        ('frame_init', features.Feature(features.frame_init)),
        ('frame_end', features.Feature(features.frame_end)),
        ('intensity', features.Feature(features.intensity)),
        ('skewness', features.Feature(features.skewness)),
        ('kurtosis', features.Feature(features.kurtosis)),
        ('zcr', features.Feature(features.zcr)),
        ('power_spectrum_features_8', features.Feature(features.fft_features))
    ]
    combined = FeatureUnion(extractors)
    x_train_f = combined.fit_transform(x_train)
    x_test_f = combined.fit_transform(x_test)

    # Load model
    model = VGG16_GAP_HCF(include_top=True, weights=None, input_shape=x_train.shape[1:], num_features=x_train_f.shape[-1])
    model.compile(optimizer=Adam(learning_rate=5e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Callbacks
    label_list = ["stay", "walk", "jog", "skip", "stUp", "stDown"]
    cf_callback = ConfusionMatrixLogger(model, [x_test, x_test_f], y_test,
                                        label_list=label_list)
    fm_callback = FMeasureLogger(model, [x_test, x_test_f], y_test,
                                 label_list=label_list)

    bench_callback = PerformanceLogger()

    # training
    stack = model.fit(x=[x_train, x_train_f], y=y_train, batch_size=batch, epochs=epochs,
                      validation_data=([x_test, x_test_f], y_test_),
                      verbose=1, callbacks=[cf_callback, fm_callback, bench_callback])

    predict = model([x_test, x_test_f])
    predict = np.argmax(predict, axis=1)

    # calc metrics
    utils.calc_metrics(y_test, predict)

    # test for competition
    def test(model, extractor, path="./HASC_Apple_100/配布用/test/"):
        print("evaluate test...")
        files = list(Path(path).glob('*.csv'))
        x = np.array([pd.read_csv(f).values.copy() for f in files])

        feature = extractor.fit_transform(x)

        predict = model.predict([x, feature])

        # one-hot vectorの場合
        if predict.shape == (len(predict), 6):
            predict = np.argmax(predict, axis=1)

        result = pd.DataFrame()
        result['name'] = list(map(lambda x: x.name, files))
        result['pred'] = predict
        print("write output.csv")
        result.to_csv('./HASC_Apple_100/配布用/output.csv', header=False, index=False)

    test(model, combined)
