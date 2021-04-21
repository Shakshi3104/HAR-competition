import numpy as np
import pandas as pd
import scipy.stats as sp

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tfxtend.metrics import confusion_error_matrix

import dataset
import features

CLASSES = 6
window_size = 512
stride = 512

if __name__ == "__main__":
    # Set data path
    paths = ["./HASC_Apple_100/配布用/dataset_{}".format(i) for i in range(4)]

    data = []
    target = []

    # Load data
    for path in paths:
        hasc = dataset.HASC(path)
        data_, target_, _ = hasc.load(window_size, stride)

        data.append(data_)
        target.append(target_)

    # train test
    x_train = np.concatenate(data[:3])
    y_train = np.concatenate(target[:3])
    x_test = data[-1]
    y_test = target[-1]

    # Feature extractor
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
        ('zcr', features.Feature(features.zcr))
    ]
    combined = FeatureUnion(extractors)
    # x = combined.fit_transform(data)

    pipeline = Pipeline([('feature_extractor', combined),
                         ('classifier', RandomForestClassifier(n_estimators=100, max_depth=100))])

    pipeline.fit(x_train, y_train)

    predict = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predict)
    print(accuracy)

    cf = confusion_error_matrix(predict, y_test, target_names=["stay", "walk", "jog", "skip", "stUp", "stDown"])
    print(cf)
