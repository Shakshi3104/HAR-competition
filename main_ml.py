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
    path = "./HASC_Apple_100/配布用/dataset_0"

    # Load data
    hasc = dataset.HASC(path)
    data, target, subject = hasc.load(window_size, stride)

    # train-test split
    train_person = hasc.person_list[:30]
    train_index = [i for i, x in enumerate(subject) if x in train_person]
    test_index = [i for i, x in enumerate(subject) if x not in train_person]

    x_train, y_train = data[train_index], target[train_index]
    x_test, y_test = data[test_index], target[test_index]

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
