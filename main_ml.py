import numpy as np
import pandas as pd
import scipy.stats as sp

import random

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from tfxtend.metrics import confusion_error_matrix

import dataset
import features
import metrics

CLASSES = 6
SEED = 0

if __name__ == "__main__":
    # Set random seed
    np.random.seed(SEED)
    random.seed(SEED)

    # Load dataset
    x_train, y_train, x_test, y_test = dataset.load_hasc()
    print(x_train.shape)
    print(y_test.shape)

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
        ('zcr', features.Feature(features.zcr)),
        ('power_spectrum_feature', features.Feature(features.fft_features))
    ]
    combined = FeatureUnion(extractors)
    # features = combined.fit_transform(data)

    # classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=SEED)

    pipeline = Pipeline([('feature_extractor', combined),
                         ('classifier', clf)])

    # training
    pipeline.fit(x_train, y_train)

    # predict
    predict = pipeline.predict(x_test)

    # calc metrics
    metrics.calc_metrics(y_test, predict)

    # test data for competition
