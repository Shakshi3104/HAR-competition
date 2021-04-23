import numpy as np
import pandas as pd
import scipy.stats as sp

import random

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier

from tfxtend.benckmark import EstimatorPerformance

import dataset
import features
import utils

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
    # 特徴量数 75 (time domain: 16*3 = 48, frequency domain: 8*3 = 24)

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
    # features = combined.fit_transform(data)

    # 特徴量のリスト
    features_list = [f[0] for f in combined.transformer_list[:-1]]
    features_list += ["power_max", "power_2nd", "power_std", "power_first_quartiles", "power_median",
                      "power_third_quartiles", "power_iqr", "power_corrcoef"]

    features_list = ["{}-{}".format(axis, f) for f in features_list for axis in ["x", "y", "z"]]

    # classifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=100, random_state=SEED)

    pipeline = Pipeline([('feature_extractor', combined),
                         ('classifier', clf)])

    bench = EstimatorPerformance(pipeline)

    # training
    bench.fit(x_train, y_train)

    # predict
    predict = bench.predict(x_test)

    # calc metrics
    utils.calc_metrics(y_test, predict)

    # test for competition
    utils.test(pipeline)

    # importance
    # importanceの値が高い順に特徴量をソートする
    sorted_features_list = [f for (idx, f) in sorted(zip(np.argsort(clf.feature_importances_), features_list), reverse=True)]
    feature_importance = pd.DataFrame({"feature": sorted_features_list,
                                       "importance": sorted(clf.feature_importances_, reverse=True)})
    print(feature_importance)
