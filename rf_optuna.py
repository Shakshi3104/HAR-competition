import numpy as np
import pandas as pd
import scipy.stats as sp

import optuna

import random

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

import dataset
import features

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
        ('power_spectrum_features_8', features.Feature(features.fft_features))
    ]
    combined = FeatureUnion(extractors)
    # features = combined.fit_transform(data)

    # optunaの目的関数
    def objective(trial: optuna.trial.Trial):

        params = {
            "n_estimators": trial.suggest_int('n_estimators', 10, 300),
            "max_depth": trial.suggest_int('max_depth', 10, 300),
        }

        clf = RandomForestClassifier(**params, random_state=SEED)

        pipeline = Pipeline([('feature_extractor', combined),
                             ('classifier', clf)])

        # training
        pipeline.fit(x_train, y_train)

        # predict
        predict = pipeline.predict(x_test)

        return accuracy_score(y_test, predict)


    # Optunaで最適化
    opt = optuna.create_study(direction='maximize')
    opt.optimize(objective, n_trials=30)

    trial = opt.best_trial
    print("")
    print("best acc: {}".format(trial.value))
    print("parameters: {}".format(trial.params))
