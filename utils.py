from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tfxtend.metrics import confusion_error_matrix


def calc_metrics(y_true, y_pred, labels=["stay", "walk", "jog", "skip", "stUp", "stDown"]):
    accuracy = accuracy_score(y_true, y_pred)
    print("acc: {}%".format(accuracy * 100))
    f_measure = f1_score(y_true, y_pred, average='macro')
    print("f1 : {}%".format(f_measure * 100))

    print("confusion matrix")
    cf = confusion_error_matrix(y_pred, y_true, target_names=labels)
    print(cf)
    print("")


def test(model, path="./HASC_Apple_100/配布用/test/"):
    files = list(Path(path).glob('*.csv'))
    x = np.array([pd.read_csv(f).values.copy() for f in files])

    predict = model.predict(x)
    result = pd.DataFrame()
    result['name'] = list(map(lambda x: x.name, files))
    result['pred'] = predict
    result.to_csv('./HASC_Apple_100/配布用/output.csv', header=False, index=False)