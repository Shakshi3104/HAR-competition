import os
import pandas as pd
import numpy as np


class HASC:
    def __init__(self, path):
        self.path = path
        self.label_list = ["stay", "walk", "jog", "skip", "stUp", "stDown"]
        self.person_list = None

    def __load_hasc(self):
        label_path = os.path.join(self.path, "label.csv")
        labels = pd.read_csv(label_path)

        data = []

        self.person_list = labels["person"].unique()

        for filename in labels["name"]:
            filepath = os.path.join(self.path, filename)
            data_ = pd.read_csv(filepath)
            data.append(data_)

        return {"data": data, "activity": labels["act"].values.tolist(), "person": labels["person"].values.tolist()}

    def __load(self):
        """
        時系列分割する
        :return:
        """
        raw_data = self.__load_hasc()

        data = [data_.values for data_ in raw_data["data"]]
        data = np.array(data)
        target = np.array(raw_data["activity"])

        return data, target, raw_data["person"]

    def load(self, features=None):
        if features is None:
            return self.__load()
        else:
            raise NotImplementedError


def load_hasc(path="./HASC_Apple_100/配布用/"):
    """
    Parameters
    -------------
    path:
    window_size:
    stride:

    Returns
    -------------
    x_train, y_train, x_test, y_test
    """
    paths = [os.path.join(path, "dataset_{}".format(i)) for i in range(4)]

    data = []
    target = []

    # Load data
    for path in paths:
        hasc = HASC(path)
        data_, target_, _ = hasc.load()

        data.append(data_)
        target.append(target_)

    # train test
    # dataset_0-2をtrain, dataset_3をtest
    x_train = np.concatenate(data[:3])
    y_train = np.concatenate(target[:3])
    x_test = data[-1]
    y_test = target[-1]

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    hasc = HASC("./HASC_Apple_100/配布用/dataset_0")
    data, target, subject = hasc.load()
