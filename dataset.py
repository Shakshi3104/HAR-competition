import os
import pandas as pd
import numpy as np

import sensorutils


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

    def __load(self, window_size=512, stride=512):
        """
        時系列分割する
        :return:
        """
        raw_data = self.__load_hasc()

        if window_size == 512 and stride == 512:
            data = [data_.values for data_ in raw_data["data"]]
            data = np.array(data)
            target = np.array(raw_data["activity"])

            return data, target, raw_data["person"]

        """
        コンペ的には以降のコードは必要ない
        """
        data = []
        target = []
        subject = []

        for data_, label_, person_ in zip(raw_data["data"], raw_data["activity"], raw_data["person"]):

            split_data = sensorutils.to_frames(data_.values, window_size=window_size, stride=stride)

            target_ = [label_] * len(split_data)
            target_ = np.array(target_)

            data.append(split_data)
            target.append(target_)

            subject += [person_] * len(split_data)

        # np.arrayに変換
        data = np.concatenate(data)
        target = np.concatenate(target)

        return data, target, subject

    def load(self, window_size=256, stride=256, features=None):
        if features is None:
            return self.__load(window_size, stride)
        else:
            raise NotImplementedError


def load_hasc(path="./HASC_Apple_100/配布用/", window_size=512, stride=512):
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
        data_, target_, _ = hasc.load(window_size, stride)

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
    data, target, subject = hasc.load(512, 512)
