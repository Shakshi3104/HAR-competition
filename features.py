import numpy as np
import scipy.stats as sp

from sklearn.base import BaseEstimator, TransformerMixin


class BasePlumber(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self


class Feature(BasePlumber):
    def __init__(self, function=np.max):
        self.function = function

    def transform(self, X):
        data = [np.expand_dims(self.function(x, axis=0), 0) for x in X]
        data = np.concatenate(data)
        return data


"""
    特徴量は、3軸まとめてウィンドウごとに抽出する形で関数を書く
    
    Parameters
    -------------
    x: np.ndarray
        x.shape = (ウィンドウサイズ, 3)
    
    axis: int, optional
        特徴量を計算する軸, 使わなくても必要
    
    Returns
    ------------
    feature: np.ndarray
        feature.shape = (特徴量の次元数,)
"""


def first_quartiles(x, axis=0):
    return sp.scoreatpercentile(x, 25, axis=axis)


def third_quartiles(x, axis=0):
    return sp.scoreatpercentile(x, 75, axis=axis)


def abs_mean(x, axis=0):
    abs_ = np.abs(x)
    return np.mean(abs_, axis=axis)


def abs_min(x, axis=0):
    abs_ = np.abs(x)
    return np.amin(abs_, axis=axis)


def abs_max(x, axis=0):
    abs_ = np.abs(x)
    return np.amax(abs_, axis=axis)


def abs_std(x, axis=0):
    abs_ = np.abs(x)
    return np.std(abs_, axis=axis)


def rms(x, axis=0):
    return np.sqrt(np.square(x).mean(axis=axis))


def corrcoef(x, axis=0):
    def cor(x, y):
        return np.corrcoef(x, y)[1, 0]

    cor_ = [
        cor(x[:, 0], x[:, 1]),
        cor(x[:, 0], x[:, 2]),
        cor(x[:, 1], x[:, 2])
    ]

    cor_ = np.array(cor_)

    return cor_


def abs_corrcoef(x, axis=0):
    def cor(x, y):
        return np.corrcoef(x, y)[1, 0]

    abs_ = np.abs(x)

    cor_ = [
        cor(abs_[:, 0], abs_[:, 1]),
        cor(abs_[:, 0], abs_[:, 2]),
        cor(abs_[:, 1], abs_[:, 2])
    ]

    cor_ = np.array(cor_)

    return cor_


def frame_init(x, axis=0):
    return x[0]


def frame_end(x, axis=0):
    return x[-1]


def intensity(x, axis=0):
    def __intensity(x):
        n = len(x)
        tmp = [x[i] - x[i + 1] for i in range(0, n - 1)]
        return np.sum(tmp) / (n - 1)

    inte_ = [
        __intensity(x[:, 0]),
        __intensity(x[:, 1]),
        __intensity(x[:, 2])
    ]

    inte_ = np.array(inte_)
    return inte_


def skewness(x, axis=0):
    def __skewness(x):
        n = len(x)
        u = np.std(x)
        m = np.mean(x)

        tmp = [((x_ - m) / u) ** 3 for x_ in x]
        return n * np.sum(tmp) / ((n - 1) * (n - 2))

    skew_ = [
        __skewness(x[:, 0]),
        __skewness(x[:, 1]),
        __skewness(x[:, 2])
    ]

    skew_ = np.array(skew_)
    return skew_


def kurtosis(x, axis=0):
    def __kurtosis(x):
        n = len(x)
        u = np.std(x)
        m = np.mean(x)

        tmp = [((x_ - m) / u) ** 4 for x_ in x]
        return (n * (n + 1)) * np.sum(tmp) / ((n - 1) * (n - 2) * (n - 3)) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

    kurt_ = [
        __kurtosis(x[:, 0]),
        __kurtosis(x[:, 1]),
        __kurtosis(x[:, 2])
    ]

    kurt_ = np.array(kurt_)
    return kurt_


def zcr(x, axis=0):
    def __zcr(x):
        zero_crosses = np.nonzero(np.diff(x > np.mean(x)))[0]
        return len(zero_crosses) / (len(x) - 1)

    zcr_ = [
        __zcr(x[:, 0]),
        __zcr(x[:, 1]),
        __zcr(x[:, 2])
    ]

    zcr_ = np.array(zcr_)
    return zcr_


def fft_features(x, axis=0):
    # パワースペクトル
    fft_ = np.fft.fft(x, axis=axis)
    abs_ = np.abs(fft_)
    power = abs_ ** 2

    # 周波数
    ts = 1.0 / 100.0
    freqs = np.fft.fftfreq(power.shape[0], ts)
    idx = np.argsort(freqs)
    freqs = freqs[idx]

    def power_max(power, axis=0):
        return np.max(power, axis=axis)

    def power_2nd(power, axis=0):
        return np.sort(power, axis=axis)[-2]

    def power_std(power, axis=0):
        return np.std(power, axis=axis)

    def power_first_quartiles(power, axis=0):
        return first_quartiles(power, axis)

    def power_median(power, axis=0):
        return np.median(power, axis=axis)

    def power_third_quartiles(power, axis=0):
        return third_quartiles(power, axis)

    def power_iqr(power, axis=0):
        return sp.iqr(power, axis=axis)

    def power_corrcoef(power, axis=0):
        return corrcoef(power, axis)

    fft_feature = [
        power_max(power),
        power_2nd(power),
        power_std(power),
        power_first_quartiles(power),
        power_median(power),
        power_third_quartiles(power),
        power_iqr(power),
        power_corrcoef(power)
    ]

    fft_feature = np.concatenate(fft_feature)
    return fft_feature
