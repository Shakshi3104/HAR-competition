# HAR competition

## Feature extraction and classical machine learning

[main_ml.py](main_ml.py)

### Features

- min
- max
- mean
- standard deviation
- 1st quartiles
- median
- 3rd quartiles
- interquartile range
- correlation coefficient between axes
- correlation coefficient of absolute values between axes
- initial value in the frame
- final value in the frame
- intensity
- skewness
- kurtosis
- zero-crossing rate
- max of power spectrum
- 2nd max of power spectrum
- standard deviation of power spectrum
- 1st quartiles of power spectrum
- median of power spectrum
- 3rd quartiles of power spectrum
- interquartile range of power spectrum
- correlation coefficient of power spectrum between axes

### Classifier

Random Forest

- n_estimators = 200
- max_depth = 100

### Accuracy

0.80854

## Deep learning

[main_dl.py](main_dl.py)

### Model architecture

PyramidNet 34

- Implementation: [TensorFlow_CNN_Collections_for_HASC.applications.pyramidnet.PyramidNet34](https://github.com/haselab-dev/TensorFlow_CNN_Collections_for_HASC/blob/master/applications/pyramidnet.py)

### Accuracy

0.93320


