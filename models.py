from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model


def PoolingNet(input_shape=None, classes=6, classifier_activation='softmax'):

    inputs = Input(input_shape)
    x_max_1 = MaxPooling1D(pool_size=2, padding='same')(inputs)
    x_avg_1 = AveragePooling1D(pool_size=2, padding='same')(inputs)
    x = Concatenate(axis=1)([x_max_1, x_avg_1])

    x_max_2 = MaxPooling1D(pool_size=2, padding='same')(x)
    x_avg_2 = AveragePooling1D(pool_size=2, padding='same')(x)
    x = Concatenate(axis=1)([x_max_2, x_avg_2])

    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(classes, activation=classifier_activation)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    model = PoolingNet(input_shape=(512*3, 1))
    print(model.summary())
