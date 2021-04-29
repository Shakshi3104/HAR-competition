from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model

from tensoract.applications.vgg16 import VGG16


def VGG16_GAP(include_top=True, weights=None, input_shape=None, classes=6, classifier_activation='softmax'):
    backbone = VGG16(include_top=False, weights=weights, pooling='avg', input_shape=input_shape)
    y = Dense(classes, activation=classifier_activation)(backbone.output)

    model = Model(inputs=backbone.input, outputs=y)
    return model


def VGG16_GAP_HCF(include_top=True, weights=None, input_shape=None, num_features=72, classes=6, classifier_activation='softmax'):
    backbone = VGG16(include_top=False, weights=weights, pooling='avg', input_shape=input_shape)

    inputs_hcf = Input(shape=(num_features,))
    x = Concatenate()([backbone.output, inputs_hcf])
    y = Dense(classes, activation=classifier_activation)(x)

    model = Model(inputs=[backbone.input, inputs_hcf], outputs=y)
    return model


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
