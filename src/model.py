import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Reshape
from keras.regularizers import l2


def build(input_shape: int, output_shape: tuple = (68, 2)):
    """Builds a model"""
    model = Sequential(name='KeypointsDetectorv3.1.0')
    model.add(Conv2D(64, (1, 1), padding='same',
              input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(64, (1, 1), kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
              strides=(2, 2), padding='valid'))

    model.add(Conv2D(96, (3, 3), kernel_regularizer=l2(0.001)))
    model.add(Conv2D(96, (5, 5), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
              strides=(2, 2), padding='valid'))

    model.add(Conv2D(64, (1, 1), kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
              strides=(2, 2), padding='valid'))

    model.add(Conv2D(128, (1, 1), kernel_regularizer=l2(0.001)))
    model.add(Conv2D(128, (3, 3), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(500, activation='relu'))
    model.add(Dense(output_shape[0] * output_shape[1]))
    model.add(Reshape(output_shape))
    return model


def compile_model(input_shape: int, output_shape: tuple, model=None):
    """Compiles the model with a given {input_size}"""
    if model is None:
        model = build(input_shape, output_shape)

    model.compile(
        loss=tf.keras.losses.mean_squared_error,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['mse', 'accuracy'])

    return model
