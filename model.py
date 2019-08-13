from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten

def color_net(num_classes):
    model = Sequential()

    model.add(Convolution2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                            input_shape=(224, 224, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Convolution2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=num_classes, activation='softmax'))

    model.summary()
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model