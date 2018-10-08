import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
X = numpy.load('vehicleimg')
Y = numpy.load('vehicletrgt')
for i in range(len(Y)):
    Y[i] = Y[i] - 2

input_shape1, input_shape2 = len(X[0]), len(X[0][0])

#X = X.reshape(X.shape[0], 3, input_shape1, input_shape2)
dataset_size = len(X)
Y = np_utils.to_categorical(Y, 3)
#print(X.shape)
model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(7, 7), activation='relu', data_format='channels_last', input_shape=(input_shape1,input_shape2, 3)))
#print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2), strides=2, data_format='channels_last'))
#print(model.output_shape)
model.add(Convolution2D(filters=64, kernel_size=(7, 7), activation='relu', data_format='channels_last'))
#print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2), strides=2, data_format='channels_last'))
#print(model.output_shape)
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y,
          batch_size=32, nb_epoch=1, verbose=1)
