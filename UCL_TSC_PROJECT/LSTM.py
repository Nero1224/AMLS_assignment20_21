from keras import models, layers, optimizers
import tensorflow as tf

# gpu memory management
# GPU memory management
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

model = models.Sequential()
model.add(layers.LSTM(2, input_shape=(2, 1)))
model.add(layers.Dense(2))
model.add(layers.Activation('sigmoid'))

model.compile(loss='mse',
              optimizer=optimizers.SGD(lr=0.4, momentum=0.3),
              metrics=['acc'])