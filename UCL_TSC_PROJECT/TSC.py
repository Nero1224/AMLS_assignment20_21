from keras import models
from keras import layers
from keras  import optimizers



model = models.Sequential()
model.add(layers.LSTM(50, input_shape=10))
model.add(layers.Dense(1, activation='softmax'))

model.summary()

model.compile(loss='mae',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['acc'])

history = model.fit()