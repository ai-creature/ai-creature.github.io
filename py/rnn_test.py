import numpy as np
import tensorflow as tf

batch_size = 10
timesteps = 11
size = 1
shape = (timesteps,size)
batch_shape = (batch_size, *shape)
units = 16
output_shape = (timesteps, units)

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(
    shape = shape, 
    batch_size = batch_size
  ),
  tf.keras.layers.GRU(
    units,
    stateful = True,
    return_sequences = True,
    return_state = False,
  )
  # tf.keras.layers.Dense(units)
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              # metrics=['accuracy']
              )

print(model.summary())

max_it = 2000
i = 0

x_batch, y_batch = [], []
x_steps, y_steps = [], []

while True:
  i += 1

  x_train = np.ones(size)         if i%3 == 0         else np.zeros(size) #tf.Tensor(np.ones(shape), shape=shape, dtype=int32)
  y_train = np.zeros(units) if i%4 == 0 else np.ones(units)

  x_steps.append(x_train)
  y_steps.append(y_train)

  if len(x_steps) < timesteps:
    continue

  x_batch.append(np.stack(x_steps))
  y_batch.append(np.stack(y_steps))
  x_steps, y_steps = [], []

  if len(x_batch) < batch_size:
    continue

  model.fit(np.stack(x_batch), np.stack(y_batch), 
    epochs = 1, 
    shuffle = False,
    batch_size = batch_size)

  model.evaluate(np.stack(x_batch), np.stack(y_batch))

  x_batch = []
  y_batch = []