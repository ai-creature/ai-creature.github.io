import numpy as np
import tensorflow as tf
from random import randrange
import tensorflow_probability as tfp

arr = np.array([i for i in range(3072)], )
arr = arr.reshape((64, 4, 12)) #.transpose((1, 2, 0))
arr = np.moveaxis(arr, 0, 0)

res, line = [], []
for i, tile in enumerate(arr):
  line.append(tile)
  if (i+1) % 8 == 0 and i:
    res.append(np.concatenate(line, 1))
    line = []

arr = np.stack(res)

# arr = arr.reshape((48, 64))
print(arr[0][1])

##########

_log_alpha = tf.Variable(0.0)
_alpha = tfp.util.DeferredTensor(_log_alpha, tf.exp)

print('just alpha = ', _alpha.numpy())
print('conv alpha = ', tf.convert_to_tensor(_alpha))

##########

import gym
from gym import spaces

action_space = spaces.Box( np.array([-1,-1,-1]), np.array([+1,+1,+1]))  # steer, gas, brake
print(-np.prod(action_space.shape))



batch_size = 1
timesteps = 1
size = 1
units = 16
shape = (timesteps, size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape = shape, batch_size = batch_size),
  tf.keras.layers.GRU(
    units,
    stateful = True,
    return_sequences = True
  )
])

model.compile(optimizer='adam',
              loss='mean_squared_error')

print(model.summary())

SAME = False

i = 0
prevIsBlack = False
x_batch, y_batch = [], []
x_steps, y_steps = [], []

while True:
  i += 1

  isBlack = randrange(4) == 2

  if SAME: prevIsBlack = isBlack 

  x_train = np.ones(size)   if isBlack else np.zeros(size)
  y_train = np.zeros(units) if prevIsBlack else np.ones(units)

  if not SAME: prevIsBlack = isBlack 

  x_steps.append(x_train)
  y_steps.append(y_train)

  if len(x_steps) < timesteps:
    continue

  x_batch.append(np.stack(x_steps))
  y_batch.append(np.stack(y_steps))

  x_steps, y_steps = [], []

  if len(x_batch) < batch_size:
    continue

  res = model.fit(np.stack(x_batch), np.stack(y_batch), 
    epochs = 1, 
    shuffle = False,
    batch_size = batch_size)

  print(res.history["loss"])

  x_batch = []
  y_batch = []