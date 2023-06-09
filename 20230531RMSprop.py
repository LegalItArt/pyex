import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras


o_model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])

# 손실최적화 함수 적용
o_model.compile(loss='mse', optimizer='RMSprop')

xs = [0]
ys = [[0,1,0]]

# log_dir = "logs/fit/"
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# loss_his = o_model.fit(xs, ys, epochs=1000000,callbacks=[tensorboard_callback])
loss_his = o_model.fit(xs, ys, epochs=1000)
xt = [1]
pred = o_model.predict(xt)
print(pred)

o_model.evaluate([1],[[0,1,0]])


# ValueError: `distribute` argument in compile is not available in TF 2.0. Please create the model under the `strategy.scope()`. Received: none.

plt.style.use('default')
plt.rcParams['figure.figsize'] = [4,3]
plt.rcParams['font.size'] = 14


loss =loss_his.history['loss']
plt.plot(loss)
plt.xlabel('count')
plt.ylabel('loss')
plt.show()



# Epoch 998/1000
# 1/1 [==============================] - ETA: 0s 1/1 [==============================] - 0s 3ms/step - loss: 5.3748e-07
# Epoch 999/1000
# 1/1 [==============================] - ETA: 0s 1/1 [==============================] - 0s 3ms/step - loss: 5.3034e-07
# Epoch 1000/1000
# 1/1 [==============================] - ETA: 0s 1/1 [==============================] - 0s 2ms/step - loss: 5.2329e-07
# 1/1 [==============================] - 0s 60ms/step
# [[ 1.1966408  0.3868481 -0.9103187]]


# =========] - ETA: 0s - los1/1 [==============================] - 0s 3ms/step - loss: 5.4472e-07
# Epoch 998/1000
# 1/1 [==============================] - ETA: 0s - los1/1 [==============================] - 0s 3ms/step - loss: 5.3748e-07
# Epoch 999/1000
# 1/1 [==============================] - ETA: 0s - los1/1 [==============================] - 0s 3ms/step - loss: 5.3034e-07
# Epoch 1000/1000
# 1/1 [==============================] - ETA: 0s - los1/1 [==============================] - 0s 3ms/step - loss: 5.2329e-07
# 1/1 [=====================1/1 [==============================] - 0s 55ms/step
# [[ 0.7245085   0.99174875 -0.4208542 ]]
# 1/1 [==============================] - ETA: 0s - los1/1 [==============================] - 0s 77ms/step - loss: 0.2340
# o_model.compile(loss='sparse_categorical_crossentropy', optimizer='adagrad' ,distribute='none')

# raise ValueError(  




# 100만돌리기


#  lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999917/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999918/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999919/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 5ms/step - loss: 0.0000e+00
# Epoch 999920/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999921/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999922/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 5ms/step - loss: 0.0000e+00
# Epoch 999923/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999924/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999925/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999926/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999927/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 2ms/step - loss: 0.0000e+00
# Epoch 999928/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999929/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999930/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 6ms/step - loss: 0.0000e+00
# Epoch 999931/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 5ms/step - loss: 0.0000e+00
# Epoch 999932/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 5ms/step - loss: 0.0000e+00
# Epoch 999933/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 5ms/step - loss: 0.0000e+00
# Epoch 999934/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999935/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999936/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999937/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999938/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999939/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 6ms/step - loss: 0.0000e+00
# Epoch 999940/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999941/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999942/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999943/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999944/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 5ms/step - loss: 0.0000e+00
# Epoch 999945/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999946/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999947/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999948/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999949/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999950/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999951/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999952/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999953/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999954/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 5ms/step - loss: 0.0000e+00
# Epoch 999955/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 7ms/step - loss: 0.0000e+00
# Epoch 999956/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 5ms/step - loss: 0.0000e+00
# Epoch 999957/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999958/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999959/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999960/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999961/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999962/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999963/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999964/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999965/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999966/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999967/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999968/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999969/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999970/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999971/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999972/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 2ms/step - loss: 0.0000e+00
# Epoch 999973/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999974/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 5ms/step - loss: 0.0000e+00
# Epoch 999975/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999976/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999977/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999978/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999979/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999980/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 6ms/step - loss: 0.0000e+00
# Epoch 999981/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999982/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999983/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999984/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999985/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999986/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 2ms/step - loss: 0.0000e+00
# Epoch 999987/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 5ms/step - loss: 0.0000e+00
# Epoch 999988/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 2ms/step - loss: 0.0000e+00
# Epoch 999989/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 3ms/step - loss: 0.0000e+00
# Epoch 999990/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# Epoch 999991/1000000
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 6ms/step - loss: 0.0000e+00
# Epoch 999992/1000000
# =======================] - ETA: 0s - lo1/1 [==============================] - 0s 4ms/step - loss: 0.0000e+00
# 1/1 [==============================] - 0s 215ms/step
# [[-0.32036757  0.09834909  0.34821296]]
# 1/1 [==============================] - ETA: 0s - lo1/1 [==============================] - 0s 133ms/step - loss: 0.3456
