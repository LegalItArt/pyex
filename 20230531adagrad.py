import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras


o_model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])

# 손실최적화 함수 적용

o_model.compile(loss='mse', optimizer='sgd')

xs = [0]
ys = [[0,1,0]]

loss_his = o_model.fit(xs, ys, epochs=1000)

xt = [1]
pred = o_model.predict(xt)
print(pred)

# Epoch 998/1000
# 1/1 [==============================] - ETA: 0s 1/1 [==============================] - 0s 3ms/step - loss: 5.3748e-07
# Epoch 999/1000
# 1/1 [==============================] - ETA: 0s 1/1 [==============================] - 0s 3ms/step - loss: 5.3034e-07
# Epoch 1000/1000
# 1/1 [==============================] - ETA: 0s 1/1 [==============================] - 0s 2ms/step - loss: 5.2329e-07
# 1/1 [==============================] - 0s 60ms/step
# [[ 1.1966408  0.3868481 -0.9103187]]

o_model.evaluate([1],[[0,1,0]])


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
# ValueError: `distribute` argument in compile is not available in TF 2.0. Please create the model under the `strategy.scope()`. Received: none.

plt.style.use('default')
plt.rcParams['figure.figsize'] = [4,3]
plt.rcParams['font.size'] = 14


loss =loss_his.history['loss']
plt.plot(loss)
plt.xlabel('count')
plt.ylabel('loss')
plt.show()



# zj
o_model.compile(loss='sparse_categorical_crossentropy', optimizer='adagrad' )