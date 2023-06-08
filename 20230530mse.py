import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras

mse_model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])
mse_model.compile(loss='mse')

# 동일 mse_model.compile(loss='mse',  optimizer='rmsprop') root mean square 제곱근

#MSE Mean Square of Errors .평균제곱오차 로 손실함수 설정
# model.compile(loss='mean_squared_error', optimizer='sgd')


# xs = np.array([-1.0, 0.0, 1.0, 2.0,3.0,4.0,5.0], dtype=float)
# ys = np.array([-3.0, -1.0, 1.0,3.0,5.0,7.0,9.0], dtype=float)
# # mse_model.fit(xs, ys, epochs=1500)

xt = [0]
pred = mse_model.predict(xt)
print(pred)

mse_model.evaluate(xt, [[0,1,0]])

print(mse_model.summary())

