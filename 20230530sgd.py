import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras

# Dense 하나의 뉴런층을 구성
# units 뉴런또는 출력노드이 갯수
#input_shape 입력형태를 결정


#  간단하 ㄴ신경망 예제
# keras.Sequencial
# keras.layers.Dense(([keras.layers.Dense(units=출력노드수, input_shape=[입력형태])])
model = keras.Sequential([keras.layers.Dense(units=10, input_shape=[1])])

model.compile(loss='mean_squared_error', optimizer='sgd')

xs = np.array([-1.0, 0.0, 1.0, 2.0,3.0,4.0,5.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0,3.0,5.0,7.0,9.0], dtype=float)
model.fit(xs, ys, epochs=1500)

pred = model.predict([5.0])
print(pred)

print(model.summary())


#MSE Mean Square of Errors .평균제곱오차
# model.compile(loss='mean_squared_error', optimizer='sgd')




