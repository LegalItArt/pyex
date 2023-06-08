import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
import tensorflow  as tf

#model_and = keras.Sequential(keras.layers.Dense(units=1, input_shape=[[2]]))

#입력이 2개 들어가고 만든것에서 하나를 출력하는 것으로 바꿈
model_and = keras.Sequential(
    [keras.layers.Dense(units=3, input_shape=[2], activation='relu'),
                             keras.layers.Dense(units=1)
    ]
)
#활성화 함수
#입력신호의 총합을 사용하지 않고 출력신호로 변환하는 함수

#random으로 돌리므로 결과값 동일하게 하기위해
tf.random.set_seed(0)

#훈련데이터
x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[0],[0],[1]]

model_and.compile(loss="mse", optimizer='adam')

result_before = model_and.predict(x_train)
print("훈련전 예측값: ", result_before)
# 1/1 [==============================] -1/1 [==============================] - 0s 82ms/step
# 훈련전 예측값:  [[ 0.       ]
#  [ 0.6666167]
#  [-0.3990256]
#  [ 0.4972965]]

# model_and.fit(x_train, y_train, epochs=10000, verbose=0)
loss_his = model_and.fit(x_train, y_train, epochs=10000, verbose=0)

plt.style.use('default')
plt.rcParams['figure.figsize'] = [4,3]
plt.rcParams['font.size'] = 14


loss =loss_his.history['loss']
plt.plot(loss)
plt.xlabel('count')
plt.ylabel('loss')
plt.show()

result_after = model_and.predict(x_train)
print("훈련한 후 예측값: ", result_after)
# 1/1 [==============================] -1/1 [==============================] - 0s 31ms/step
# 훈련한 후 예측값:  [[4.5318642e-35]   
#  [4.5318642e-35]
#  [4.5318642e-35]
#  [1.0000000e+00]]


#텐서플로케라스 #원리와응용 #딥러닝 # 


