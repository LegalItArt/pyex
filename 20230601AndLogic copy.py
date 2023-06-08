import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
import tensorflow  as tf
from keras.callbacks import EarlyStopping

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
# 1/1 [==============================] - ETA: 1/1 [==============================] - 0s 80ms/step
# 훈련전 예측값:  
# [[ 0.        ]        
#  [-0.2971428 ]
#  [ 0.        ]
#  [-0.03418684]]

model_and.fit(x_train, y_train, epochs=10000, verbose=0)


# 검증 손실(val_loss)이 10번 연속으로 개선되지 않으면 학습을 중지합니다.
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 모델 훈련
model_and.fit(x_train, y_train, epochs=10000, validation_data=(x_train, y_train), callbacks=[early_stopping])

# 학습이 중지된 이후에도 가장 좋은 모델의 가중치를 저장합니다.
model_and.save_weights('best_model.h5')


result_after = model_and.predict(x_train)
print("훈련한 후 예측값: ", result_after)
# 1/1 [==============================] -1/1 [==============================] - 0s 29ms/step
# 훈련한 후 예측값:  
# [[-0.01775046]
#  [ 0.00714871]
#  [ 0.1347434 ]
#  [ 0.9580456 ]]
loss_his = model_and.fit(x_train, y_train, epochs=10000, verbose=0)

plt.style.use('default')
plt.rcParams['figure.figsize'] = [4,3]
plt.rcParams['font.size'] = 14

loss =loss_his.history['loss']
plt.plot(loss)
plt.xlabel('count')
plt.ylabel('loss')
plt.show()