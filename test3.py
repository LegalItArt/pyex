# [문제3] 
# 입력값이 1개 이고 출력값이 3개인 간단한 신경망을 구성하고

# 손실함수='mean_squared_error'
# 옵티마이즈='sgd' 를 사용하여 훈련시킬 때
 
# # 입력값 [0], 출력값 [[0, 1, 0]] 에 대한 손실값을 출력하는 test3.py 코드를 작성하시오

from tensorflow import keras
import matplotlib.pyplot as plt

model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])

# 손실최적화 함수 적용
model.compile(loss='mean_squared_error', optimizer='sgd')

xs = [0]
ys = [[0,1,0]]

loss_his = model.fit(xs, ys, epochs=100,verbose=0)
loss =loss_his.history['loss']

#손실함수그래프 구현 코드
plt.plot(loss)
plt.xlabel('count')
plt.ylabel('loss')
plt.show()
