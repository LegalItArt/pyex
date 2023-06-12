# 문제4
# 문제 3에서 옵티마이저를 Adam 과 RMSprop 을 사용하는 경우 각각의 손실값을 나타내는 test4.py
# 코드를 작성하시오


#epoch = 100으로 할 시 그래프 차이가 선명하지 않아 epoch = 1000 으로 돌립니다.
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])

# 손실최적화 함수 적용
model.compile(loss='mean_squared_error', optimizer='Adam')

xs = [0]
ys = [[0,1,0]]


plt.figure(figsize=(10,5))

# 첫 번째 그래프
plt.subplot(1, 2, 1)

#훈련과정보여주기 생략 설정
loss_his = model.fit(xs, ys, epochs=1000,verbose=0)
loss =loss_his.history['loss']
plt.plot(loss)
plt.title('Adam')
plt.xlabel('count')
plt.ylabel('loss')



# 손실최적화 함수 적용
model.compile(loss='mean_squared_error', optimizer='RMSprop')

xs = [0]
ys = [[0,1,0]]

plt.subplot(1, 2, 2)
#훈련과정 보여주기 자동옵션선택
loss_his = model.fit(xs, ys, epochs=1000,verbose='auto')
loss =loss_his.history['loss']
plt.plot(loss)
plt.title('RMSprop')
plt.xlabel('count')
plt.ylabel('loss')

plt.show()
