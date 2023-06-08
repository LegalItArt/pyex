
#5.2 합성곱 신경망 맛보기


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# fashion_mnist = tf.keras.datasets.fashion_mnist
# print(fashion_mnist)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal',   'Shirt',   'Sneaker',  'Bag',   'Ankle boot']
 
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
# plt.show()


#x_train, x_test = x_train / 255.0, x_test / 255.0
#(x_train, y_train), (x_test, y_test) = x_train / 255.0, x_test / 255.0
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0


#모델생성
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
  tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])



model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#학습 훈련
model.fit(x_train, y_train, epochs=10)

#손실과 정확도 출력
loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)


loss_data = []
accuracy_data = []
for i in range(10):
    model.fit(x_train, y_train, epochs=1)
    loss_data.append(model.evaluate(x_test, y_test)[0])
    accuracy_data.append(model.evaluate(x_test, y_test)[1])

import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.figsize'] = (4,3)
plt.rcParams['font.size'] = 12

plt.plot(loss_data) #y
plt.ylim([0.1, 0])
plt.xlabel('count')
plt.ylabel('loss')

plt.show()


# import sys

# # 출력할 문자열
# text = "Hello, world!"

# # UTF-8로 인코딩된 바이트 문자열로 변환
# encoded_text = text.encode('utf-8')

# # 바이트 문자열을 출력
# sys.stdout.buffer.write(encoded_text)
print('훈련테스트 손실도', loss)
print('훈련테스트 정확도', accuracy)
# # 위 코드를 실행하면, "Hello, world!"라는 문자열이 UTF-8로 인코딩되어 출력됩니다.