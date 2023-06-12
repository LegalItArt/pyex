# 문제5
# 그림과 같은 신경망을 구현하는 코드를 작성하는 test5.py 코드를 작성하시오
# 단 은닉층에서는 relu를 출력층에서는 softmax를 활성화 함수로 사용하시오
# [그림은 평가지에서 확인 바랍니다]

import tensorflow as tf
from tensorflow import keras


tf.random.set_seed(0)

# 모델을 만들기 전에 입력3개층 히든 4개층, 출력2개층 뉴런층을 정의함 
input_layers = keras.layers.InputLayer(input_shape=(3,))
hidden_layers = tf.keras.layers.Dense(units=4, activation ="relu")
output_layers = keras.layers.Dense(units=2, activation='softmax')

#모델생성
model = tf.keras.Sequential([
    input_layers,
    hidden_layers,
    output_layers])

print("shape")
print(input_layers.output.shape)
print(hidden_layers.output.shape)
print(output_layers.output.shape)
print("---------------------------------------------")

print("가중치, 뉴런층 입력rank  출력 rank, shape")
print(input_layers.weights, input_layers.input,input_layers.output, input_layers.output.shape)
print(hidden_layers.weights, hidden_layers.input,hidden_layers.output,hidden_layers.output.shape)
print(output_layers.weights, output_layers.input,output_layers.output, output_layers.output.shape)
print("---------------------------------------------")