import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
import tensorflow  as tf
from keras.callbacks import EarlyStopping


#랜덤하게 가중치 설정
# 
tf.random.set_seed(0)

# 모델을 만들기 전에 뉴런층을 정의 입력층을 만들겠다
input_layers = keras.layers.InputLayer(input_shape=(3,))
hidden_layers = tf.keras.layers.Dense(units=4, activation ="relu")
output_layers = keras.layers.Dense(units=2, activation='softmax')

#모델생성
model = tf.keras.Sequential([
    input_layers,
    hidden_layers,
    output_layers])

#모델컴파일
model.compile(loss="MSE", optimizer='Adam')

print("뉴런층 이름")
print(input_layers.name)
print(hidden_layers.name)
print(output_layers.name)
print("---------------------------------------------")

print("=뉴런층 속성 이름")
print(input_layers.name, input_layers.dtype  )
print(hidden_layers.name, hidden_layers.dtype)
print(output_layers.name, output_layers.dtype)
print("---------------------------------------------")

print("뉴런층 입력 rank")
print(input_layers.input)
print(hidden_layers.input)
print(output_layers.input)
print("---------------------------------------------")

print("뉴런층 출력 rank")
print(input_layers.output)
print(hidden_layers.output)
print(output_layers.output)
print("---------------------------------------------")

print(" 활성화 함수")
print(input_layers.activity_regularizer)
print(hidden_layers.activation)
print(output_layers.activation)
print("---------------------------------------------")

print(" 가중치")
print(input_layers.weights)
print(hidden_layers.weights)
print(output_layers.weights)
print("---------------------------------------------")

print("뉴런층 입력 출력 rank, shape")
print(input_layers.input,input_layers.output, input_layers.output.shape)
print(hidden_layers.input,hidden_layers.output,hidden_layers.output.shape)
print(output_layers.input,output_layers.output, output_layers.output.shape)
print("---------------------------------------------")

print("가중치, 뉴런층 입력 출력 rank, shape")
print(input_layers.weights, input_layers.input,input_layers.output, input_layers.output.shape)
print(hidden_layers.weights, hidden_layers.input,hidden_layers.output,hidden_layers.output.shape)
print(output_layers.weights, output_layers.input,output_layers.output, output_layers.output.shape)
print("---------------------------------------------")


