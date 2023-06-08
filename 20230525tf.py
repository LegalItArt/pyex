import tensorflow as tf

# 시작 값이 0, 종료 값이 10인 범위에서 간격이 2인 1D 텐서 생성
x = tf.linspace(0., 10., 6)

# 결과 출력
print(x)

# 위 코드에서 `tf.linspace()` 함수의 첫 번째 인자는 시작 값, 두 번째 인자는 종료 값, 세 번째 인자는 배열의 크기입니다. 위 코드에서는 시작 값이 0, 종료 값이 10인 범위에서 간격이 2인 1D 텐서를 생성하도록 지정하였습니다. 결과는 다음과 같습니다.
# tf.Tensor([ 0.  2.  4.  6.  8. 10.], shape=(6,), dtype=float32))
# 위 결과에서 볼 수 있듯이, `tf.linspace()` 함수는 지정한 범위 내에서 일정한 간격의 숫자 배열을 생성하여 1D 텐서로 반환합니다.

a_shape = tf.constant([3,3])
print(a_shape.shape)
print(a_shape.dtype)


# a=tf.add(1,2)
# b=tf.subtract(10,5)
# c=tf.square(3)
# d=tf.reduce_sum(1, 2, 3)
# e=tf.reduce_mean([1,2,3])

# print(a)
# print(b)
# print(c)
# print(d)
# print(e)


n = tf.constant([1,2,3,4,5])
print(n)
print(n.numpy())

print('----------------------------------------')



import numpy as np
np_arr = np.array([[1,2,3], [4,5,6]])
print(np_arr)
tf.matrix = tf.convert_to_tensor(np_arr)
print(tf.matrix)