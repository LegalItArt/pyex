import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


# 스칼라 확인
scalar = tf.constant(1)

# 벡터 확인
vector = tf.constant([1, 2, 3, 4, 5])

# 메트릭스 확인
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 텐서 확인
tensor = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
                                          )

print(tf.rank(scalar))  # 출력: 0
print(tf.rank(vector))  # 출력: 1
print(tf.rank(matrix))  # 출력: 2
print(tf.rank(tensor))  # 출력: 3


a = tf.zeros(1)
b = tf.zeros([2])
c = tf.zeros([2,3])

print(a)
print(b)
print(c)

a1 = tf.ones(1)
b1 = tf.ones([2])
c1 = tf.ones([2,3])

print(a1)
print(b1)
print(c1)

a2 = tf.range(1)
b2 = tf.range(0,3)
c2 = tf.range(1,5,2)
d2 = tf.range(1,5,3)

print(a2)
print(b2)
print(c2)
print(d2)



a_shape = tf.constant([3,3])
print(a_shape.shape)
print(a_shape.dtype)


