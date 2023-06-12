# [문제1] 
# 텐서플로우를 설치하고
 
# [1, 2, 3] 값을 가지는 vector
# [[1, 2, 3], [4, 5, 6], [7, 8, 9]] 값을 가지는 matrix
# [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
# [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
# [[1, 2, 3], [4, 5, 6], [7, 8, 9]]] 값을 가지는 tensor 변수를 선언하고
 
# 값을 출력하는 test1.py 코드를 작성하시오


import tensorflow as tf

# Vector
vector = tf.constant([1, 2, 3])

# Matrix
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Tensor
tensor = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                 [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                 [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

print("----------vector---------")
print(vector)
print("----------matrix----------")
print(matrix)
print("----------tensor-----------")
print(tensor)