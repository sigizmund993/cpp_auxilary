#1.5sec
import tensorflow as tf
import numpy as np
from time import time

# Размер массива
N = 10000

# Создание индексов
i = tf.range(N, dtype=tf.float64)
j = tf.range(N, dtype=tf.float64)
ii, jj = tf.meshgrid(i, j, indexing='ij')
print(ii,jj)
@tf.function
def compute(ii, jj):
    return tf.math.sin(ii ** 3 + jj ** 3) * tf.math.exp(tf.math.cos(ii * jj))

# Запуск на GPU
start = time()
with tf.device('/GPU:0'):
    result = compute(ii, jj)
    result_np = result.numpy()
end = time()

print("Время выполнения (TensorFlow):", end - start)
# print(result_np)
