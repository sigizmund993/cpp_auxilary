#too long :(
import tensorflow as tf
import numpy as np
from time import time

# Размер массива
N = 10000

# Создаем пустой тензор
array = tf.Variable(tf.zeros((N, N), dtype=tf.float64))

# Функция вычисления
@tf.function
def compute():
    for i in tf.range(N):
        for j in tf.range(N):
            val = tf.math.sin(tf.cast(i, tf.float64) ** 3 + tf.cast(j, tf.float64) ** 3)
            val *= tf.math.exp(tf.math.cos(tf.cast(i * j, tf.float64)))
            array[i, j].assign(val)
    return array

# Запуск
start = time()
with tf.device('/GPU:0'):
    result = compute()
    result_np = result.numpy()
end = time()

print("Время выполнения (TensorFlow, поэлементно):", end - start)
# print(result_np)
