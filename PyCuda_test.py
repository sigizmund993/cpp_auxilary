#0.84
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
from time import time

# Размер массива
N = 10000

# CUDA-ядро
mod = SourceModule("""
__global__ void compute(double *array, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        double val = sin(pow((double)i, 3.0) + pow((double)j, 3.0)) * exp(cos((double)(i * j)));
        array[i * N + j] = val;
    }
}
""")

# Подготовка массива
array = np.zeros((N, N), dtype=np.float64)
array_gpu = cuda.mem_alloc(array.nbytes)

# Получаем ядро
func = mod.get_function("compute")

# Конфигурация сетки
block_size = (16, 16, 1)
grid_size = (N // 16, N // 16, 1)

# Запуск
start = time()
func(array_gpu, np.int32(N), block=block_size, grid=grid_size)
cuda.memcpy_dtoh(array, array_gpu)
end = time()

print("Время выполнения (PyCUDA):", end - start)
print(array)
