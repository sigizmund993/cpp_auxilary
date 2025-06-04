import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from time import time
FIELD_DX = 4500
FIELD_DY = 3000
grid_dens = 100
start_time = time()
# Читаем обновленный CUDA-код
with open("aux.cu", "r") as f:
    cuda_code = f.read()

# Компилируем
mod = SourceModule(cuda_code)

# Получаем функцию
find_pass_point = mod.get_function("find_best_pass_point")

# Параметры
N = int(FIELD_DX*2/grid_dens*FIELD_DY*2/grid_dens)+1
a_points = np.array([(19.0, 22.0), (30.0, 4.0), (0.0, 10.0), (10.0, 10.0), (20.0, 20.0)], dtype=np.float32)
res = 0
for p in a_points:
    res+=np.sqrt(p[0]**2+p[1]**2)
# Объединяем пары в структуру Point
Point = np.dtype([('x', np.float32), ('y', np.float32)])
a = np.array([tuple(row) for row in a_points], dtype=Point)
field_size = np.array([FIELD_DX*2,FIELD_DY*2],dtype=np.float32)
# Выделяем память на GPU
a_gpu = cuda.mem_alloc(a.nbytes)
out_gpu = cuda.mem_alloc(2 * np.float32().nbytes)
field_size_gpu = cuda.mem_alloc(field_size.nbytes)
# Копируем на GPU
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(field_size_gpu, field_size)
# Вызываем ядро
block_size = 256
grid_size = (N + block_size - 1) // block_size
print(grid_size)
find_pass_point(a_gpu,field_size_gpu, out_gpu, np.int32(N), block=(256, 1, 1), grid=(grid_size, 1))

# Читаем результат
out = np.zeros(2, dtype=np.float32)
cuda.memcpy_dtoh(out, out_gpu)
end_time = time()
print(end_time-start_time)
print(out)

