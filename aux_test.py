import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from time import time
import aux
FIELD_DX = 4500
FIELD_DY = 3000
GOAL_PEN_DX = 1000
GOAL_PEN_DY = 2000
GOAL_DX = FIELD_DX
GOAL_DY = 1000
POLARITY = 1
grid_dens = 100
start_time = time()


with open("aux.cu", "r") as f:
    cuda_code = f.read()
mod = SourceModule(cuda_code)
find_pass_point = mod.get_function("find_best_pass_point")

N = int(FIELD_DX*2/grid_dens*FIELD_DY*2/grid_dens)+1
enemies = np.array([(19.0, 22.0), (30.0, 4.0), (0.0, 10.0), (10.0, 10.0), (20.0, 20.0), (30.0, 4.0)], dtype=np.float32)
kick_point = aux.Point(100,100)
# a_points = np.array([(19.0, 22.0), (30.0, 4.0), (0.0, 10.0), (10.0, 10.0), (20.0, 20.0)], dtype=np.float32)
Point = np.dtype([('x', np.float32), ('y', np.float32)])
enemies_n = np.array([tuple(row) for row in enemies], dtype=Point)
kick_point_n = np.array((kick_point.x, kick_point.y), dtype=Point)
# a = np.array([tuple(row) for row in a_points], dtype=Point)

field_info = np.array([GOAL_DX,GOAL_DY,GOAL_PEN_DX,GOAL_PEN_DY,FIELD_DY,POLARITY],dtype=np.float32)

enemies_gpu = cuda.mem_alloc(enemies_n.nbytes)
kick_point_gpu = cuda.mem_alloc(kick_point_n.nbytes)
field_info_gpu = cuda.mem_alloc(field_info.nbytes)
# a_gpu = cuda.mem_alloc(a.nbytes)
out_gpu = cuda.mem_alloc(2 * np.float32().nbytes)
# field_size_gpu = cuda.mem_alloc(field_size.nbytes)

cuda.memcpy_htod(enemies_gpu, enemies_n)
cuda.memcpy_htod(kick_point_gpu, kick_point_n)
cuda.memcpy_htod(field_info_gpu, field_info)
# cuda.memcpy_htod(a_gpu, a)
# cuda.memcpy_htod(field_size_gpu, field_size)

block_size = 256
grid_size = (N + block_size - 1) // block_size

print(grid_size)
find_pass_point(np.int8(grid_dens),out_gpu, np.int32(N), block=(256, 1, 1), grid=(grid_size, 1))
# find_pass_point(field_info_gpu, enemies_gpu, np.int8(len(enemies)),kick_point_gpu,np.int8(grid_dens),out_gpu, np.int32(N), block=(256, 1, 1), grid=(grid_size, 1))
#extern "C" __global__ void find_best_pass_point(float *field_info,Point *enemies, int en_count, Point kick_point,float grid_dens, float *out, int N)
out = np.zeros(2, dtype=np.float32)
cuda.memcpy_dtoh(out, out_gpu)
end_time = time()
print(end_time-start_time)
print(out)

