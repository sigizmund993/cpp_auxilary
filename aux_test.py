import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from time import time
import aux
FIELD_DX = 4500
FIELD_DY = 3000
# GOAL_PEN_DX = 1000
# GOAL_PEN_DY = 2000
# GOAL_DX = FIELD_DX
# GOAL_DY = 1000
# POLARITY = 1
grid_dens = 50
out_size = int(FIELD_DX*2/grid_dens*FIELD_DY*2/grid_dens)
print(out_size)
ball_pos = aux.Point(100,100)
enemies_poses = [aux.Point(120,1410),aux.Point(2800,-2210),aux.Point(1223,-93),aux.Point(2939,-229),aux.Point(-294.3211),aux.Point(-1032,-3420)]

with open("aux.cu", "r") as f:
    cuda_code = f.read()
mod = SourceModule(cuda_code)
find_pass_point = mod.get_function("find_best_pass_point")
if True:
    start_time = time()
    N = int(FIELD_DX*2/grid_dens*FIELD_DY*2/grid_dens)+1
    field_poses:list[tuple[float,float]] = [(ball_pos.x,ball_pos.y)]
    for en in enemies_poses:
        field_poses.append((en.x,en.y))
    field_poses_n = np.array(field_poses, dtype=np.float32)
    Point = np.dtype([('x', np.float32), ('y', np.float32)])
    field_poses_n = np.array([tuple(row) for row in field_poses], dtype=Point)
    field_poses_gpu = cuda.mem_alloc(field_poses_n.nbytes)
    out_gpu = cuda.mem_alloc(out_size * np.float32().nbytes)
    cuda.memcpy_htod(field_poses_gpu, field_poses_n)
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    # print(grid_size)
    find_pass_point(field_poses_gpu,np.int32(len(enemies_poses)),np.int32(grid_dens),out_gpu, np.int32(N), block=(256, 1, 1), grid=(grid_size, 1))
    out = np.zeros(out_size , dtype=np.float32)
    cuda.memcpy_dtoh(out, out_gpu)
    
    # print(out)
    minV = 1e10
    minId = -1
    for i in range(grid_size):
        if(out[i]<minV):
            minV = out[i]
            minId = out[i*2]
    end_time = time()
    print(minV)
    minPos = aux.Point(grid_dens * (minId % int(FIELD_DX*2 / grid_dens))-FIELD_DX,grid_dens * int(minId / int(FIELD_DX*2 / grid_dens))-FIELD_DY)
    print(minId)
    print(minPos)
    print(end_time-start_time)
