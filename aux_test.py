import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from time import time
import aux
import matplotlib.pyplot as plt
import matplotlib.patches as patches
MAX_ACCELERATION = 1500
MAX_SPEED = 1500
with open("aux.cu", "r") as f:
    cuda_code = f.read()
mod = SourceModule(cuda_code)
find_speed = mod.get_function("find_best_bb_speed")
speeds_n_poses_gived = [aux.Point(-1000,-1000),aux.Point(-300,300),aux.Point(312.72,333.23),aux.Point(795.39,606.10),aux.Point(2500,2000),aux.Point(-1000,0)]

# start_time = time()
N = 100
speeds_n_poses:list[tuple[float,float]] = []
for sp in speeds_n_poses_gived:
    speeds_n_poses.append((sp.x,sp.y))
speeds_n_poses_n = np.array(speeds_n_poses, dtype=np.float32)
Point = np.dtype([('x', np.float32), ('y', np.float32)])
speeds_n_poses_n = np.array([tuple(row) for row in speeds_n_poses], dtype=Point)
speeds_n_poses_gpu = cuda.mem_alloc(speeds_n_poses_n.nbytes)
out_gpu = cuda.mem_alloc(2*np.float32().nbytes)
cuda.memcpy_htod(speeds_n_poses_n, speeds_n_poses_gpu)
block_size = 256
grid_size = (N + block_size - 1) // block_size

# extern "C" __global__ void find_best_bb_speed(Point *speeds_n_poses,float *out,int N,float max_acc,float max_speed)
find_speed(speeds_n_poses_gpu,out_gpu,np.int32(N),np.int32(MAX_ACCELERATION),np.int32(MAX_SPEED))
out = np.zeros(2 , dtype=np.float32)
cuda.memcpy_dtoh(out, out_gpu)