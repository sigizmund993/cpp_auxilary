import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from time import time
import aux
import matplotlib.pyplot as plt
import matplotlib.patches as patches
FIELD_DX = 4500
FIELD_DY = 3000
grid_dens = 10
ball_pos = aux.Point(100,100)
enemies_poses = [aux.Point(120,1410),aux.Point(2800,-2210),aux.Point(1223,-93),aux.Point(2939,-229),aux.Point(-294.-1211),aux.Point(-1032,-2420)]
out_size = int(FIELD_DX*2/grid_dens*FIELD_DY*2/grid_dens)
with open("aux.cu", "r") as f:
    cuda_code = f.read()
mod = SourceModule(cuda_code)
find_pass_point = mod.get_function("find_best_pass_point")


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
minV = 1e10
minId = -1
for i in range(grid_size):
    if(out[i]<minV):
        minV = out[i]
        minId = out[i*2]
end_time = time()
print(minV)
minPos = aux.Point(grid_dens * (minId % int(FIELD_DX*2 / grid_dens))-FIELD_DX,grid_dens * int(minId / int(FIELD_DX*2 / grid_dens))-FIELD_DY)
print(minPos)
print(end_time-start_time)

minV = 1e10
minId = -1
fig, ax = plt.subplots()

for i,x in enumerate(out):
    if(i>grid_size*2):
        if(x<minV):
            minV = x
            minId = i
        c = (-x+7)/10
        if(grid_dens>=50):
            ax.plot(grid_dens * (i % int(FIELD_DX*2 / grid_dens))-FIELD_DX, grid_dens * int(i / int(FIELD_DX*2 / grid_dens))-FIELD_DY, marker='o', color=[1-c,c,0])
minCpuPos = aux.Point(grid_dens * (minId % int(FIELD_DX*2 / grid_dens))-FIELD_DX,grid_dens * int(minId / int(FIELD_DX*2 / grid_dens))-FIELD_DY)
for en in enemies_poses:
    ax.plot(en.x,en.y,marker = 'o',color = 'r')    
ax.plot(ball_pos.x,ball_pos.y,marker = 'o',color = 'g')
ax.plot(1780,1510,marker = 'o',color = 'b')
# ax.plot(minCpuPos.x,minCpuPos.y,marker = 'o',color = 'lightblue')
# ax.plot(1780,1510,marker = 'o',color = 'lightblue')
rect = patches.Rectangle((-FIELD_DX, -FIELD_DY), FIELD_DX*2, FIELD_DY*2, linewidth=2, edgecolor='blue', facecolor='none')
ax.add_patch(rect)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-4500, 4500)
ax.set_ylim(-3000, 3000)
ax.grid(True)
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.show()
"""
grid_dens, mm | time, ms
    100       |   ~1
     10       |   ~3

"""
