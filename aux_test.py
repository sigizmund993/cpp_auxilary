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
# GOAL_PEN_DX = 1000
# GOAL_PEN_DY = 2000
# GOAL_DX = FIELD_DX
# GOAL_DY = 1000
# POLARITY = 1
grid_dens = 50

ball_pos = aux.Point(100,100)
enemies_poses = [aux.Point(120,1410),aux.Point(2800,-2210),aux.Point(1223,-93),aux.Point(2939,-229),aux.Point(-294.-1211),aux.Point(-1032,-2420)]

with open("aux.cu", "r") as f:
    cuda_code = f.read()
mod = SourceModule(cuda_code)
find_pass_point = mod.get_function("find_best_pass_point")
# while True:
if True:
    start_time = time()
    N = int(FIELD_DX*2/grid_dens*FIELD_DY*2/grid_dens)+1
    field_poses:list[tuple[float,float]] = [(ball_pos.x,ball_pos.y)]
    for en in enemies_poses:
        field_poses.append((en.x,en.y))
    # print(field_poses)
    field_poses_n = np.array(field_poses, dtype=np.float32)

    # a_points = np.array([(19.0, 22.0), (30.0, 4.0), (0.0, 10.0), (10.0, 10.0), (20.0, 20.0)], dtype=np.float32)
    Point = np.dtype([('x', np.float32), ('y', np.float32)])
    field_poses_n = np.array([tuple(row) for row in field_poses], dtype=Point)
    # kick_point_n = np.array((kick_point.x, kick_point.y), dtype=Point)
    # a = np.array([tuple(row) for row in a_points], dtype=Point)

    # field_info = np.array([GOAL_DX,GOAL_DY,GOAL_PEN_DX,GOAL_PEN_DY,FIELD_DY,POLARITY],dtype=np.float32)

    field_poses_gpu = cuda.mem_alloc(field_poses_n.nbytes)
    # kick_point_gpu = cuda.mem_alloc(kick_point_n.nbytes)
    # field_info_gpu = cuda.mem_alloc(field_info.nbytes)
    # a_gpu = cuda.mem_alloc(a.nbytes)
    out_gpu = cuda.mem_alloc(int(FIELD_DX*2/grid_dens*FIELD_DY*2/grid_dens) * np.float32().nbytes)
    # field_size_gpu = cuda.mem_alloc(field_size.nbytes)

    cuda.memcpy_htod(field_poses_gpu, field_poses_n)
    # cuda.memcpy_htod(kick_point_gpu, kick_point_n)
    # cuda.memcpy_htod(field_info_gpu, field_info)
    # cuda.memcpy_htod(a_gpu, a)
    # cuda.memcpy_htod(field_size_gpu, field_size)

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    print(grid_size)

    find_pass_point(field_poses_gpu,np.int32(len(enemies_poses)),np.int32(grid_dens),out_gpu, np.int32(N), block=(256, 1, 1), grid=(grid_size, 1))
    # find_pass_point(field_info_gpu,field_poses_gpu,np.int32(len(enemies_poses)),np.int32(grid_dens),out_gpu, np.int32(N), block=(256, 1, 1), grid=(grid_size, 1))
    # find_pass_point(field_info_gpu, enemies_gpu, np.int8(len(enemies)),kick_point_gpu,np.int8(grid_dens),out_gpu, np.int32(N), block=(256, 1, 1), grid=(grid_size, 1))
    #extern "C" __global__ void find_best_pass_point(float *field_info,Point *enemies, int en_count, Point kick_point,float grid_dens, float *out, int N)
    out = np.zeros(int(FIELD_DX*2/grid_dens*FIELD_DY*2/grid_dens), dtype=np.float32)
    cuda.memcpy_dtoh(out, out_gpu)
    
    
    print(out)
# x = [out[0]]
# y = [out[1]]
minVal = 1e10
minX = -1
minY = -1

fig, ax = plt.subplots()
for i,x in enumerate(out):
    # c = (x+1)/10
    # ax.plot(grid_dens * (i % int(FIELD_DX*2 / grid_dens))-FIELD_DX, grid_dens * int(i / int(FIELD_DX*2 / grid_dens))-FIELD_DY, marker='o', color=[c,c,c])
    if(x<minVal):
        minVal = x
        minX = grid_dens * (i % int(FIELD_DX*2 / grid_dens))-FIELD_DX
        minY = grid_dens * int(i / int(FIELD_DX*2 / grid_dens))
# for en in enemies_poses:
#     ax.plot(en.x,en.y,marker = 'o',color = 'r')



end_time = time()
print(end_time-start_time)
for i,x in enumerate(out):
    c = (x+7)/15
    ax.plot(grid_dens * (i % int(FIELD_DX*2 / grid_dens))-FIELD_DX, grid_dens * int(i / int(FIELD_DX*2 / grid_dens))-FIELD_DY, marker='o', color=[c,c,c])
for en in enemies_poses:
    ax.plot(en.x,en.y,marker = 'o',color = 'r')    
ax.plot(ball_pos.x,ball_pos.y,marker = 'o',color = 'g')
# Создание прямоугольника: (x, y, ширина, высота)
rect = patches.Rectangle((-FIELD_DX, -FIELD_DY), FIELD_DX*2, FIELD_DY*2, linewidth=2, edgecolor='blue', facecolor='lightblue')

# Добавление прямоугольника на оси
ax.add_patch(rect)
ax.set_aspect('equal', adjustable='box')

# Настройка пределов осей
ax.set_xlim(-4500, 4500)
ax.set_ylim(-3000, 3000)

# Сетка и отображение
ax.grid(True)
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.show()