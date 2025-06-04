import numpy as np
from time import time
start_time = time()
FIELD_DX = 4500
FIELD_DY = 3000
grid_dens = 5
a_points = np.array([(19.0, 22.0), (30.0, 4.0), (0.0, 10.0), (10.0, 10.0), (20.0, 20.0)], dtype=np.float32)
min = 10000000
minX = -1
minY = -1
for i in range(int(FIELD_DX/grid_dens)):
    for j in range(int(FIELD_DY/grid_dens)):
        res = 0
        for x in a_points:
            res+=np.sqrt((x[0]-i*grid_dens)**2+(x[1]-j*grid_dens))
        if(res<min):
            min = res
            minX = i*grid_dens
            minY = j*grid_dens
end_time = time()
print(min,minX,minY)
print(end_time-start_time)