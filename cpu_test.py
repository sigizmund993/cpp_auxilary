#298.703 = 5min
import numpy as np
from time import time
array = np.zeros((10000,10000),float)
timer = time()
for i in range(len(array)):
    for j in range(len(array[0])):
        # print("yalubluputina yzhe ",time()-timer," let")
        f = np.sin(i ** 3 + j ** 3) * np.e ** np.cos(i * j)
        array[i][j] = f
timer = time()-timer
print(timer)
# print(array)