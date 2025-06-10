import solver
from time import time
Vm = solver.bangbang([5, 5], [1, -12], [180, 80], 1, 13)
start = time()
Vm = solver.bangbang([5, 5], [1, -12], [180, 80], 1, 13)
end= time()
print((end-start) * 1e6)