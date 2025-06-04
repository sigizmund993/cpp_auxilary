#include <stdio.h>
#include <math.h>

struct Point {
    float x, y;
    __host__ __device__ Point() : x(0), y(0) {}
    __host__ __device__ Point(float x_, float y_) : x(x_), y(y_) {}
    __host__ __device__ Point operator-(const Point& b) const { return Point(x - b.x, y - b.y); }
    __host__ __device__ float mag() const { return sqrtf(x * x + y * y); }
};

__host__ __device__ float dist(Point p1, Point p2) {
    return (p2 - p1).mag();
}
// Кернел для нахождения локальных минимумов в каждом блоке
extern "C" __global__ void find_block_min(Point *field, float *field_size, float *block_mins, int *block_locs, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedVals[256];
    __shared__ int sharedX[256];
    __shared__ int sharedY[256];

    float curVal = 1e10f;
    int curX = -1, curY = -1;

    if (idx < N) {
        int grid_dens = 100;
        Point cur_pos(grid_dens * (idx % int(field_size[0] / grid_dens)),
                       grid_dens * (idx / int(field_size[0] / grid_dens)));

        curVal = 0.0f;
        for (int i = 0; i <= 7; i++) {
            curVal += dist(cur_pos, field[i]);
        }
        curX = (int)cur_pos.x;
        curY = (int)cur_pos.y;
    }

    sharedVals[threadIdx.x] = curVal;
    sharedX[threadIdx.x] = curX;
    sharedY[threadIdx.x] = curY;

    __syncthreads();

    // Поток 0 ищет минимум в блоке
    if (threadIdx.x == 0) {
        float minV = 1e10f;
        int minX = -1, minY = -1;
        for (int i = 0; i < blockDim.x; i++) {
            if (sharedVals[i] < minV) {
                minV = sharedVals[i];
                minX = sharedX[i];
                minY = sharedY[i];
            }
        }
        // Сохраняем локальный минимум блока в глобальной памяти
        block_mins[blockIdx.x] = minV;
        block_locs[blockIdx.x * 2] = minX;
        block_locs[blockIdx.x * 2 + 1] = minY;
    }
}

// Кернел для нахождения глобального минимума из локальных минимумов
extern "C" __global__ void find_global_min(float *block_mins, int *block_locs, float *out, int numBlocks) {
    float minV = 1e10f;
    int minX = -1, minY = -1;

    for (int i = 0; i < numBlocks; i++) {
        if (block_mins[i] < minV) {
            minV = block_mins[i];
            minX = block_locs[i * 2];
            minY = block_locs[i * 2 + 1];
        }
    }

    // Результат
    out[0] = minX;
    out[1] = minY;
    printf("Global minVal=%f at (%d, %d)\n", minV, minX, minY);
}

extern "C" __global__ void find_best_pass_point(Point *field, float *field_size, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Выходной массив для каждого потока
    float globVals[8438];
    int globX[8438];
    int globY[8438];
    for(int i = 0;i<8438;i++){
        globVals[i] = 1e10f;
    }
    __shared__ float localVals[256];  // 256 - размер блока
    __shared__ int localX[256];
    __shared__ int localY[256];
    
    float curVal = 1e10f;
    int curX = -1, curY = -1;

    if (idx < N) {
        int grid_dens = 10;
        Point cur_pos(grid_dens * (idx % int(field_size[0] / grid_dens)),
                       grid_dens * int(idx / int(field_size[0] / grid_dens)));
        curVal = 0.0f;
        for (int i = 0; i <= 7; i++) {
            curVal += dist(cur_pos, field[i]);
        }
        curX = (int)cur_pos.x;
        curY = (int)cur_pos.y;
    }
    // printf("%f %i %i\n",curVal,curX,curY);
    // Записываем свои значения в shared memory
    localVals[threadIdx.x] = curVal;
    localX[threadIdx.x] = curX;
    localY[threadIdx.x] = curY;

    __syncthreads();

    // Поток 0 внутри блока находит минимум
    if (threadIdx.x == 0) {
        float minV = 1e10f;
        int minX = -1, minY = -1;
        for (int i = 0; i < blockDim.x; i++) {
            if (localVals[i] < minV) {
                minV = localVals[i];
                minX = localX[i];
                minY = localY[i];
            }
        }
        // Записываем минимум этого блока в выходной массив
        globVals[blockIdx.x] = minV;
        globX[blockIdx.x] = minX;
        globY[blockIdx.x] = minY;
        // printf("Found minVal=%f at (%d, %d)\n", minV, minX, minY);
    }
    __syncthreads();
    if(idx == 0)
    {
        float minV = 1e10f;
        int minX = -1, minY = -1;
        for (int i = 0; i < 8438; i++) {
            if (globVals[i] < minV) {
                minV = globVals[i];
                minX = globX[i];
                minY = globY[i];
            }
        }
        printf("end min Val: %f at %i, %i\n",minV,minX,minY);
        out[0] = minX;
        out[1] = minY;
    }
}
