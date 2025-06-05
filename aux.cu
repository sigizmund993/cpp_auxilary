#include <stdio.h>
#include <math.h>
#define GRID_SIZE 2110
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
__device__ float globVals[GRID_SIZE];
__device__ int globX[GRID_SIZE];
__device__ int globY[GRID_SIZE];
extern "C" __global__ void find_best_pass_point(Point *field, float *field_size, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Выходной массив для каждого потока
    
    for(int i = 0;i<GRID_SIZE;i++){
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
        // Point p2(0,0);
        curVal = 0.0f;
        // for (int i = 0; i <= 7; i++) {
        //     curVal += dist(cur_pos, field[i]);
        // }
        curVal = sin(pow((double)cur_pos.x, 3.0) + pow((double)cur_pos.y, 3.0)) * exp(cos((double)(cur_pos.x * cur_pos.y)));
        curX = cur_pos.x;
        curY = cur_pos.y;
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
        for (int i = 0; i < GRID_SIZE; i++) {
            if (globVals[i] < minV) {
                minV = globVals[i];
                minX = globX[i];
                minY = globY[i];
            }
        }
        // for(int i = 0;i<GRID_SIZE;i++)
        // {
        //     printf("%f at %i,%i\n",globVals[i],globX[i],globY[i]);
        // }
        printf("end min Val: %f at %i, %i\n",minV,minX,minY);
        out[0] = minX;
        out[1] = minY;
    }
}
