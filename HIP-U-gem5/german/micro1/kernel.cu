#include "hip/hip_runtime.h"
#include "kernel.h"

__global__ void kernel(int size, float* d_in, float* d_out){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

hipError_t call_gpukernel(int size, float* d_in, float* d_out){
    dim3 dimBlock(16, 16);
    int threads_per_block = dimBlock.x * dimBlock.y;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    dim3 dimGrid(blocks, 1);

    hipLaunchKernelGGL(kernel, dim3(dimGrid), dim3(dimBlock), 0, 0, size, d_in, d_out);
    hipError_t err = hipGetLastError();
    return err;
}