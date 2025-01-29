#include "hip/hip_runtime.h"

void run_cpu_threads(int size, int elements_gpu, int n_threads, float* h_in, float* h_out);

hipError_t call_gpukernel(int size, float* d_in, float* d_out);
