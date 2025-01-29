#include "kernel.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>

// CPU threads-----------------------------------------------------------------
void run_cpu_threads(int size, int elements_gpu, int n_threads, float* h_in, float* h_out) {

    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < n_threads; k++) {
        cpu_threads.push_back(std::thread([=]() {
            int elements_thread = size/n_threads;
            int offset = elements_gpu + k*elements_thread;
            if(k+1==n_threads)
                elements_thread += size%n_threads;
            for(int i=0; i<elements_thread; i++){
                //printf("Calculating %d element\n", offset+i);
                h_out[offset+i]=h_in[offset+i]*h_in[offset+i];
            }
        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}