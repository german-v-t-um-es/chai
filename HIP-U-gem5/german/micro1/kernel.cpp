#include "kernel.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>

// CPU threads-----------------------------------------------------------------
void run_cpu_threads(int size, int elements_gpu, int n_threads, float* h_in, float* h_out) {

    int elements_thread = size/n_threads;
    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < n_threads; k++) {
        cpu_threads.push_back(std::thread([=]() {
            if(k+1==n_threads)
                elements_thread += size%n_threads;
            for(int i=elements_thread; i<elements_thread; i++){
                h_out[elements_gpu+n_threads*i]=h_in[elements_gpu+n_threads*i]*h_in[elements_gpu+n_threads*i];
            }
        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}