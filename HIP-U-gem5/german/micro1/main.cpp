#include "kernel.h"

#include <unistd.h>
#include <thread>
#include <assert.h>

// This benchmark is going to compute the power of 2 of 1000000000 fp elements

// ROI incorporation
#include <gem5/m5ops.h>

// Params ---------------------------------------------------------------------
struct Params {

    int         device;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    int         size;

    Params(int argc, char **argv) {
        device        = 0;
        n_threads     = 4;
        n_warmup      = 0;
        n_reps        = 1;
        alpha         = 0.1;
        size          = 1000000000;
        int opt;
        while((opt = getopt(argc, argv, "hd:t:w:r:a:s:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': device        = atoi(optarg); break;
            case 't': n_threads     = atoi(optarg); break;
            case 'w': n_warmup      = atoi(optarg); break;
            case 'r': n_reps        = atoi(optarg); break;
            case 'a': alpha         = atof(optarg); break;
            case 's': size          = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        if(alpha == 0.0) {
            // assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            // assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        } else if(alpha == 1.0) {
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else if(alpha > 0.0 && alpha < 1.0) {
            // assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            // assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else {
            //assert((n_gpu_threads > 0 && n_gpu_blocks > 0 || n_threads > 0) && "Invalid # of host + device workers!");
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./micro1 [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    HIP device ID (default=0)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=0)"
                "\n    -r <R>    # of timed repetition iterations (default=1)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of output elements to process on host (default=0.1)"
                "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -s <S>    number of elements to process (default=1000000000)"
                "\n");
    }
};

// Initialize Data ------------------------------------------------------------
void init_data(float* h_in, float* h_out, const Params &p) {
    for(int i=0; i<p.size; i++)
    {
        h_in[i] = i+3.14/12;
        h_out[i] = 0;
    }
}

// Verification process -------------------------------------------------------
void verify(float* h_in, float* h_out, int size) {
    for(int i=0; i<size; i++)
    {
        assert(h_in[i]*h_in[i]==h_out[i]);
    }
}

// Main -----------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    hipError_t  hipStatus;

    // Allocate
    int array_size =  p.size * sizeof(float);
    
    // int n_tasks_i = divceil(p.out_size_i, p.n_gpu_threads);
    // int n_tasks_j = divceil(p.out_size_j, p.n_gpu_threads);
    // int n_tasks   = n_tasks_i * n_tasks_j;

    int n_elements_gpu = p.size * p.alpha;
    int n_elements_cpu = p.size - n_elements_gpu;
    
    // Pointers for the CPU
    float*  h_in  = (float*) malloc(array_size);
    float*  h_out = (float*) malloc(array_size);
    
    // Pointers for the GPU (why needed?) 
    // XYZ * d_in   = h_in;
    // XYZ * d_out  = h_out;
    // Not using worklist for now
    //std::atomic_int * worklist = (std::atomic_int *)malloc(sizeof(std::atomic_int));

    //ALLOC_ERR(h_in, h_out);

    // Initialize
    fprintf(stderr, "Initializing data\n");
    init_data(h_in, h_out, p);
    hipDeviceSynchronize(); // assuming that we need it

    // Call to exitSimLoop to begin ROI
    m5_roi_begin();

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; ++rep) {

        // Launch GPU threads
        // Kernel launch
        // if(p.n_gpu_blocks > 0) {
            hipStatus = call_gpukernel(n_elements_gpu, h_in, h_out);
            if(hipStatus != hipSuccess) { fprintf(stderr, "HIP error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
        // }

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, n_elements_cpu, n_elements_gpu, p.n_threads, h_in, h_out);

        hipDeviceSynchronize();
        main_thread.join();
    }

    // Call to exitSimLoop to end ROI
    m5_roi_end();

    // Verify answer
    verify(h_in, h_out, p.size);
    printf("Verification Passed\n");

    // Free memory
    free(h_in);
    free(h_out);
    //free(worklist);

    if(hipStatus != hipSuccess) { fprintf(stderr, "HIP error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;

    return 0;
}