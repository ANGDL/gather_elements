//
// Created by ang on 2021/1/11.
//
#include <cassert>
#include <cmath>

# include "gather_elements.cuh"

#define KERNEL_BLOCK 1024

// cuda_gridsize
static
dim3 cuda_gridsize(unsigned int n, unsigned int blocks) {
    unsigned int k = (n - 1) / blocks + 1;
    unsigned int x = k;
    unsigned int y = 1;
    if (x > 65535) {
        x = static_cast<unsigned int>(ceil(sqrt((float) k)));
        y = (n - 1) / (x * blocks) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*KERNEL_BLOCK);
    return d;
}


__global__
void gather_elements_kernel(
        const float* input, const unsigned long* index, float* output, const unsigned int axis,
        unsigned int in_c, unsigned int in_h, unsigned int in_w,
        unsigned int idx_c, unsigned int idx_h, unsigned int idx_w){

    unsigned int out_idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (out_idx >= idx_c * idx_h * idx_w){
        return;
    }

    unsigned int i = out_idx / (idx_w * idx_h);
    unsigned int j = (out_idx - i * idx_w * idx_h) / idx_w;
    unsigned int k = out_idx - (i * idx_w * idx_h) - (j * idx_w);

//    printf("%u, %u %u %u\n", out_idx, i, j, k);

    unsigned int in_idx;

    if (0 == axis) {
        in_idx = index[out_idx] * in_h * in_w + j * in_w + k;
    }
    else if (1 == axis) {
        in_idx = i * in_h * in_w + index[out_idx] * in_w + k;
    }
    else{
        in_idx = i * in_h * in_w + j * in_w + index[out_idx];
    }

//    assert(out_idx < in_c * in_h * in_w);

//    printf("%u: %f\n", in_idx, input[in_idx]);

    output[out_idx] = input[in_idx];

//    printf("output: %f\n", output[out_idx]);
}


void gather_elements(
        const void* const* input,
        void** output,
        unsigned long axis,
        unsigned int in_c, unsigned int in_h, unsigned int in_w,
        unsigned int idx_c, unsigned int idx_h, unsigned int idx_w,
        cudaStream_t stream){

    unsigned int data_size = idx_c * idx_h * idx_w;
    unsigned int blocks = KERNEL_BLOCK;

    if (KERNEL_BLOCK > data_size){
        blocks = data_size;
    }
    gather_elements_kernel<<<cuda_gridsize(data_size, blocks), blocks, 0, stream>>>(
            (float*)input[0], (unsigned long*)input[1], (float*)output[0], axis,
            in_c, in_h, in_w, idx_c, idx_h, idx_w);

    cudaDeviceSynchronize();
}