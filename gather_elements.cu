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


__device__
void calc_strides(unsigned int* strides, int n_dim, const unsigned int*dims){
    strides[n_dim-1] = 1;

    for(int i = n_dim-2; i >= 0; --i){
        strides[i] = strides[i+1] * dims[i+1];
//        printf("%u\n", strides[i]);
    }
}

__global__
void get_strides(unsigned int* tensor_strides, int n_dim_t, const unsigned int*dims_t,
                 unsigned int* index_strides, int n_dim_i, const unsigned int* dims_i){

    unsigned int i = threadIdx.x;
    if (i == 0)
        calc_strides(tensor_strides, n_dim_t, dims_t);
    else
        calc_strides(index_strides, n_dim_i, dims_i);
}


__device__
unsigned int get_input_index(
        const unsigned int* indices,
        const unsigned int* strides,
        const unsigned int axis,
        const unsigned int axis_replace_idx,
        unsigned int n_dim){

    unsigned int input_index = 0;

//    printf("axis=[%u],axis_replace_idx=[%u]\n", axis, axis_replace_idx);

    for(unsigned int i = 0; i < n_dim; ++i){

        if (i == axis){
//            printf("dim=[%u],indices=[%u], stride=[%u]\n",i, axis_replace_idx, strides[i]);
            input_index += (axis_replace_idx * strides[i]);
        }
        else{
//            printf("dim=[%u],indices=[%u], stride=[%u]\n",i, indices[i], strides[i]);
            input_index += (indices[i] * strides[i]);
        }

    }

    return input_index;
}

__device__
void calc_indices(
        unsigned int idx,
        unsigned int* indices,
        const unsigned int* strides,
        const unsigned int* dims,
        const unsigned int n_dim){

    for(unsigned int i = 0; i != n_dim; ++i){
        indices[i] = (idx / strides[i]) % dims[i];
    }
}


__global__
void gather_elements_kernel(
        const float* input, const unsigned int* index, float* output,
        const unsigned int axis, unsigned int n_dim,
        const unsigned int* index_dims,
        const unsigned int* tensor_strides, const unsigned int* index_strides,
        unsigned int* indices,
        unsigned int max_indices_len){

    unsigned int out_idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

//    printf("index[%u]\n", index[out_idx]);

    if (out_idx >= max_indices_len){
        return;
    }

    unsigned int* p_indices = indices + n_dim * out_idx;

    // calculate indices
    calc_indices(out_idx, p_indices, index_strides, index_dims, n_dim);

//    for(unsigned int i = 0; i != n_dim; ++i){
//        printf("dim[%u], loc[%u]\n", i, p_indices[i]);
//    }

    unsigned int in_idx = get_input_index(p_indices, tensor_strides, axis, index[out_idx], n_dim);

    output[out_idx] = input[in_idx];

//    printf("in_dix, in_value: %u, %f\n", in_idx, input[in_idx]);

//    printf("output: %f\n", output[out_idx]);
}

/*
 * workspace
 * :n_dim                                       tensor_dims
 * n_dim: n_dim * 2                             index_dims
 * n_dim*2 : n_dim * 3                          index_strides
 * n_dim*3 : n_dim * 4                          input_tensor_strides
 * n_dim*4 : n_dim * 4 + n_dim * size(index)    input_index location x,y,c ...
 */

void gather_elements(
        const void* const* input,
        void** output,
        unsigned int axis,
        int n_dim,
        const unsigned int* tensor_dims,
        const unsigned int* index_dims,
        unsigned int* workspace,
        cudaStream_t stream){

    unsigned int data_size = 1;
    for(int i = 0; i != n_dim; ++i){
        data_size *= index_dims[i];
    }
    unsigned int blocks = KERNEL_BLOCK;

    if (KERNEL_BLOCK > data_size){
        blocks = data_size;
    }

    auto tensor_dims_d = workspace;
    auto index_dims_d = tensor_dims_d + n_dim;

    unsigned int* index_strides_d = index_dims_d + n_dim;
    unsigned int* input_tensor_strides_d = index_strides_d + n_dim;
    unsigned int* indices_d = input_tensor_strides_d + n_dim;

    gpuErrchk(cudaMemcpyAsync(tensor_dims_d, tensor_dims, sizeof(unsigned int) * n_dim, cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaMemcpyAsync(index_dims_d, index_dims, sizeof(unsigned int) * n_dim, cudaMemcpyHostToDevice, stream));
    //    printf("%d", cudaGetLastError());

    get_strides<<<1, 2, 0>>>(input_tensor_strides_d, n_dim, tensor_dims_d, index_strides_d, n_dim, index_dims_d);

    cudaStreamSynchronize(stream);

    gather_elements_kernel<<<cuda_gridsize(data_size, blocks), blocks, 0, stream>>>(
            (float*)input[0], (unsigned int*)input[1], (float*)output[0], axis, n_dim,
            index_dims_d, input_tensor_strides_d, index_strides_d, indices_d, data_size);

    cudaStreamSynchronize(stream);
}