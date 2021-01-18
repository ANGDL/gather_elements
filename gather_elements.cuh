//
// Created by ang on 2021/1/11.
//

#ifndef GATHER_ELEMENTS_GATHER_ELEMENTS_CUH
#define GATHER_ELEMENTS_GATHER_ELEMENTS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void gather_elements(
        const void *const *input,
        void *const *output,
        unsigned int axis,
        int n_dim,
        const int *tensor_dims,
        const int *index_dims,
        void *workspace,
        cudaStream_t stream);

#endif //GATHER_ELEMENTS_GATHER_ELEMENTS_CUH
