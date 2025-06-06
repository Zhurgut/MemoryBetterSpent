

#include <torch/extension.h>

#include "helpers.h"

#define BLOCKSIZE 32

// A = (M, N)
// B = (N, P)
// C = (M, S)
// X = (M, P)
// Y = (N, S)


__global__ void kernel_coalesce(
        const float* A, int a_pitch, const float* B, const float* C, 
        float* X, int x_pitch, float* Y,
        int M, int N, int P, int S) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x % BLOCKSIZE; // increase with threadIdx.x
    int row = blockIdx.y * blockDim.y + threadIdx.x / BLOCKSIZE; // same for all in warp
    

    // Compute X = A * B
    if (row < M && col < P) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += getindex(A, a_pitch, row, i) * getindex(B, P, i, col);
        }
        getindex(X, x_pitch, row, col) = sum;
    }

    // Compute Y = Aᵗ * C
    if (row < N && col < S) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += getindex_T(A, a_pitch, row, i) * getindex(C, S, i, col);
        }
        getindex(Y, S, row, col) = sum;
    }
}


void coalesce(torch::Tensor A, int64_t a_pitch, torch::Tensor B, torch::Tensor C, torch::Tensor X, int64_t x_pitch, torch::Tensor Y) {
    int64_t M = A.size(0);
    int64_t N = A.size(1);
    int64_t P = B.size(1);
    int64_t S = C.size(1);

    const dim3 threads_per_block(BLOCKSIZE * BLOCKSIZE);
    const dim3 nr_blocks(CEIL_DIV(MAX(P, S), BLOCKSIZE), CEIL_DIV(MAX(M, N), BLOCKSIZE));

    kernel_base<<<nr_blocks, threads_per_block>>>(
        A.data_ptr<float>(), a_pitch, 
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        X.data_ptr<float>(), x_pitch,
        Y.data_ptr<float>(),
        M, N, P, S);
}



TORCH_LIBRARY(my_cuda_kernels, m) {
    // m.def("base(Tensor a, Tensor b, Tensor c, Tensor(x!) x, Tensor(y!) y, int bias_grad, Tensor(z!) g) -> ()");
    m.def("coalesce(Tensor a, int a_pitch, Tensor b, Tensor c, Tensor(x!) x, int x_pitch, Tensor(y!) y) -> ()");
}


TORCH_LIBRARY_IMPL(my_cuda_kernels, CUDA, m) {
    m.impl("coalesce", &coalesce);
}




