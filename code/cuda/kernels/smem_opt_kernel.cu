

#include <torch/extension.h>

#include "helpers.h"

#define BLOCKSIZE 32

// A = (M, N)
// B = (N, P)
// C = (M, S)
// X = (M, P)
// Y = (N, S)


__global__ void kernel_smem_opt(
        const float* A, int a_pitch, const float* B, const float* C, 
        float* X, int x_pitch, float* Y,
        int M, int N, int P, int S) {
    
    __shared__ float m1[BLOCKSIZE*BLOCKSIZE];
    __shared__ float m2[BLOCKSIZE*BLOCKSIZE];

    int bk_c = threadIdx.x % BLOCKSIZE;
    int bk_r = threadIdx.x / BLOCKSIZE;
    
    int out_c = blockIdx.x * BLOCKSIZE + bk_c;
    int out_r = blockIdx.y * BLOCKSIZE + bk_r;
    
    float sum = 0.0f;

    for (int bk = 0; bk < N/BLOCKSIZE; bk++) {

        if (out_r < M) {
            getindex(m1, BLOCKSIZE, bk_r, bk_c) = getindex(A, a_pitch, out_r, bk*BLOCKSIZE + bk_c);
        }
        if (out_c < P) {
            getindex(m2, BLOCKSIZE, bk_r, bk_c) = getindex(B, P, bk*BLOCKSIZE + bk_r, out_c);
        }

        __syncthreads();

        if (out_r < M && out_c < P) {
            for (int i = 0; i < BLOCKSIZE; i++) {
                sum += getindex(m1, BLOCKSIZE, bk_r, i) * getindex(m2, BLOCKSIZE, i, bk_c);
            }
        }

        __syncthreads();
        
    }

    if (out_r < M && N/BLOCKSIZE*BLOCKSIZE + bk_c < N) {
        getindex(m1, BLOCKSIZE, bk_r, bk_c) = getindex(A, a_pitch, out_r, N/BLOCKSIZE*BLOCKSIZE + bk_c);
    }

    if (out_c < P && N/BLOCKSIZE*BLOCKSIZE + bk_r < N) {
        getindex(m2, BLOCKSIZE, bk_r, bk_c) = getindex(B, P, N/BLOCKSIZE*BLOCKSIZE + bk_r, out_c);
    }

    __syncthreads();

    if (out_r < M && out_c < P) {
        for (int i = 0; i < N % BLOCKSIZE; i++) {
            sum += getindex(m1, BLOCKSIZE, bk_r, i) * getindex(m2, BLOCKSIZE, i, bk_c);
        }
    }



    if (out_r < M && out_c < P) {
        getindex(X, x_pitch, out_r, out_c) = sum;
    }


    __syncthreads(); // make sure all threads are done reading the values for X, then can move on
    sum = 0.0f;

    for (int bk = 0; bk < M/BLOCKSIZE; bk++) {

        int out_rT = blockIdx.y * BLOCKSIZE + bk_c; // still use .y, but with bk_c
        if (out_rT < N) {
            getindex(m1, BLOCKSIZE, bk_c, bk_r) = getindex_T(A, a_pitch, out_rT, bk*BLOCKSIZE + bk_r); // slightly better
        }
        if (out_c < S) {
            getindex(m2, BLOCKSIZE, bk_r, bk_c) = getindex(C, S, bk*BLOCKSIZE + bk_r, out_c);
        }

        __syncthreads();

        if (out_r < N && out_c < S) {
            for (int i = 0; i < BLOCKSIZE; i++) {
                sum += getindex(m1, BLOCKSIZE, bk_r, i) * getindex(m2, BLOCKSIZE, i, bk_c);
            }
        }

        __syncthreads();
        
    }


    if (out_r < N && M/BLOCKSIZE*BLOCKSIZE + bk_c < M) {
        getindex(m1, BLOCKSIZE, bk_r, bk_c) = getindex_T(A, a_pitch, out_r, M/BLOCKSIZE*BLOCKSIZE + bk_c);
    }

    if (out_c < S && M/BLOCKSIZE*BLOCKSIZE + bk_r < M) {
        getindex(m2, BLOCKSIZE, bk_r, bk_c) = getindex(C, S, M/BLOCKSIZE*BLOCKSIZE + bk_r, out_c);
    }


    __syncthreads();

    if (out_r < N && out_c < S) {
        for (int i = 0; i < M % BLOCKSIZE; i++) {
            sum += getindex(m1, BLOCKSIZE, bk_r, i) * getindex(m2, BLOCKSIZE, i, bk_c);
        }
    }



    if (out_r < N && out_c < S) {
        getindex(Y, S, out_r, out_c) = sum;
    }
}


void smem_opt(torch::Tensor A, int64_t a_pitch, torch::Tensor B, torch::Tensor C, torch::Tensor X, int64_t x_pitch, torch::Tensor Y) {
    int64_t M = A.size(0);
    int64_t N = A.size(1);
    int64_t P = B.size(1);
    int64_t S = C.size(1);

    const dim3 threads_per_block(BLOCKSIZE * BLOCKSIZE);
    const dim3 nr_blocks(CEIL_DIV(MAX(P, S), BLOCKSIZE), CEIL_DIV(MAX(M, N), BLOCKSIZE));

    kernel_smem_opt<<<nr_blocks, threads_per_block>>>(
        A.data_ptr<float>(), a_pitch, 
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        X.data_ptr<float>(), x_pitch,
        Y.data_ptr<float>(),
        M, N, P, S);
}





TORCH_LIBRARY_IMPL(my_cuda_kernels, CUDA, m) {
    m.impl("smem_opt", &smem_opt);
}




