
#include "helpers.h"




__global__ void mm(const float* A, const float* B, const float* input, float* out, int N, int M, int K, int BS) {
    // out = input * [A; A*B]^T = Xa * A^T + Xab * B^T = (Xab*B^T .+ Xa) * A^T
    // input    (BS, M)
    // input_a  (BS, K)
    // input_ab (BS, M-K)
    // W        (N, M)
    // A        (N, K)
    // B        (K, M-K)
    // out      (BS, N)

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < BS && col < N) {
        float sum = 0.0f;

        for (int i = 0; i < K; i++) {

            float mid_row_i = getindex_input_a(input, N, M, K, row, i)
            for (int j = 0; j < M-K; j++) {
                mid_row_i += getindex_input_ab(input, N, M, K, row, j) * getindex_B_T(B, N, M, K, j, i)
            }

            sum += mid_row_i * getindex_A_T(A, N, M, K, i, col);
        }

        getindex_out(out, N, M, K, row, col) = sum;
    }
}


//    // Allocate host matrices
//     float *h_A = (float*)malloc(bytes);
//     float *h_B = (float*)malloc(bytes);
//     float *h_C = (float*)malloc(bytes);

//     // Initialize A and B
//     for (int i = 0; i < n * n; ++i) {
//         h_A[i] = 1.0f;  // example: all ones
//         h_B[i] = (i % n == i / n) ? 2.0f : 0.0f;  // example: diagonal = 2
//     }

//     // Allocate device matrices
//     float *d_A, *d_B, *d_C;
//     cudaMalloc(&d_A, bytes);
//     cudaMalloc(&d_B, bytes);
//     cudaMalloc(&d_C, bytes);

//     // Copy inputs to device
//     cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

//     // Launch kernel: one thread per element
//     dim3 block(16, 16);
//     dim3 grid((n + block.x - 1) / block.x,
//               (n + block.y - 1) / block.y);
//     matmul<<<grid, block>>>(d_A, d_B, d_C, n);

//     // Copy result back to host
//     cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
