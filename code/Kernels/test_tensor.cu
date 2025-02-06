
#include <stdlib.h>
#include <stdio.h>

#include <mma.h>
using namespace nvcuda;

#include <cuda_fp16.h>


#define BLOCK_SIZE 16


//  nvcc -arch=sm_86 test_tensor.cu for rtx 3090, sm_80 for a100



void init_matrices(int M, int K, int N, half** Ap, half** Bp, half** Cp) {

    *Ap = (half*) malloc(M*K * 2);
    *Bp = (half*) malloc(K*N * 2);
    *Cp = (half*) malloc(M*N * 2);

    half* A = *Ap;
    half* B = *Bp;
    half* C = *Cp;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A[i*K + j] = __float2half((float) ((i*j+1) % K) / K);
    
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[i*N + j] = __float2half((float) ((i*j+1) % M) / M);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            float tmp = 0.0;
            for (int k = 0; k < K; k++) {
                tmp += __half2float(A[i*K + k]) * __half2float(B[k*N + j]);
            }
            C[i*N + j] = __float2half(tmp);
        }
    }
            
}

void print_matrices(int M, int K, int N, half* A, half* B, half* C) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%.1f ", __half2float(A[i*K + j]));
        }
        printf("\n");
    }
    printf("\n");

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f ", __half2float(B[i*N + j]));
        }
        printf("\n");
    }
    printf("\n");

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f ", __half2float(C[i*N + j]));
        }
        printf("\n");
    }
    printf("\n");
}



__global__ void wmma_ker(int M, int K, int N, half *a, half *b, half *c) {

    int bi = blockIdx.x;
    int bj = blockIdx.y;

    half* c_tl = c + bi*BLOCK_SIZE*N + bj*BLOCK_SIZE;

    half* a_row_tl = a + bi*BLOCK_SIZE*K;
    half* b_col_tl = b + bj*BLOCK_SIZE;

    half* a_tl = a_row_tl; // address of first elemnt of current block
    half* b_tl = b_col_tl;


    // Declare the fragments
    wmma::fragment<wmma::matrix_a,    BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, half> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    for (int bk=0; bk < K/BLOCK_SIZE; bk++) { // accumulate all blocks into c_frag

        a_tl = a_row_tl + bk*BLOCK_SIZE; // address of first elemnt of current block
        b_tl = b_col_tl + bk*BLOCK_SIZE*N;

        // Load the inputs
        wmma::load_matrix_sync(a_frag, a_tl, K);
        wmma::load_matrix_sync(b_frag, b_tl, N);

        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    }
    // Store the output
    wmma::store_matrix_sync(c_tl, c_frag, N, wmma::mem_row_major);
}


void tensor_mma(int M, int K, int N, half *A, half *B, half *C) {

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(M/BLOCK_SIZE, N/BLOCK_SIZE);
    wmma_ker<<<grid,block>>>(M, K, N, A, B, C); 
    cudaDeviceSynchronize();

    cudaError_t cudaStatus = cudaGetLastError();
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }

}


void run_kernel(int M, int K, int N) {

    half *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, M*K*sizeof(half));
    cudaMalloc((void**) &B_d, K*N*sizeof(half));
    cudaMalloc((void**) &C_d, M*N*sizeof(half));

    half* A = NULL;
    half* B = NULL;
    half* C = NULL;

    init_matrices(M, K, N, &A, &B, &C);

    print_matrices(M, K, N, A, B, C);

    
    cudaMemcpy((void*) A_d, (void*) A, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) B_d, (void*) B, K*N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) C_d, (void*) C, M*N*sizeof(half), cudaMemcpyHostToDevice);


    tensor_mma(M, K, N, A_d, B_d, C_d),

    cudaMemcpy(C, C_d, M*N*sizeof(half), cudaMemcpyDeviceToHost); 

    print_matrices(M, K, N, A, B, C);


    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A);
    free(B);
    free(C); 
}


int main(){

    run_kernel(32, 32, 32); // must be multiples of 16


}