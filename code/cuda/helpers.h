#IFNDEF HELPERSH
#DEFINE HELPERSH

// input    (BS, M)
// input_a  (BS, K)
// input_ab (BS, M-K)
// W        (N, M)
// A        (N, K)
// B        (K, M-K)
// out      (BS, N)
// out = input * W^T

// ATTENTION, MACROS!
// tread carefully!

#define getindex_A(A, N, M, K, i, j) A[i*K+j]
#define getindex_A_T(A, N, M, K, i, j) getindex_A(A, N, M, K, j, i)

#define getindex_B(B, N, M, K, i, j) B[i*(M-K)+j]
#define getindex_B_T(B, N, M, K, i, j) getindex_B(B, N, M, K, j, i)

#define getindex_input(in, N, M, K, i, j) in[i*M + j]
#define getindex_input_a(in, N, M, K, i, j)  getindex_input(in, N, M, K, i, j) // input[:, :rank]
#define getindex_input_ab(in, N, M, K, i, j) getindex_input(in, N, M, K, i, j+K) // input[:, rank:]

#define getindex_out(out, N, M, K, i, j) out[i*N + j]


#ENDIF