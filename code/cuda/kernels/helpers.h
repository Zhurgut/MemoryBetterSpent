#ifndef HELPERSH
#define HELPERSH

// A = (M, N)
// B = (N, P)
// C = (M, S)
// X = (M, P)
// Y = (N, S)

// ATTENTION, MACROS!
// tread carefully!

#define MAX(x, y) (x > y ? x : y)
#define MIN(x, y) (x < y ? x : y)

#define getindex(ptr, pitch, r_idx, c_idx) ptr[r_idx * pitch + c_idx] // ptr[r_idx, c_idx]
#define getindex_T(ptr, pitch, r_idx, c_idx) getindex(ptr, pitch, c_idx, r_idx) // ptr.transpose()[r_idx, c_idx]


#endif