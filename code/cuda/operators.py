
import torch
try: # instead of torch.ops.load_library(path_to_.so_file)
    import my_cuda_kernels
except ImportError:
    pass

def reference(a, a_pitch, b, c, x, x_pitch, y):
    torch.matmul(a, b, out=x)
    torch.matmul(a.T, c, out=y)

def mm_and_mTm(op):
    
    def fn(A, B, C, X, Y):

        M, P = X.size()
        m, N = A.size()
        n, p = B.size()
        assert m == M
        assert n == N
        assert p == P
        n, S = Y.size()
        m, s = C.size()
        assert n == N
        assert m == M
        assert s == S

        assert B.is_contiguous()
        assert C.is_contiguous()
        assert Y.is_contiguous()

        assert A.stride(1) == 1
        assert X.stride(1) == 1

        return op(A, A.stride(0), B, C, X, X.stride(0), Y)

    return fn
    

versions = {k: mm_and_mTm(op) for (k, op) in {
    "reference": reference,
    "base": torch.ops.my_cuda_kernels.base,
}.items()}