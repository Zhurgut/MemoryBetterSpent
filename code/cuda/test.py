
import torch
import operators

from time import perf_counter
from math import floor, sqrt, ceil

device = torch.device("cuda")


def validate():
    for ver in operators.versions.keys():
        failed = False
        for i in range(10):
            M, N, S, P = torch.randint(64, 256, (4,)).tolist()
            # M, N, S, P = 64, 64, 64, 64

            ref_A = torch.rand(M, N, device=device)
            ref_B = torch.rand(N, P, device=device)
            ref_C = torch.rand(M, S, device=device)
            ref_X = torch.zeros(M, P, device=device)
            ref_Y = torch.zeros(N, S, device=device)

            operators.versions["reference"](ref_A, ref_B, ref_C, ref_X, ref_Y)

        

            if ver != "reference":

                A = ref_A.clone()
                B = ref_B.clone()
                C = ref_C.clone()
                X = torch.zeros(M, P, device=device)
                Y = torch.zeros(N, S, device=device)

                operators.versions[ver](A, B, C, X, Y)

                try:
                    torch.testing.assert_close(A, ref_A)
                    torch.testing.assert_close(B, ref_B)
                    torch.testing.assert_close(C, ref_C)
                    torch.testing.assert_close(X, ref_X)
                    torch.testing.assert_close(Y, ref_Y)
                except:
                    print("validation FAILED for ", ver, ", sizes: ", M, ", ", N, ", ", S, ", ", P)
                    print("A - ref_A = ", torch.norm(A - ref_A).item())
                    print("B - ref_B = ", torch.norm(B - ref_B).item())
                    print("C - ref_C = ", torch.norm(C - ref_C).item())
                    print("X - ref_X = ", torch.norm(X - ref_X).item())
                    print("Y - ref_Y = ", torch.norm(Y - ref_Y).item())
                    
                    M, N, S, P = 33, 34, 35, 36

                    A = torch.ones(M, N, device=device)
                    B = torch.ones(N, P, device=device)
                    C = torch.ones(M, S, device=device)
                    X = torch.zeros(M, P, device=device)
                    Y = torch.zeros(N, S, device=device)

                    operators.versions[ver](A, B, C, X, Y)

                    torch.set_printoptions(threshold=1400, linewidth=250)

                    print(X)
                    print(Y)

                    failed = True
                    break
                
        if not failed:
            print("validation SUCCESS for ", ver)
                    

median_lb99s = [
    0,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,6,6,6,7,7,7,8,8,9,9,9,10,10,11,11,11,12,12,13,13,14,14,14,15,15,15,16,16,17,17,17,18,19,19,19,20,20,20,21,21,22,22,22,23,23,24,24
]
median_ub99s = [
    7,8,9,10,10,11,11,12,13,14,14,15,15,16,17,18,18,19,19,20,20,21,22,23,23,24,24,25,25,26,26,27,28,29,29,30,30,31,32,32,32,33,34,35,35,36,36,37,37,38,39,39,39,40,41,42,42,43,43,44,44,45,45,46,46
]

def median_ci99_indices(n):
    if n <= 7:
        return 0,n-1
    if n <= 72:
        return median_lb99s[n-8], median_ub99s[n-8]
    else:
        return int(floor(0.5*n - 1.288*sqrt(n))), int(ceil(0.5*n + 1.288*sqrt(n)))


def benchmark(fn):
    min_mments = 10
    max_mments = 200
    timeout_s = 20 # hard timeout
    stop_after_s = 5 # if enough measurements were collected, just call it

    times = []
    start_time = perf_counter()

    for i in range(min_mments):
        t1 = perf_counter()
        fn()
        torch.cuda.synchronize()
        t2 = perf_counter()
        times.append(t2-t1)
        if perf_counter() - start_time > timeout_s:
            break
    
    for i in range(max_mments - min_mments):
        if perf_counter() - start_time > stop_after_s:
            break
        t1 = perf_counter()
        fn()
        torch.cuda.synchronize()
        t2 = perf_counter()
        times.append(t2-t1)
    
    times.sort()
    n = len(times)
    lb, ub = median_ci99_indices(n)
    mid = times[(n-1) // 2] if n%2 == 1 else 0.5*(times[n // 2] + times[(n-1) // 2])

    return n, times[lb], mid, times[ub]

def extend_name(name, n):
    m = len(name)
    return name + (n-m)*" "

def benchmark_all():
    size = 8192
    M, N, S, P = size, size, size, size

    A = torch.rand(M, N, device=device)
    B = torch.rand(N, P, device=device)
    C = torch.rand(M, S, device=device)
    X = torch.zeros(M, P, device=device)
    Y = torch.zeros(N, S, device=device)

    for ver in operators.versions.keys():

        def bm_fn():
            operators.versions[ver](A, B, C, X, Y)

        n, lb, med, ub = benchmark(bm_fn)
        print(extend_name(ver, 14), ": ", round(1000*med, ndigits=2), "ms, (", round(1000*lb, ndigits=2), "ms, ", round(1000*ub, ndigits=2), "ms); \t", n, " measurements")
    

def profile(ver):
    size = 1024
    M, N, S, P = size, size, size, size
    A = torch.rand(M, N, device=device)
    B = torch.rand(N, P, device=device)
    C = torch.rand(M, S, device=device)
    X = torch.zeros(M, P, device=device)
    Y = torch.zeros(N, S, device=device)

    operators.versions[ver](A, B, C, X, Y)