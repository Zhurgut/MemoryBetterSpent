
import torch

def latin_square_power2(n):
    out = torch.zeros(n, n, dtype=torch.int)
    
    for k in range(1, n+1):
        shift = k * 2 - 1
        c = k-1
        for r in range(n):
            out[r, c % n] = k
            c += shift
    
    return out


def latin_square_any_n(n):
    out = torch.zeros(n, n, dtype=torch.int)
    
    for k in range(1, n+1):
        shift = 1
        c = k-1
        for r in range(n):
            out[r, c % n] = k
            c += shift
    
    return out


def latin_square_product(s1, s2):
    n1, n2 = s1.shape[0], s2.shape[0]
    
    if n1 > n2:
        return latin_square_product(s2, s1)
    
    out = torch.zeros(n1*n2, n1*n2, dtype=torch.int)
    
    for br in range(n1):
        for bc in range(n1):
            r = br*n2
            c = bc*n2
            out[r:r+n2, c:c+n2] = s2 + (s1[br, bc]-1)*n2
    
    
    return out



def latin_square(n):
    
    def factorize_power_of_two(n):
        k = 1
        while n % 2 == 0:
            n //= 2
            k *= 2
        return k, n  # returns (power_of_two, remaining_factor)
    
    k, m = factorize_power_of_two(n)
    
    s1 = latin_square_power2(k)
    s2 = latin_square_any_n(m)
    
    return latin_square_product(s1, s2)