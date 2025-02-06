
import torch
from torch import nn
import numpy as np


# size always the first parameter

def Dense(size):
    return nn.Linear(size, size)



def LowRank(size, rank):
    return nn.Sequential(nn.Linear(size, rank), nn.Linear(rank, size))


class BlockDiagonal(nn.Module):
    
    def __init__(self, size, nr_blocks):
        super().__init__()
        
        assert size % nr_blocks == 0 
        
        self.size = size
        self.nr_blocks = nr_blocks
        self.block_size = size // nr_blocks

        w = torch.zeros(nr_blocks, self.block_size, self.block_size)
        nn.init.kaiming_normal_(w)
        self.weight = nn.Parameter(w)
        
        b = torch.zeros(size)
        nn.init.uniform_(b, -1, 1)
        self.bias   = nn.Parameter(np.sqrt(1/size) * b)
    
    def forward(self, x):
        
        bs, d = x.shape
        assert d == self.size
        
        input_mats = x.reshape(bs, self.nr_blocks, self.block_size, 1)
        
        # matmul broadcasts the bmm
        return torch.matmul(self.weight, input_mats).reshape(bs, d) + self.bias



class Permute(nn.Module):
    
    # permute as in apply the monarch permutation matrix
    
    def __init__(self, size, nr_blocks):
        super().__init__()
        
        assert size % nr_blocks == 0 
        
        self.size = size
        self.nr_blocks = nr_blocks

        # self.perm = self.permutation(nr_blocks, size)

    
    def forward(self, x):
        bs, s = x.shape
        return x.reshape(bs, self.nr_blocks, -1).permute((0, 2, 1)).reshape(bs, s)
        
    # def permutation(self, b, n):
    #     p = torch.zeros(n, dtype=torch.int)
    #     for i in range(n):
    #         p[i] = (i % b) * (n // b) + (i // b)
    #     return p
        
        
        
def Monarch(size, nr_blocks):
    return nn.Sequential(
        BlockDiagonal(size, nr_blocks), 
        Permute(size, nr_blocks),        # P
        BlockDiagonal(size, nr_blocks), 
        Permute(size, size // nr_blocks) # P^T
    )
        

class TT(nn.Module):
    
    
    def __init__(self, size, nr_cores, rank):
        super().__init__()
        
        self.core_size = int(round(size**(1/nr_cores)))
        
        assert self.core_size ** nr_cores == size

        self.size = size
        self.nr_cores = nr_cores
        self.rank = rank
        
        assert rank <= self.core_size # otherwise doesnt make sense, or does?
        
        # I swap rank a size here, otherwise, for absolute correctness, I would have to transpose these two dimensions later before computation.. not that it really matters here
        self.G1 = self.init_cores((1, self.core_size, rank, self.core_size))
        
        self.Gi = None
        if self.nr_cores - 2 > 0:
            self.Gi = self.init_cores((self.nr_cores - 2, self.rank, self.core_size, self.rank, self.core_size))
        
        self.Gc = self.init_cores((rank, self.core_size, 1, self.core_size))   
        
        b = torch.zeros(size)
        nn.init.uniform_(b, -1, 1)
        self.bias = nn.Parameter(np.sqrt(1/size) * b)
        
        self.init_permutations(self.nr_cores)


    def init_cores(self, dims):
        d = dims[-1] 
        G = torch.zeros(dims)
        nn.init.kaiming_normal_(G.reshape(-1, d, d))
        return nn.Parameter(G)
    
    
    def init_permutations(self, c):

        p = [None]
        for t in range(1, c+1):
            p.append((0, 1, t+1, *range(2, t+1), *range(t+2, c+2)))
        
        up = [None]
        for t in range(1, c+1):
            up.append((0, 1, *range(3, t+2), 2, *range(t+2, c+2)))
        
        self.perms = p
        self.unperms = up
    
    
    def contract_core(self, t, G, x):
        r, i, a, j = G.shape
        BS = x.shape[0] # BS, a, k..., j, l...

        z = x.permute(self.perms[t]) # BS, a, j, k..., l...

        S = a*j
        y = G.reshape(-1, S).matmul(z.reshape(BS, S, -1)) # matmul over a*j, new shape: r, i, k..., l...
        
        z = y.reshape(BS, r, i, *z.shape[3:]).permute(self.unperms[t]) # shape: r, k..., i, l...
        
        return z

        
    def forward(self, x):
        assert len(x.shape) > 1 # there needs to be a batch dimension pleease
        
        BS = x.shape[0]
        x = x.reshape(BS, 1, *[self.core_size for i in range(self.nr_cores)])
        
        t = self.nr_cores
        z = self.contract_core(t, self.Gc, x)
        t -= 1
        
        if self.Gi is not None:
            while t > 1:
                z = self.contract_core(t, self.Gi[t-2, :, :, :, :], z)
                t -= 1
            
        z = self.contract_core(1, self.G1, z)
        
        return z.reshape(BS, -1) + self.bias
        
        

def Kronecker(size):
    return TT(size, 2, 1)


class SkipConnection(nn.Module):
    
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return x + self.layer(x)


        
# model = Permute(32, 8)

# x = torch.arange(32).reshape(1, -1)

# print(model(x))


# model = Kronecker(4)
# x = torch.rand(1, 4)

# print(model(x))

