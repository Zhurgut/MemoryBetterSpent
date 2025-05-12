
import torch
from torch import nn
import numpy as np
from math import sqrt

from latin_squares import latin_square

from ViT import *
import lowrankLight



init_scale = 1.0 # change init scale for Monarch, TT

def set_scale(s):
    global init_scale
    init_scale = s


# size always the first parameter


def to_dense(fn):
    
    def construct_linear(W, b):
        out_d, in_d = W.shape
        f = nn.Linear(in_d, out_d)
        with torch.no_grad():
            f.weight = nn.Parameter(W)
            f.bias   = nn.Parameter(b)
        return f
    if isinstance(fn, BlockDiagonal):
        W = fn.to_matrix()
        b = fn.bias if fn.use_bias else torch.zeros(W.shape[0])
        return W, b, construct_linear(W.T, b)
    if isinstance(fn, nn.Linear):
        return fn.weight.T, fn.bias, fn
    elif isinstance(fn, LowRank):
        W1 = fn.fn[0].weight
        W2 = fn.fn[1].weight
        W = W2 @ W1
        b  = fn.bias
        return W.T, b, construct_linear(W, b)
    elif isinstance(fn, Monarch):
        W = fn.fn[0].to_matrix()
        W = fn.fn[1](W)
        W2 = fn.fn[2].to_matrix()
        b = fn.bias

        W = W @ W2
        
        W = fn.fn[3](W)
        return W, b, construct_linear(W.T, b)
    elif isinstance(fn, TT) or isinstance(fn, BTT):
        W = fn.to_matrix()
        b = fn.bias
        return W, b, construct_linear(W.T, b)
    else:
        print("hello there...")



def Dense(size):
    return nn.Linear(size, size)


class MaskedSparse(nn.Module):
    
    def __init__(self, size, mask):
        super().__init__()
        
        self.mask = mask
        
        self.size = size
        
        w = torch.zeros(size, size)
        nn.init.kaiming_normal_(w)
        self.weight = nn.Parameter(init_scale * w)
        
        b = torch.zeros(size)
        nn.init.uniform_(b, -1, 1)
        self.bias   = nn.Parameter(np.sqrt(1/size) * b)
    
    
    def forward(self, x):
        return x @ (self.mask * self.weight).T + self.bias
    
    
    
    def to(self, device):

        super(MaskedSparse, self).to(device)

        self.mask = self.mask.to(device)
        
        return self


class Unstructured(MaskedSparse):
    
    def __init__(self, size, sparsity):
        
        m = torch.rand(size, size)
        t = torch.sort(m.flatten(), descending=True)[0][(sparsity * torch.ones(1) * size*size).floor().int()]
        mask = m >= t
        
        super().__init__(size, mask)
        

    @staticmethod
    def from_mag_pruned(M, sparsity):
        size = M.shape[0]
        assert M.shape == (size, size)
        
        t = torch.sort(M.abs().flatten(), descending=True)[0][(sparsity * torch.ones(1) * size*size).floor().int()]
        mask = M.abs() >= t

        out = Unstructured(size, sparsity)
        out.weight = nn.Parameter(M.clone().detach())
        out.mask = mask
        
        return out



class BlockSparse(MaskedSparse):
        
    def __init__(self, size, nr_blocks_per_row, nr_blocks_to_drop_per_row):
            
        S = latin_square(nr_blocks_per_row)
        block_size = size // nr_blocks_per_row

        m = S > nr_blocks_to_drop_per_row
        mask = m.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
        
        super().__init__(size, mask)


    @staticmethod
    def from_mag_pruned(M, nr_blocks_per_row, nr_blocks_to_drop_per_row):
        size = M.shape[0]
        assert M.shape == (size, size)
        
        block_size = size // nr_blocks_per_row
        blocks_to_keep = nr_blocks_per_row * (nr_blocks_per_row - nr_blocks_to_drop_per_row)
        
        Z = nn.AvgPool2d(block_size, stride=block_size, divisor_override=1)(M.abs().unsqueeze(0).unsqueeze(0)).squeeze()
        t = torch.sort(Z.flatten(), descending=True)[0][blocks_to_keep-1]
        m = Z >= t
        mask = m.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

        out = BlockSparse(size, nr_blocks_per_row, nr_blocks_to_drop_per_row)
        out.weight = nn.Parameter(M.clone().detach())
        out.mask = mask
        
        return out



class LowRank(nn.Module):
    
    def __init__(self, size, rank):
        super().__init__()
        
        self.fn = nn.Sequential(nn.Linear(size, rank, bias=False), nn.Linear(rank, size, bias=False))
    
        
        b = torch.zeros(size)
        nn.init.uniform_(b, -1, 1)
        self.bias   = nn.Parameter(np.sqrt(1/size) * b)
    
    def forward(self, x):
        return self.fn(x) + self.bias


def LowRankLight(size, rank):
    return lowrankLight.LowRankLight(size, size, rank)

LowRankLight.from_matrix = lowrankLight.LowRankLight.from_matrix  


class BlockDiagonal(nn.Module):
    
    def __init__(self, size, nr_blocks, bias=True):
        super().__init__()
        
        assert size % nr_blocks == 0 
        
        self.use_bias = bias
        
        self.size = size
        self.nr_blocks = nr_blocks
        self.block_size = size // nr_blocks

        w = torch.zeros(nr_blocks, self.block_size, self.block_size)
        nn.init.kaiming_normal_(w)

        self.weights = nn.Parameter(init_scale * w)
        
        if self.use_bias:
            b = torch.zeros(size)
            nn.init.uniform_(b, -1, 1)
            self.bias   = nn.Parameter(np.sqrt(1/size) * b)
    
    def forward(self, x):
        z = x.reshape(-1, self.size)
        batch_size, n = z.shape
        
        input_mats = z.reshape(batch_size, self.nr_blocks, self.block_size).transpose(0, 1) # (nr_blocks, batch_size, block_size) * weights: (nr_blocks, block_size, block_size)
        # 
        out = torch.bmm(input_mats, self.weights) # nr_blocks, batch_size, block_size
        
        if self.use_bias:
            return (out.transpose(0, 1).reshape(batch_size, n) + self.bias).reshape(x.shape)
        else:
            return out.transpose(0, 1).reshape(batch_size, n).reshape(x.shape)
    
    def to_matrix(self):
        return torch.block_diag(*self.weights)



class Permute(nn.Module):
    
    # permute as in apply the monarch permutation matrix
    
    def __init__(self, size, nr_blocks):
        super().__init__()
        
        assert size % nr_blocks == 0 
        
        self.size = size
        self.nr_blocks = nr_blocks

        # self.perm = self.permutation(nr_blocks, size)

    
    def forward(self, x):
        z = x.reshape(-1, self.size)
        bs, s = z.shape
        return z.reshape(bs, self.nr_blocks, -1).transpose(1, 2).reshape(x.shape)
        
    # def permutation(self, b, n):
    #     p = torch.zeros(n, dtype=torch.int)
    #     for i in range(n):
    #         p[i] = (i % b) * (n // b) + (i // b)
    #     return p


class Monarch(nn.Module):
    
    def __init__(self, size, nr_blocks):
        super().__init__()
        self.size = size
        
        self.fn = nn.Sequential(
            BlockDiagonal(size, nr_blocks, bias=False), 
            Permute(size, nr_blocks),        # P
            BlockDiagonal(size, nr_blocks, bias=False), 
            Permute(size, size // nr_blocks) # P^T
        )
        
        b = torch.zeros(size)
        nn.init.uniform_(b, -1, 1)
        self.bias = nn.Parameter(np.sqrt(1/size) * b)

    def forward(self, x):
        M = self.fn(torch.eye(self.size, self.size).to(x.device))
        return x @ M.T + self.bias
    



class TT(nn.Module):
        
    def __init__(self, size, nr_cores, rank):
        super().__init__()
        
        self.core_size = int(round(size**(1/nr_cores)))
        
        assert self.core_size ** nr_cores == size
        assert self.check_rank_not_too_big(size, rank, nr_cores), "rank too big, more parameters than dense model..."

        self.size = size
        self.nr_cores = nr_cores
        self.rank = rank
        
        d = self.core_size
        r = self.rank
        
        self.cores = nn.ParameterList([
              nn.Parameter(self.init_core((1, d, d, r))), # Gc
            *(nn.Parameter(self.init_core((r, d, d, r))) for i in range(self.nr_cores - 2)),
              nn.Parameter(self.init_core((r, d, d, 1))) # G1
        ])
        
        b = torch.zeros(size)
        nn.init.uniform_(b, -1, 1)
        self.bias = nn.Parameter(np.sqrt(1/size) * b)


    def check_rank_not_too_big(self, size, rank, nr_cores):
        nr_params = rank * self.core_size * self.core_size * (2 + (nr_cores - 2) * rank)
        return nr_params <= size*size
    

    def init_core(self, dims):
        r = self.rank
        c = self.nr_cores
        t = 1/sqrt(self.size) # target std of entire weight matrix, to mimic kaiming_normal
        sigma = (t * (sqrt(r) ** (1-c))) ** (1/c)
        G = torch.zeros(dims)
        nn.init.normal_(G, std=sigma)
        return init_scale * G

    def to_matrix(self):
        G = self.cores[0][0, :, :, :]
        for core in self.cores[1:]:
            G = torch.einsum("...a,aijb->...ijb", G, core)

        return G.reshape(self.size, self.size)
          
    def forward(self, x):
        # manifest the matrix, then matmul, much faster than the "efficient" way, probably because large batch size 
        G = self.to_matrix()

        return (torch.matmul(x.reshape(-1, self.size), G) + self.bias).reshape(x.shape)
    





def Kronecker(size):
    return TT(size, 2, 1)



class BTT(nn.Module):
    
    def __init__(self, size, nr_cores, rank):
        super().__init__()
        
        self.core_size = int(round(size**(1/nr_cores)))
        
        assert self.core_size ** nr_cores == size
        # assert self.check_rank_not_too_big(size, rank, nr_cores), "rank too big, more parameters than dense model..."

        self.size = size
        self.nr_cores = nr_cores
        self.rank = rank
        
        d = self.core_size
        r = self.rank
        
        self.cores = nn.ParameterList([
              nn.Parameter(self.init_core((*(d for i in range(nr_cores+1)), r, 1))),
            *(nn.Parameter(self.init_core((*(d for i in range(nr_cores+1)), r, r))) for i in range(self.nr_cores - 2)),
              nn.Parameter(self.init_core((*(d for i in range(nr_cores+1)), 1, r)))
        ])
        
        b = torch.zeros(size)
        nn.init.uniform_(b, -1, 1)
        self.bias = nn.Parameter(np.sqrt(1/size) * b)


    def check_rank_not_too_big(self, size, rank, nr_cores):
        nr_params = rank * (self.core_size ** (nr_cores+1)) * (2 + (nr_cores - 2) * rank)
        return nr_params <= size*size
    

    @staticmethod
    def from_matrix(M, rank):
        size = M.shape[0]
        assert M.shape == (size, size)
        
        out = BTT(size, 2, rank)
        d = out.core_size
        
        m = M.reshape(d, d, d, d)
        
        
        with torch.no_grad():
            for i in range(d):
                for j in range(d):
                    c = m[i, j, :, :]
                    # print(i, ", ", j, ":\n", c)
                    U, S, Vh = torch.linalg.svd(c)
                    # if i == j == 0:
                    #     print(c)
                    #     print(U)
                    #     print(S)
                    #     print(Vh)
                        
                    s = S[0:rank].sqrt()
                    
                    for r in range(rank):
                        out.cores[0].reshape(d, d, d, rank)[i, j, :, r] = s[r] * U[:, r]
                        out.cores[1].reshape(d, d, d, rank)[i, j, :, r] = s[r] * Vh[r, :]
        
        return out
    
    
    def init_core(self, dims):
        # r = self.rank
        # c = self.nr_cores
        # t = 1/sqrt(self.size) # target std of entire weight matrix, to mimic kaiming_normal
        # sigma = (t * (sqrt(r) ** (1-c))) ** (1/c)
        # G = torch.zeros(dims)
        # nn.init.normal_(G, std=sigma)
        G = torch.zeros(dims)
        nn.init.kaiming_normal_(G.reshape(-1, self.core_size, self.core_size))
        
        return init_scale * G

    def to_matrix(self):
        # efficient way, dont manifest matrix: (runs out of gpu memory because I use large batch size)
        d = self.core_size
        c = self.nr_cores
        # z = x.reshape(x.shape[0], *(d for _ in range(c)), 1)
        
        # for idx, core in enumerate(self.cores):
        #     t = self.nr_cores - idx - 1
        #     j = d ** t
        #     i = d ** idx
        #     z = torch.einsum("Bjxia,jxyiba->Bjyib", z.reshape(z.shape[0], j, d, i, z.shape[-1]), core.reshape(j, d, d, i, *core.shape[-2:]))

        # return z.reshape(x.shape[0], -1) + self.bias
        
        # manifest the matrix, then matmul
        G = self.cores[0]
        for idx, core in enumerate(self.cores[1:]):
            # t = self.nr_cores - 1 - idx
            j = d ** (idx+1)
            i = d ** (self.nr_cores - idx - 1)
            G = torch.einsum("jixcb,jiyba->jiyxca", core.reshape(j, i, d, *core.shape[-2:]), G.reshape(j, i, d ** (idx + 1), *G.shape[-2:]))

        return G.reshape(self.size, self.size)
    
    
          
    def forward(self, x):
        
        G = self.to_matrix()
        # print(G.std())
        # exit()

        return (torch.matmul(x.reshape(-1, self.size), G) + self.bias).reshape(x.shape)




class BTTLight(nn.Module):
    
    def __init__(self, size, rank): # nr cores = 2
        super().__init__()
        
        self.size = size

        self.core_size = int(round(sqrt(size)))
        assert self.core_size * self.core_size == size
        
        self.nr_blocks = size

        self.A = nn.Parameter(nn.init.kaiming_normal_(torch.empty(self.nr_blocks, self.core_size, rank)))
        self.B = nn.Parameter(nn.init.kaiming_normal_(torch.empty(self.nr_blocks, rank, self.core_size - rank)))  

        b = torch.zeros(size)
        nn.init.uniform_(b, -1, 1)
        self.bias = nn.Parameter(np.sqrt(1/size) * b)
    
    
    def forward(self, x):
        AB = torch.bmm(self.A, self.B) # nr_blocks, core_size, core_size - rank
        W = torch.cat((self.A, AB), dim = 2).reshape(self.size, self.size)
        
        return x @ W.T + self.bias