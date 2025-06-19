
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from latin_squares import latin_square

import lowrankLight



init_scale = 1.0 # change init scale for Monarch, TT

def set_scale(s):
    global init_scale
    init_scale = s



class Projectable(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lr = 1e-3
        self.nr_steps = 5000

    """
    adjusts it's parameters to match the behaviour of the input fn
    i.e. project the input onto the space of matrices that I (self) can represent
    """
    def project(self, fn: nn.Linear):
        
        out_dim, in_dim = fn.weight.shape
        assert self.in_dim == in_dim and self.out_dim == out_dim

        lr = self.lr
        nr_steps = self.nr_steps
        
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=nr_steps, eta_min=lr / 100
        )

        if self.bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_dim, device=fn.weight.device, requires_grad=False))

        I = torch.eye(in_dim, in_dim, device=fn.weight.device)
        y = (fn(I) - fn.bias).detach()

        for i in range(nr_steps):
            
            opt.zero_grad()

            loss = torch.linalg.norm(y - self(I), ord="fro")
            loss.backward()

            opt.step()
            scheduler.step()

        if self.bias is not None: # respect the no bias having of self
            self.bias = nn.Parameter(fn.bias.clone().detach())


class SkipConnection(nn.Module):
    
    def __init__(self, fn, drop_rate=0.0):
        super().__init__()
        self.fn = fn
        self.drop_rate = drop_rate
    
    def forward(self, x):
        if self.training and torch.rand(()).item() < self.drop_rate:
            return x
        
        if self.drop_rate == 0.0:
            return x + self.fn(x)
        
        return x + (1/(1-self.drop_rate)) * self.fn(x)




def Dense(in_dim, out_dim):
    return nn.Linear(in_dim, out_dim)


class MaskedSparse(Projectable):
    
    def __init__(self, in_dim, out_dim, mask):
        super().__init__(in_dim, out_dim)
        
        self.mask = mask
        
        w = torch.zeros(out_dim, in_dim)
        nn.init.kaiming_normal_(w)
        self.weight = nn.Parameter(init_scale * w)
        
        b = torch.zeros(out_dim)
        nn.init.uniform_(b, -1, 1)

        self.bias = nn.Parameter(np.sqrt(1/out_dim) * b)
    
    
    def forward(self, x):
        return x @ (self.mask * self.weight).T + self.bias
    

    def to(self, device):

        super(MaskedSparse, self).to(device)

        self.mask = self.mask.to(device)
        
        return self


class Unstructured(MaskedSparse):
    
    def __init__(self, in_dim, out_dim, sparsity):

        self.sparsity = sparsity
        
        m = torch.rand(out_dim, in_dim)
        t = torch.sort(m.flatten(), descending=True)[0][(sparsity * torch.ones(1) * out_dim*in_dim).floor().int()]
        mask = m >= t
        
        super().__init__(in_dim, out_dim, mask)
        

    def project(self, fn):
        sparsity = self.sparsity
        M = fn.weight

        assert M.shape == (self.out_dim, self.in_dim)
        
        t = torch.sort(M.abs().flatten(), descending=True)[0][(sparsity * torch.ones(1) * self.out_dim*self.in_dim).floor().int()]
        self.mask = M.abs() >= t

        self.weight = nn.Parameter(M.clone().detach())
        self.bias = nn.Parameter(fn.bias.clone().detach())


class BlockSparse(MaskedSparse):
        
    def __init__(self, size, size2, nr_blocks_per_row, nr_blocks_to_drop_per_row):

        assert size == size2

        self.nr_blocks_per_row = nr_blocks_per_row
        self.nr_blocks_to_drop_per_row = nr_blocks_to_drop_per_row
            
        S = latin_square(nr_blocks_per_row)
        block_size = size // nr_blocks_per_row

        m = S > nr_blocks_to_drop_per_row
        mask = m.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
        
        super().__init__(size, size, mask)


    def from_mag_pruned(self, fn):
        M = fn.weight
        size = M.shape[0]
        assert M.shape == (self.out_dim, self.in_dim)

        nr_blocks_per_row = self.nr_blocks_per_row
        nr_blocks_to_drop_per_row = self.nr_blocks_to_drop_per_row
        
        block_size = size // nr_blocks_per_row
        blocks_to_keep = nr_blocks_per_row * (nr_blocks_per_row - nr_blocks_to_drop_per_row)
        
        Z = nn.AvgPool2d(block_size, stride=block_size, divisor_override=1)(M.abs().unsqueeze(0).unsqueeze(0)).squeeze()
        t = torch.sort(Z.flatten(), descending=True)[0][blocks_to_keep-1]
        m = Z >= t
        self.mask = m.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

        self.weight = nn.Parameter(M.clone().detach())
        self.bias = nn.Parameter(fn.bias.clone().detach())





# bound = 1 / sqrt(in_dim)
# self.A = nn.Parameter(nn.init.uniform_(torch.empty(out_dim, rank), a=-bound, b=bound))

# b = F.normalize(nn.init.normal_(torch.empty(rank, in_dim)), p=2, dim=0)

# self.B = nn.Parameter(b * 0.5*bound / (bound / sqrt(3))) 

# self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())


class LowRank(Projectable):
    
    def __init__(self, in_dim, out_dim, rank):
        super().__init__(in_dim, out_dim)

        l = nn.Linear(in_dim, out_dim)

        U, S, Vt = torch.linalg.svd(l.weight)
        s = torch.diag(S).sqrt()[:rank, :rank]

        self.A = nn.Parameter(U[:, :rank] @ s)
        self.B = nn.Parameter(s @ Vt[:rank, :])
        
        self.bias = nn.Parameter(l.bias)
    
    def forward(self, x):
        return (x @ self.B.T) @ self.A.T + self.bias


    def project(self, fn):
        
        out_dim, in_dim = fn.weight.shape
        assert self.in_dim == in_dim and self.out_dim == out_dim

        rank = self.B.size(0)

        U, S, Vt = torch.linalg.svd(fn.weight)
        s = torch.diag(S).sqrt()[:rank, :rank]

        self.A = nn.Parameter(U[:, :rank] @ s)
        self.B = nn.Parameter(s @ Vt[:rank, :])

        self.bias = nn.Parameter(fn.bias.clone().detach())


class LowRankLight(Projectable):

    def __init__(self, in_dim, out_dim, rank):
        super().__init__(in_dim, out_dim)

        assert rank <= min(out_dim, in_dim)
        
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.rank = rank
        
        bound = 1 / math.sqrt(in_dim)
        self.A = nn.Parameter(nn.init.uniform_(torch.empty(out_dim, rank), a=-bound, b=bound))
        # self.A = nn.Parameter(nn.init.kaiming_normal_(torch.empty(out_dim, rank)))
        
        b = F.normalize(nn.init.normal_(torch.empty(rank, in_dim - rank)), p=2, dim=0)

        self.B = nn.Parameter(b * 0.5*bound / (bound / math.sqrt(3))) # now all values in [A; A*B] roughly from the same distribution as A, which is the same distribution as nn.Linear
        
        self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())


    def forward(self, x):
        M = x.shape[-1]
        BS = math.prod(x.shape[:-1])
        shape = x.shape
        X = x.reshape(BS, M)

        X_a = X[:, :self.rank]
        X_ab = X[:, self.rank:]

        mid = torch.addmm(X_a, X_ab, self.B.T)
        out = torch.addmm(self.bias, mid, self.A.T)
        
        return out.reshape(*(shape[:-1]), -1)
    

    def project(self, fn):
        
        out_dim, in_dim = fn.weight.shape
        assert self.in_dim == in_dim and self.out_dim == out_dim

        rank = self.rank

        U, S, Vt = torch.linalg.svd(fn.weight)
        S = torch.diag(S)
        
        A = U[:, :rank] @ S[:rank, :rank]
        B = Vt[:rank, :]
        B1 = B[:, :rank]
        B2 = B[:, rank:]
        
        X = A @ B1
        Y = None
        try:
            Y = torch.linalg.solve(B1, B2)
        except:
            # B1 not invertible :(
            Y = torch.linalg.pinv(B1) @ B2
        
        self.A = nn.Parameter(X)
        self.B = nn.Parameter(Y)

        self.bias = nn.Parameter(fn.bias.clone().detach())



class BlockDiagonal(Projectable):
    
    def __init__(self, in_dim, out_dim, nr_blocks, bias=True):
        assert in_dim % nr_blocks == 0 
        assert out_dim % nr_blocks == 0 

        super().__init__(in_dim, out_dim)
        
        self.nr_blocks = nr_blocks
        self.block_size_in = in_dim // nr_blocks
        self.block_size_out = out_dim // nr_blocks

        bound = 1 / math.sqrt(self.block_size_in)
        w = nn.init.uniform_(torch.empty(nr_blocks, self.block_size_in, self.block_size_out), a=-bound, b=bound)
        
        self.weights = nn.Parameter(init_scale * w)
        
        if bias:
            self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())
        else:
            self.bias = None
    
    def forward(self, x):
        BS, in_dim = x.shape
        assert in_dim == self.in_dim
        
        input_mats = x.reshape(BS, self.nr_blocks, self.block_size_in).transpose(0, 1) # (nr_blocks, batch_size, block_in_dim) * weights: (nr_blocks, block_in_dim, block_out_dim)

        out = torch.bmm(input_mats, self.weights).transpose(0, 1).reshape(BS, self.out_dim) # nr_blocks, batch_size, block_size
        
        if self.bias is not None:
            return out + self.bias
        else:
            return out
    
    def to_matrix(self):
        return torch.block_diag(*self.weights)



class Permute(nn.Module):
    
    # permute as in apply the monarch permutation matrix
    
    def __init__(self, size, nr_blocks):
        assert size % nr_blocks == 0 
        
        super().__init__()
        
        self.size = size
        self.nr_blocks = nr_blocks

        # self.perm = self.permutation(nr_blocks, size)

    
    def forward(self, x):
        BS, D = x.shape
        assert D == self.size
        return x.reshape(BS, self.nr_blocks, -1).transpose(1, 2).reshape(BS, D)
        
    # def permutation(self, b, n):
    #     p = torch.zeros(n, dtype=torch.int)
    #     for i in range(n):
    #         p[i] = (i % b) * (n // b) + (i // b)
    #     return p


class Monarch(Projectable):
    
    def __init__(self, in_dim, out_dim, nr_blocks):
        super().__init__(in_dim, out_dim)
        
        inner_dim = ((in_dim*out_dim-1) // (in_dim+out_dim) // nr_blocks + 1) * nr_blocks
        inner_dim = min(in_dim, out_dim)

        assert inner_dim % nr_blocks == 0
        
        self.fn = nn.Sequential(
            BlockDiagonal(in_dim, inner_dim, nr_blocks, bias=False), # R^T
            Permute(inner_dim, inner_dim // nr_blocks),       # P
            BlockDiagonal(inner_dim, out_dim, nr_blocks, bias=False), # L^T 
            Permute(out_dim, nr_blocks) # P^T
        )
        
        self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())

    def forward(self, x):
        return self.fn(x) + self.bias
    
        # M = self.fn(torch.eye(self.in_dim, self.in_dim).to(x.device))
        # return x @ M.T + self.bias
    



class TT(nn.Module):
        
    def __init__(self, size, size2, nr_cores, rank):
        super().__init__()

        assert size == size2
        
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
        t = 1/math.sqrt(self.size) # target std of entire weight matrix, to mimic kaiming_normal
        sigma = (t * (math.sqrt(r) ** (1-c))) ** (1/c)
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
    
    def __init__(self, size, size2, nr_cores, rank):
        super().__init__()

        assert size == size2
        
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




# class BTTLight(nn.Module):
    
#     def __init__(self, size, rank): # nr cores = 2
#         super().__init__()
        
#         self.size = size

#         self.core_size = int(round(sqrt(size)))
#         assert self.core_size * self.core_size == size
        
#         self.nr_blocks = size

#         self.A = nn.Parameter(nn.init.kaiming_normal_(torch.empty(self.nr_blocks, self.core_size, rank)))
#         self.B = nn.Parameter(nn.init.kaiming_normal_(torch.empty(self.nr_blocks, rank, self.core_size - rank)))  

#         b = torch.zeros(size)
#         nn.init.uniform_(b, -1, 1)
#         self.bias = nn.Parameter(np.sqrt(1/size) * b)
    
    
#     def forward(self, x):
#         AB = torch.bmm(self.A, self.B) # nr_blocks, core_size, core_size - rank
#         W = torch.cat((self.A, AB), dim = 2).reshape(self.size, self.size)
        
#         return x @ W.T + self.bias