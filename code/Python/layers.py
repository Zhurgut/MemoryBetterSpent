
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import sys

from latin_squares import latin_square

import lowrankLight



init_scale = 1.0 # change init scale for Monarch, TT

def set_scale(s):
    global init_scale
    init_scale = s


def to2D(x):
    N = x.shape[-1]
    BS = math.prod(x.shape[:-1])
    X = x.reshape(BS, N)

    return X, x.shape

def undo_to2D(out, shape):
    
    return out.reshape(*(shape[:-1]), -1)



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


# TODO adjust all masked sparse layers to take percentage as sparsity parameter, so type int!


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
    
    def __init__(self, in_dim, out_dim, density):

        density = density/100 # pass arg as percentage

        self.density = density
        
        m = torch.rand(out_dim, in_dim)
        t = torch.sort(m.flatten(), descending=True)[0][(density * torch.ones(1) * out_dim*in_dim).floor().int()]
        mask = m >= t
        
        super().__init__(in_dim, out_dim, mask)
        

    def project(self, fn):
        density = self.density
        M = fn.weight

        assert M.shape == (self.out_dim, self.in_dim)
        
        t = torch.sort(M.abs().flatten(), descending=True)[0][(density * torch.ones(1) * self.out_dim*self.in_dim).floor().int()]
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
    


    def project_precise(self, fn):

        out_dim, in_dim = fn.weight.shape
        assert self.in_dim == in_dim and self.out_dim == out_dim

        rank = self.rank

        U, S, Vt = torch.linalg.svd(fn.weight)
        S = torch.diag(S)
        
        A = U[:, :rank] @ S[:rank, :rank]
        v = Vt[:rank, :]
        B1 = v[:, :rank]
        B2 = v[:, rank:]
        
        out_U = A @ B1
        out_V = torch.linalg.solve(B1, B2) # will never actually fail, because svd is noisy

        self.A = nn.Parameter(out_U)
        self.B = nn.Parameter(out_V)

        self.bias = nn.Parameter(fn.bias.clone().detach())
    
    def project_regularized(self, fn):

        out_dim, in_dim = fn.weight.shape
        assert self.in_dim == in_dim and self.out_dim == out_dim

        rank = self.rank

        W1 = fn.weight[:, :rank]
        W2 = fn.weight[:, rank:]
        
        # solve W1*B = W2 with ridge regression
        h = 1e-5 * torch.trace(W1.T @ W1) / rank
        out_V = torch.linalg.solve(W1.T @ W1 + h * torch.eye(rank, rank, device=W1.device), W1.T @ W2)

        self.A = nn.Parameter(W1)
        self.B = nn.Parameter(out_V)

        self.bias = nn.Parameter(fn.bias.clone().detach())
    

    def project_GD(self, fn):

        out_dim, in_dim = fn.weight.shape
        assert self.in_dim == in_dim and self.out_dim == out_dim

        rank = self.rank

        W1 = fn.weight[:, :rank]
        W2 = fn.weight[:, rank:]

        # A = nn.Parameter(  (W1.clone() + torch.randn(W1.size(), device=W1.device)).detach() )
        # B = nn.Parameter(  (torch.linalg.pinv(A) @ W2).detach()  )

        A = nn.Parameter( torch.randn(W1.size(), device=W1.device) )
        B = nn.Parameter( torch.randn(rank, self.in_dim-rank, device=W1.device) )

        m, k = A.shape
        K, n = B.shape
        
        lr = 1e-2
        nr_steps = 15000
        
        opt = torch.optim.AdamW([A, B], lr=lr, weight_decay=0.001)

        x = fn.weight.abs().max()

        for i in range(nr_steps):
            
            opt.zero_grad()

            loss = (A-W1).norm().square() / (m*k) + (A@B-W2).norm().square() / (K*n)

            loss.backward()

            opt.step()

            if B.abs().max() > 2*x:
                break
        

        self.A = nn.Parameter(A.clone().detach())
        self.B = nn.Parameter(B.clone().detach())

        self.bias = nn.Parameter(fn.bias.clone().detach())


    def project(self, fn):

        self.project_precise(fn)
        # self.project_regularized(fn)
        # self.project_GD(fn) # not work well



class BlockDiagonal(Projectable):
    
    def __init__(self, in_dim, out_dim, nr_blocks, bias=True):
        assert in_dim % nr_blocks == 0 
        assert out_dim % nr_blocks == 0 

        super().__init__(in_dim, out_dim)
        
        self.nr_blocks = nr_blocks
        self.block_size_in = in_dim // nr_blocks
        self.block_size_out = out_dim // nr_blocks

        bound = math.sqrt(1 / math.sqrt(in_dim))
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

        self.inner_dim = out_dim # projection only works with this somehow
        inner_dim = self.inner_dim
        self.nr_blocks = nr_blocks

        self.fn = nn.Sequential(
            BlockDiagonal(in_dim, inner_dim, nr_blocks, bias=False), # R^T
            Permute(inner_dim, nr_blocks),       # P
            BlockDiagonal(inner_dim, out_dim, out_dim // nr_blocks, bias=False), # L^T 
            Permute(out_dim, out_dim // nr_blocks) # P^T
        )
        
        self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())

        self.properly_initialized = False

        


    def forward(self, x):

        if not self.properly_initialized:
            l = nn.Linear(self.in_dim, self.out_dim, device=x.device)
            self.project(l)
            self.properly_initialized = True

        x, shape = to2D(x)
        out = self.fn(x) + self.bias
        return undo_to2D(out, shape)
    

    def project(self, fn):

        out_dim, in_dim = fn.weight.shape
        assert self.in_dim == in_dim and self.out_dim == out_dim

        nr_blocks = self.nr_blocks

        m = fn.weight.T.reshape(nr_blocks, self.in_dim // nr_blocks, nr_blocks, self.out_dim // nr_blocks)

        with torch.no_grad():
            for i in range(nr_blocks):
                for j in range(self.out_dim // nr_blocks):
                    y = m[i, :, :, j]
                    s = torch.linalg.svd(y)
                    d = math.sqrt(s.S[0])
                    self.fn[0].weights[i, :, j] = d * s.U[:, 0]
                    self.fn[2].weights[j, i, :] = d * s.Vh[0, :]

        self.bias = nn.Parameter(fn.bias.clone().detach())

        self.properly_initialized = True

    



class TT(Projectable):
        
    def __init__(self, in_dim, out_dim, nr_cores, rank):
        super().__init__(in_dim, out_dim)

        assert in_dim == out_dim
        size = in_dim
        
        self.core_size = int(round(size**(1/nr_cores)))
        
        assert self.core_size ** nr_cores == size
        # assert self.check_rank_not_too_big(size, rank, nr_cores), "rank too big, more parameters than dense model..."

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

        G = self.to_matrix()

        x, shape = to2D(x)
        out = x @ G + self.bias
        return undo_to2D(out, shape)
    





def Kronecker(size):
    return TT(size, 2, 1)



class BTT(Projectable):
    
    def __init__(self, in_dim, out_dim, nr_cores, rank):
        super().__init__(in_dim, out_dim)

        assert in_dim == out_dim
        size = in_dim
        
        self.core_size = int(round(size**(1/nr_cores)))
        
        # assert self.core_size ** nr_cores == size
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
        
        self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())


    def check_rank_not_too_big(self, size, rank, nr_cores):
        nr_params = rank * (self.core_size ** (nr_cores+1)) * (2 + (nr_cores - 2) * rank)
        return nr_params <= size*size

    
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

        d = self.core_size
        c = self.nr_cores
        z = x.reshape(x.shape[0], *(d for _ in range(c)), 1)
        
        for idx, core in enumerate(self.cores):
            t = self.nr_cores - idx - 1
            j = d ** t
            i = d ** idx
            z = torch.einsum("Bjxia,jxyiba->Bjyib", z.reshape(z.shape[0], j, d, i, z.shape[-1]), core.reshape(j, d, d, i, *core.shape[-2:]))

        return z.reshape(x.shape[0], -1) + self.bias
        
        # G = self.to_matrix()

        # x, shape = to2D(x)
        # out = x @ G + self.bias
        # return undo_to2D(out, shape)


class Blast(Projectable):

    def __init__(self, in_dim, out_dim, block_size, rank):
        super().__init__(in_dim, out_dim)

        assert in_dim % block_size == 0
        assert out_dim % block_size == 0

        self.block_size = block_size
        self.rank = rank

        b_in = in_dim // block_size
        b_out = out_dim // block_size


        self.S = nn.Parameter(nn.init.uniform_(torch.empty(b_out, b_in, rank), a=0, b=2)) # just like in paper

        # with S fixed like this, and U, V having entries drawn from Unif(-b, b), we expect a distribution in the manifested matrix of roughly N(0, 4nb^4/27), 
        # that is a standart deviation of 2sqrt(n)b^2/(3*sqrt(3)), where n is the rank here
        # to mimic dense, we want that the standart deviation = 1/2 * k, where entries in dense are drawn from Unif(-k, k), k = 1/sqrt(in_dim)
        # therefor we chose
        bound = math.sqrt(math.sqrt(27) / (4*rank))

        # self.U = nn.Parameter(nn.init.uniform_(torch.empty(b_out, rank, block_size), a=-bound, b=bound))
        # self.Vt = nn.Parameter(nn.init.uniform_(torch.empty(b_in, block_size, rank), a=-bound, b=bound))

        # in the paper: 
        self.U = nn.Parameter(nn.init.normal_(torch.empty(b_out, rank, block_size), mean=0, std=math.sqrt(0.02)))
        self.Vt = nn.Parameter(nn.init.normal_(torch.empty(b_in, block_size, rank), mean=0, std=math.sqrt(0.02)))

        self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())
        
        
    def forward2(self, x):
        x, shape = to2D(x)

        BS, M = x.shape
        xp = x.reshape(BS, -1, self.block_size).transpose(0, 1) # b_in, BS, block_size
        y = torch.bmm(xp, self.Vt) # b_in, BS, rank
        z = torch.einsum("ibr,oir->obr", y, self.S) # b_out, BS, rank
        out = torch.bmm(z, self.U) # b_out, BS, block_size
        out = out.transpose(0, 1).reshape(BS, -1) + self.bias
        
        return undo_to2D(out, shape)

    def forward(self, x): # much faster
        x, shape = to2D(x)

        W = self.forward2(torch.eye(self.in_dim, self.in_dim, device=x.device))

        out = torch.addmm(self.bias, x, W)  
        
        return undo_to2D(out, shape)
