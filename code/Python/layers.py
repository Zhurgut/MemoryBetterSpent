import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import sys

from latin_squares import latin_square

import lowrankLight


init_scale = 1.0  # change init scale for Monarch, TT


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

        if self.bias is not None:
            self.bias = nn.Parameter(
                torch.zeros(out_dim, device=fn.weight.device, requires_grad=False)
            )

        opt = torch.optim.Adam([p for (n,p) in self.named_parameters() if not n.endswith("bias")], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=nr_steps, eta_min=lr / 100
        )

        I = torch.eye(in_dim, in_dim, device=fn.weight.device)
        y = (I @ fn.weight.T).detach()

        for i in range(nr_steps):

            opt.zero_grad()

            loss = (y - self(I)).norm()
            loss.backward()

            opt.step()
            scheduler.step()

            # if i % 10 == 0:
            #     print(loss.item())
            #     print(self.bias)

        if self.bias is not None:  # respect the no bias having of self
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

        return x + (1 / (1 - self.drop_rate)) * self.fn(x)


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

        self.bias = nn.Parameter(np.sqrt(1 / out_dim) * b)

    def forward(self, x):
        return x @ (self.mask * self.weight).T + self.bias

    def to(self, device):

        super(MaskedSparse, self).to(device)

        self.mask = self.mask.to(device)

        return self


class Unstructured(MaskedSparse):

    def __init__(self, in_dim, out_dim, density):

        density = density / 100  # pass arg as percentage

        self.density = density

        m = torch.rand(out_dim, in_dim)
        t = torch.sort(m.flatten(), descending=True)[0][
            (density * torch.ones(1) * out_dim * in_dim).floor().int()
        ]
        mask = m >= t

        super().__init__(in_dim, out_dim, mask)

    def project(self, fn):
        density = self.density
        M = fn.weight

        assert M.shape == (self.out_dim, self.in_dim)

        t = torch.sort(M.abs().flatten(), descending=True)[0][
            (density * torch.ones(1) * self.out_dim * self.in_dim).floor().int()
        ]
        self.mask = M.abs() >= t

        self.weight = nn.Parameter(M.clone().detach())
        self.bias = nn.Parameter(fn.bias.clone().detach())

    def project_no_magnitude_pruning(self, fn):
        density = self.density
        M = fn.weight

        assert M.shape == (self.out_dim, self.in_dim)

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
        mask = m.repeat_interleave(block_size, dim=0).repeat_interleave(
            block_size, dim=1
        )

        super().__init__(size, size, mask)

    def from_mag_pruned(self, fn):
        M = fn.weight
        size = M.shape[0]
        assert M.shape == (self.out_dim, self.in_dim)

        nr_blocks_per_row = self.nr_blocks_per_row
        nr_blocks_to_drop_per_row = self.nr_blocks_to_drop_per_row

        block_size = size // nr_blocks_per_row
        blocks_to_keep = nr_blocks_per_row * (
            nr_blocks_per_row - nr_blocks_to_drop_per_row
        )

        Z = nn.AvgPool2d(block_size, stride=block_size, divisor_override=1)(
            M.abs().unsqueeze(0).unsqueeze(0)
        ).squeeze()
        t = torch.sort(Z.flatten(), descending=True)[0][blocks_to_keep - 1]
        m = Z >= t
        self.mask = m.repeat_interleave(block_size, dim=0).repeat_interleave(
            block_size, dim=1
        )

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
        self.A = nn.Parameter(
            nn.init.uniform_(torch.empty(out_dim, rank), a=-bound, b=bound)
        )
        # self.A = nn.Parameter(nn.init.kaiming_normal_(torch.empty(out_dim, rank)))

        b = F.normalize(nn.init.normal_(torch.empty(rank, in_dim - rank)), p=2, dim=0)

        self.B = nn.Parameter(
            b * 0.5 * bound / (bound / math.sqrt(3))
        )  # now all values in [A; A*B] roughly from the same distribution as A, which is the same distribution as nn.Linear

        self.bias = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze()
        )

        # self.project(nn.Linear(in_dim, out_dim))

        self.regularization = 5e-5

    def forward(self, x):
        M = x.shape[-1]
        BS = math.prod(x.shape[:-1])
        shape = x.shape
        X = x.reshape(BS, M)

        X_a = X[:, : self.rank]
        X_ab = X[:, self.rank :]

        mid = torch.addmm(X_a, X_ab, self.B.T)
        out = torch.addmm(self.bias, mid, self.A.T)

        return out.reshape(*(shape[:-1]), -1)
    

    def project_regularized(self, fn, l):

        out_dim, in_dim = fn.weight.shape
        assert self.in_dim == in_dim and self.out_dim == out_dim

        rank = self.rank

        U, S, Vt = torch.linalg.svd(fn.weight)
        S = torch.diag(S)

        A = U[:, :rank] @ S[:rank, :rank]
        v = Vt[:rank, :]
        B1 = v[:, :rank]
        B2 = v[:, rank:]

        out_A = A @ B1 

        W1 = fn.weight[:, :rank]
        W2 = fn.weight[:, rank:]

        # solve A*B = W2 with ridge regression
        h = l * torch.trace(out_A.T @ out_A) / rank

        out_B = torch.linalg.solve(
            out_A.T @ out_A + h * torch.eye(rank, rank, device=out_A.device), out_A.T @ W2
        )
        # out_B = torch.linalg.lstsq(out_A, W2).solution

        self.A = nn.Parameter(out_A)
        self.B = nn.Parameter(out_B)

        # print(self.A.abs().max(), " - ", self.B.abs().max())


        self.bias = nn.Parameter(fn.bias.clone().detach())



    def project(self, fn):

        self.project_regularized(fn, self.regularization)



class SLTrain(Projectable):

    def __init__(self, in_dim, out_dim, rank):
        super().__init__(in_dim, out_dim)
        self.AB = LowRank(in_dim, out_dim, rank)
        self.sparse = Unstructured(in_dim, out_dim, 3)
        d = nn.Linear(in_dim, out_dim)
        self.bias = None
        # self.project(d)

        

    def project(self, fn):
        self.AB.project(fn)

        with torch.no_grad():
            self.AB.bias *= 0
            self.AB.bias.requires_grad = False

        R = fn.weight - self.AB(torch.eye(self.in_dim, self.in_dim, device=fn.weight.device)).T
        l = nn.Linear(self.in_dim, self.out_dim, device=fn.weight.device)
        l.weight = nn.Parameter(R)
        l.bias = nn.Parameter(fn.bias)

        self.sparse.project_no_magnitude_pruning(l)

        super().project(fn)
    
    def forward(self, x):
        return self.AB(x) + self.sparse(x)



 
class SLTrainLight(Projectable):

    def __init__(self, in_dim, out_dim, rank):
        super().__init__(in_dim, out_dim)
        self.AB = LowRankLight(in_dim, out_dim, rank)
        self.sparse = Unstructured(in_dim, out_dim, 3)
        d = nn.Linear(in_dim, out_dim)
        self.bias = None
        # self.project(d)

        

    def project(self, fn):
        self.AB.project(fn)

        with torch.no_grad():
            self.AB.bias *= 0
            self.AB.bias.requires_grad = False

        R = fn.weight - self.AB(torch.eye(self.in_dim, self.in_dim, device=fn.weight.device)).T
        l = nn.Linear(self.in_dim, self.out_dim, device=fn.weight.device)
        l.weight = nn.Parameter(R)
        l.bias = nn.Parameter(fn.bias)

        self.sparse.project_no_magnitude_pruning(l)

        super().project(fn)
    
    def forward(self, x):
        return self.AB(x) + self.sparse(x)
       


class BlockDiagonal(Projectable):

    def __init__(self, in_dim, out_dim, nr_blocks, bias=True):
        assert in_dim % nr_blocks == 0
        assert out_dim % nr_blocks == 0

        super().__init__(in_dim, out_dim)

        self.nr_blocks = nr_blocks
        self.block_size_in = in_dim // nr_blocks
        self.block_size_out = out_dim // nr_blocks

        bound = math.sqrt(1 / math.sqrt(in_dim))
        w = nn.init.uniform_(
            torch.empty(nr_blocks, self.block_size_out, self.block_size_in),
            a=-bound,
            b=bound,
        )

        self.weights = nn.Parameter(init_scale * w)

        if bias:
            self.bias = nn.Parameter(
                nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze()
            )
        else:
            self.bias = None

    def forward(self, x):
        BS, in_dim = x.shape
        assert in_dim == self.in_dim

        input_mats = x.reshape(BS, self.nr_blocks, self.block_size_in).transpose(0, 1)  # (nr_blocks, batch_size, block_in_dim) * weights: (nr_blocks, block_in_dim, block_out_dim)
        
        
        out = torch.bmm(input_mats, self.weights.transpose(1, 2)) # nr_blocks, batch_size, block_size_out
        
        out = out.transpose(0, 1).reshape(BS, self.out_dim) # nr_blocks, batch_size, block_size

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

        self.inner_dim = out_dim  # projection only works with this somehow
        inner_dim = self.inner_dim
        self.nr_blocks = nr_blocks

        self.fn = nn.Sequential(
            BlockDiagonal(in_dim, inner_dim, nr_blocks, bias=False),  # R
            Permute(inner_dim, nr_blocks),  # P
            BlockDiagonal(inner_dim, out_dim, out_dim // nr_blocks, bias=False),  # L
            Permute(out_dim, out_dim // nr_blocks),  # P^T
        )

        self.bias = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze()
        )

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

        A = fn.weight.reshape(
            nr_blocks, self.out_dim // nr_blocks, nr_blocks, self.in_dim // nr_blocks
        )

        with torch.no_grad():
            for k in range(nr_blocks):
                for j in range(self.out_dim // nr_blocks):
                    M = A[:, j, k, :].contiguous()
                    s = torch.linalg.svd(M)
                    d = math.sqrt(s.S[0])
                    self.fn[0].weights[k, j, :] = d * s.Vh[0, :]
                    self.fn[2].weights[j, :, k] = d * s.U[:, 0]

        self.bias = nn.Parameter(fn.bias.clone().detach())

        self.properly_initialized = True



class Kronecker(Projectable):

    def __init__(self, in_dim, out_dim, factor_1_size):
        super().__init__(in_dim, out_dim)

        d = factor_1_size

        self.f1 = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(d, d))
        )

        self.f2 = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(self.out_dim // d, self.in_dim // d))
        )

        self.bias = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze()
        )
    
    def forward(self, x):

        W = torch.kron(self.f1, self.f2)

        x, shape = to2D(x)
        out = torch.addmm(self.bias, x, W.T)
        return undo_to2D(out, shape)




class TT(Projectable):

    def __init__(self, in_dim, out_dim, nr_cores, rank):
        super().__init__(in_dim, out_dim)

        self.in_core_size = int(round(in_dim ** (1 / nr_cores)))
        self.out_core_size = int(round(out_dim ** (1 / nr_cores)))

        assert self.in_core_size ** nr_cores == in_dim
        assert self.out_core_size ** nr_cores == out_dim

        self.nr_cores = nr_cores
        self.rank = rank

        d_in = self.in_core_size
        d_out = self.out_core_size
        r = self.rank

        self.cores = nn.ParameterList(
            [
                nn.Parameter(self.init_core((1, d_out, d_in, r))), 
                *(
                    nn.Parameter(self.init_core((r, d_out, d_in, r)))
                    for i in range(self.nr_cores - 2)
                ),
                nn.Parameter(self.init_core((r, d_out, d_in, 1))), 
            ]
        )

        self.bias = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze()
        )

    def init_core(self, dims):
        r = self.rank
        c = self.nr_cores
        t = 1 / math.sqrt(
            0.5*(self.in_dim + self.out_dim)
        )  # target std of entire weight matrix, to mimic kaiming_normal
        sigma = (t * (math.sqrt(r) ** (1 - c))) ** (1 / c)
        G = torch.zeros(dims)
        nn.init.normal_(G, std=sigma)
        return init_scale * G

    def to_matrix(self):
        G = None

        if self.nr_cores == 2:
            G = torch.einsum("ainb,bjmc->aijnmc", *self.cores)
        if self.nr_cores == 3:
            G = torch.einsum("ainb,bjmc,ckod->aijknmod", *self.cores)
        if self.nr_cores == 4:
            G = torch.einsum("ainb,bjmc,ckod,dlpe->aijklnmope", *self.cores)

        return G.reshape(self.out_dim, self.in_dim)

    def forward(self, x):

        G = self.to_matrix()

        x, shape = to2D(x)
        out = torch.addmm(self.bias, x, G.T)
        return undo_to2D(out, shape)







def factorize(x):
    best_pair = None
    best_score = float("inf")
    for b in range(1, int(math.isqrt(x)) + 1):
        if x % b == 0:
            a = x // b
            score = abs(a - math.isqrt(x)) + abs(b - math.isqrt(x))
            if score < best_score:
                best_score = score
                best_pair = (a, b)
    return best_pair








class BTT2(Projectable):


    def __init__(self, in_dim, out_dim, rank):
        super().__init__(in_dim, out_dim)

        self.in1, self.in2 = factorize(in_dim)
        self.out1, self.out2 = factorize(out_dim)

        assert self.in1 * self.in2 == in_dim
        assert self.out1 * self.out2 == out_dim


        self.rank = rank

        self.L = nn.Parameter(nn.init.kaiming_uniform_(
            torch.empty(self.out1, self.out2, self.in1, self.rank)
        ))
        self.R = nn.Parameter(nn.init.kaiming_uniform_(
            torch.empty(self.rank, self.out2, self.in1, self.in2)
        ))

        self.bias = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze()
        )

        l = nn.Linear(in_dim, out_dim)
        self.project(l)


    def forward(self, x):
        x, shape = to2D(x)

        W = torch.einsum("abcr,rbcd->abcd", self.L, self.R).reshape(self.out_dim, self.in_dim)

        # out = torch.einsum("sbgd,Bgd->Bsbg", self.R, x.reshape(-1, self.in1, self.in2))
        # out = torch.einsum("Bsbg,abgs->Bab", out, self.L)
        # out = out.reshape(-1, self.out_dim) + self.bias

        out = torch.addmm(self.bias, x, W.T)

        return undo_to2D(out, shape)


    def project(self, fn):

        out_dim, in_dim = fn.weight.shape
        assert self.in_dim == in_dim and self.out_dim == out_dim

        b_out1 = self.out1
        b_out2 = self.out2
        b_in1 = self.in1
        b_in2 = self.in2

        A = fn.weight.reshape(b_out1, b_out2, b_in1, b_in2)

        with torch.no_grad():

            for k in range(b_in1):    
                for j in range(b_out2):   

                    M_blocklike = A[:, j, k, :]

                    U, S, Vh = torch.linalg.svd(M_blocklike, full_matrices=False)

                    d = S[:self.rank].sqrt()       

                    U_r = U[:, :self.rank]         
                    Vh_r = Vh[:self.rank, :]              

                    self.L[:, j, k, :] = U_r * d.unsqueeze(0)  
                    self.R[:, j, k, :] = d.unsqueeze(1) * Vh_r 

            if fn.bias is not None:
                self.bias = nn.Parameter(fn.bias.clone().detach())
            else:
                self.bias = None




class BlockLowRank(Projectable):

    def __init__(self, in_dim, out_dim, rank):
        super().__init__(in_dim, out_dim)

        self.in_core_size  = int(round(in_dim  ** (0.5)))
        self.out_core_size = int(round(out_dim ** (0.5)))

        assert self.in_core_size ** 2  == in_dim
        assert self.out_core_size ** 2 == out_dim


        self.rank = rank

        self.core1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.empty(self.out_core_size, self.out_core_size, self.in_core_size, self.rank)
        ))
        self.core2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.empty(self.out_core_size, self.rank, self.in_core_size, self.in_core_size)
        ))

        self.bias = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze()
        )

    def forward(self, x):
        x, shape = to2D(x)

        W = torch.einsum("iojr,irjn->iojn", self.core1, self.core2)
        W = W.reshape(self.out_dim, self.in_dim)

        out = torch.addmm(self.bias, x, W.T)

        return undo_to2D(out, shape)


    def project(self, fn):

        M = fn.weight

        outs = self.out_core_size
        ins = self.in_core_size

        with torch.no_grad():

            for i in range(self.out_core_size):
                for j in range(self.in_core_size):
                    m = M[i*outs:(i+1)*outs, j*ins:(j+1)*ins]
                    s = torch.linalg.svd(m)
                    d = s.S[:self.rank].sqrt()
                    self.core1[i, :, j, :] = torch.einsum("r,dr->dr", d, s.U[:, :self.rank])
                    self.core2[i, :, j, :] = torch.einsum("r,rd->rd", d, s.Vh[:self.rank, :])
                
            self.bias = nn.Parameter(fn.bias.clone().detach())



class AbstractBlast(Projectable):

    def __init__(self, in_dim, out_dim, b_in, b_out, bs_in, bs_out):
        super().__init__(in_dim, out_dim)


        self.b_in = b_in
        self.b_out = b_out
        self.block_size_in = bs_in
        self.block_size_out = bs_out

        rank = self.rank

        self.S = nn.Parameter(nn.init.uniform_(torch.empty(self.b_out, self.b_in, rank), a=0, b=1))  # just like in paper

        # with S fixed like this, and U, V having entries drawn from Unif(-b, b), we expect a distribution in the manifested matrix of roughly N(0, 4nb^4/27),
        # that is a standart deviation of 2sqrt(n)b^2/(3*sqrt(3)), where n is the rank here
        # to mimic dense, we want that the standart deviation = 1/2 * k, where entries in dense are drawn from Unif(-k, k), k = 1/sqrt(in_dim)
        # therefor we chose
        bound = math.sqrt(math.sqrt(27) / (4 * rank))

        # self.U = nn.Parameter(nn.init.uniform_(torch.empty(b_out, rank, block_size), a=-bound, b=bound))
        # self.Vt = nn.Parameter(nn.init.uniform_(torch.empty(b_in, block_size, rank), a=-bound, b=bound))

        # in the paper:
        self.U = nn.Parameter(
            nn.init.normal_(
                torch.empty(self.b_out, rank, self.block_size_out), mean=0, std=math.sqrt(0.02)
            )
        )
        self.Vt = nn.Parameter(
            nn.init.normal_(
                torch.empty(self.b_in, self.block_size_in, rank), mean=0, std=math.sqrt(0.02)
            )
        )

        self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())

    def weight(self, x):
        BS, M = x.shape
        xp = x.reshape(BS, -1, self.block_size_in).transpose(0, 1)  # b, BS, block_size
        y = torch.bmm(xp, self.Vt)  # b, BS, rank
        z = torch.einsum("ibr,oir->obr", y, self.S)  # b, BS, rank
        out = torch.bmm(z, self.U)  # b, BS, block_size
        out = out.transpose(0, 1).reshape(BS, -1)

        return out


    def forward(self, x):  # much faster
        x, shape = to2D(x)

        W = self.weight(torch.eye(self.in_dim, self.in_dim, device=x.device))

        out = torch.addmm(self.bias, x, W)

        return undo_to2D(out, shape)


    def precGD(self, A):
        out_d, in_d = A.shape
        assert out_d == self.out_dim, in_d == self.in_dim

        delta = 1e-2

        bo = self.block_size_out
        bi = self.block_size_in

        def A_(i, j):
            return A[i*bo:(i+1)*bo, j*bi:(j+1)*bi]

        def U_(i):
            return self.U[i, :, :].T # block_size x rank

        def V_(i):
            return self.Vt[i, :, :] # block_size x rank
        
        def S_(i, j):
            return self.S[i, j, :]

        def U_bar(j):
            return torch.einsum("orb, or -> obr", self.U, self.S[:, j, :]).reshape(-1, self.rank) # out_dim x rank
        
        def V_bar(i): 
            return torch.einsum("ibr, ir -> ibr", self.Vt, self.S[i, :, :]).reshape(-1, self.rank) # in_dim x rank

        def Pu_inv(i):
            vbar = V_bar(i)
            v2 = vbar.T @ vbar
            v2diag = v2.diagonal()
            v2diag += delta
            return v2
        
        def Pv_inv(i):
            ubar = U_bar(i)
            u2 = ubar.T @ ubar
            u2diag = u2.diagonal()
            u2diag += delta
            return u2
        
        def W(i,j):
            u = U_(i)
            v = V_(j)
            return (u.T @ u) * (v.T @ v)
        
        def Ps_inv(i,j):
            w = W(i,j)
            wdiag = w.diagonal()
            wdiag += delta
            return w
               
        def solve2(B, A):
            # compute B @ A.inverse()
            return torch.linalg.solve(A.T, B.T).T

        with torch.no_grad():
            nr_steps = 1500
            for k in range(nr_steps):
                # print(k)
                eta = math.sqrt((nr_steps - k) / nr_steps) * 0.5

                u_new = torch.empty_like(self.U)
                v_new = torch.empty_like(self.Vt)
                s_new = torch.empty_like(self.S)

                try:

                    for i in range(self.b_out):
                        vbar = V_bar(i)
                        u_new[i, :, :] = (
                            U_(i) - eta * solve2(   (U_(i) @ vbar.T - A[i*bo:(i+1)*bo, :]) @ vbar, Pu_inv(i)  )
                        ).T

                    for j in range(self.b_in):
                        ubar = U_bar(j)
                        v_new[j, :, :] = V_(j) - eta * solve2(   (ubar @ V_(j).T - A[:, j*bi:(j+1)*bi]).T @ ubar, Pv_inv(j)   )

                    for i in range(self.b_out): # b_out
                        for j in range(self.b_in): # b_in
                            s_new[i, j, :] = S_(i,j) - eta * torch.linalg.solve(Ps_inv(i, j), S_(i,j) @ W(i,j).T - (u_new[i, :, :] @ A_(i,j) @ v_new[j, :, :]).diag())

                    self.U = nn.Parameter(u_new)
                    self.Vt = nn.Parameter(v_new)
                    self.S = nn.Parameter(s_new)
                
                except:
                    break
    
    
    def project(self, fn):
        self.precGD(fn.weight)
        self.bias = nn.Parameter(fn.bias.clone().detach())

        # self.nr_steps = 100
        # super().project(fn)



class BlastPaper(AbstractBlast): # like in the paper, rectangular blocks
    
    def __init__(self, in_dim, out_dim, nr_blocks, rank):
        assert in_dim % nr_blocks == 0
        assert out_dim % nr_blocks == 0

        self.rank = rank

        super().__init__(in_dim, out_dim, nr_blocks, nr_blocks, in_dim // nr_blocks, out_dim // nr_blocks)



class Blast(AbstractBlast): # all blocks are square, works better for rectangular matrices, square case still the same as paper. 

    def __init__(self, in_dim, out_dim, block_size, rank):
        assert in_dim % block_size == 0
        assert out_dim % block_size == 0

        self.rank = rank

        super().__init__(in_dim, out_dim, in_dim // block_size, out_dim // block_size, block_size, block_size)
