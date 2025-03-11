
import torch
from torch import nn
import numpy as np
from math import sqrt



init_scale = 1.0 # change init scale for Monarch, TT

def set_scale(s):
    global init_scale
    init_scale = s


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

        self.weights = nn.Parameter(init_scale * w)
        
        b = torch.zeros(size)
        nn.init.uniform_(b, -1, 1)
        self.bias   = nn.Parameter(np.sqrt(1/size) * b)
    
    def forward(self, x):
        z = x.reshape(-1, self.size)
        batch_size, n = z.shape
        
        input_mats = z.reshape(batch_size, self.nr_blocks, self.block_size).transpose(0, 1) # (nr_blocks, batch_size, block_size) * weights: (nr_blocks, block_size, block_size)
        # 
        out = torch.bmm(input_mats, self.weights) # nr_blocks, batch_size, block_size
        
        return (out.transpose(0, 1).reshape(batch_size, n) + self.bias).reshape(x.shape)

    



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


          
    def forward(self, x):
        # manifest the matrix, then matmul, much faster than the "efficient" way, probably because large batch size 
        G = self.cores[0][0, :, :, :]
        for core in self.cores[1:]:
            G = torch.einsum("...a,aijb->...ijb", G, core)

        G = G.reshape(self.size, self.size)

        return (torch.matmul(x.reshape(-1, self.size), G) + self.bias).reshape(x.shape)
    





def Kronecker(size):
    return TT(size, 2, 1)



class BTT(nn.Module):
    
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
    

    def init_core(self, dims):
        r = self.rank
        c = self.nr_cores
        t = 1/sqrt(self.size) # target std of entire weight matrix, to mimic kaiming_normal
        sigma = (t * (sqrt(r) ** (1-c))) ** (1/c)
        G = torch.zeros(dims)
        nn.init.normal_(G, std=sigma)
        return init_scale * G


          
    def forward(self, x):
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
            G = torch.einsum("jxicb,jyiba->jxyica", core.reshape(j, d, i, *core.shape[-2:]), G.reshape(j, d ** (idx + 1), i, *G.shape[-2:]))

        G = G.reshape(self.size, self.size)

        return (torch.matmul(x.reshape(-1, self.size), G) + self.bias).reshape(x.shape)


class SkipConnection(nn.Module):
    
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)


def Patchify(patch_size):
    return nn.Unfold((patch_size, patch_size), stride=patch_size)   



class Embedding(nn.Module):
    
    def __init__(self, embed_dim):
        super().__init__()
        self.out_dim = embed_dim
        self.embed = nn.LazyLinear(self.out_dim)
        
    def forward(self, x):
        BS, d, nr_patches = x.shape
        return self.embed(x.transpose(1, 2)) # -> (BS, nr_patches, embed_dim)


class PosEncoding(nn.Module):
    
    "also adds the CLS token"
    
    def __init__(self, embed_dim, nr_patches):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.nr_patches = nr_patches
        
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.positional_encoding = nn.Parameter(torch.randn(nr_patches + 1, embed_dim))


    def forward(self, x):
        BS, nr_patches, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        return torch.cat([self.cls.expand(BS, -1, -1), x], dim=1) + self.positional_encoding



class MultiHeadAttention(nn.Module):
    
    def __init__(self, embed_dim, nr_heads, layer_fn, *args):
        super().__init__()
        
        assert embed_dim % nr_heads == 0
        
        self.nr_heads = nr_heads
        
        self.Q = layer_fn(embed_dim, *args)
        self.K = layer_fn(embed_dim, *args)
        self.V = layer_fn(embed_dim, *args)
        
    
    def forward(self, x):
        BS, seq_len, embed_dim = x.shape
        q = self.Q(x.reshape((-1, embed_dim))).reshape(BS, seq_len, self.nr_heads, -1).transpose(1, 2)
        k = self.K(x.reshape((-1, embed_dim))).reshape(BS, seq_len, self.nr_heads, -1).transpose(1, 2)
        v = self.V(x.reshape((-1, embed_dim))).reshape(BS, seq_len, self.nr_heads, -1).transpose(1, 2)
        
        out = nn.functional.scaled_dot_product_attention(q, k, v) # BS, nr_heads, seq_len, head_dim
        
        return out.transpose(1, 2).reshape(x.shape) # BS, seq_len, embed_dim 



class ClassificationHead(nn.Module):
    
    def __init__(self, nr_classes):
        super().__init__()
        
        self.fn = nn.LazyLinear(nr_classes)
    
    def forward(self, x):
        BS, seq_len, embed_dim = x.shape
        
        return self.fn(x[:, 0, :])
    



        

def TransformerBlock(embed_dim, nr_heads, layer_fn, *args):
    return nn.Sequential(
        SkipConnection(
            nn.Sequential(
                nn.LayerNorm(embed_dim), 
                MultiHeadAttention(embed_dim, nr_heads, layer_fn, *args)
            )
        ),
        SkipConnection(
            nn.Sequential(
                nn.LayerNorm(embed_dim), 
                layer_fn(embed_dim, *args),
                # nn.GELU(),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                layer_fn(embed_dim, *args),
                nn.Dropout(p=0.2),
            )
        )
    )


