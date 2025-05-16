import torch
import torch.nn as nn

import layers




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
    






def TransformerBlock(embed_dim, nr_heads, layer_fn, *args, p=0.2):

    def mlp(embed_dim, layer_fn, *args):
        k = 4
        return layers.SkipConnection(
            nn.Sequential(
                nn.LayerNorm(embed_dim), 
                layer_fn(embed_dim, k * embed_dim, *args),
                # nn.GELU(),
                nn.ReLU(),
                nn.Dropout(p=p),
                layer_fn(k * embed_dim, embed_dim, *args),
                # nn.Dropout(p=p),
            )
        )

    return nn.Sequential(
        layers.SkipConnection(
            nn.Sequential(
                nn.LayerNorm(embed_dim), 
                MultiHeadAttention(embed_dim, nr_heads, layer_fn, *args)
            )
        ),
        mlp(embed_dim, layer_fn, *args, p=p)
    )





def TransformerBlock_noIB(embed_dim, nr_heads, layer_fn, *args, p=0.2):
    
    def mlp_noIB(embed_dim, layer_fn, *args):
        return layers.SkipConnection(
            nn.Sequential(
                nn.LayerNorm(embed_dim), 
                layer_fn(embed_dim, embed_dim, *args),
                # nn.GELU(),
                nn.ReLU(),
                nn.Dropout(p=p),
                layer_fn(embed_dim, embed_dim, *args),
                # nn.Dropout(p=p),
            )
        )
    
    return nn.Sequential(
        layers.SkipConnection(
            nn.Sequential(
                nn.LayerNorm(embed_dim), 
                MultiHeadAttention(embed_dim, nr_heads, layer_fn, *args)
            )
        ),
        mlp_noIB(embed_dim, layer_fn, *args, p=p)
    )