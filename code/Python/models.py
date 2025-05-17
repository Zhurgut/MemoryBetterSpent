

from torch import nn

import layers
import ViT


# there are models with only square matrices, and models with rectangular weight matrices as well
# you may fail to create a model when the layer type does not support non-square weight matrices 

"""
MLP, with a dense embedding layer (which may do all the work, and dominate parameter count for sparse weights)
"""
def MLP_DE(in_dim, nr_blocks, width, out_dim, layer_fn, *args, p=0.2):

    def Block():
        return nn.Sequential(
            nn.LayerNorm((width,)),
            layer_fn(width, width, *args), 
            nn.ReLU(),
            nn.Dropout(p=p)
        )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, width),
        nn.ReLU(),
        nn.Dropout(p=p),
        *[Block() for i in range(nr_blocks)],
        nn.Linear(width, out_dim)
    )

    return model


"""
MLP, with a sparse embedding layer
"""
def MLP(in_dim, nr_blocks, width, out_dim, layer_fn, *args, p=0.2):

    def Block():
        return nn.Sequential(
            nn.LayerNorm((width,)),
            layer_fn(width, width, *args), 
            nn.ReLU(),
            nn.Dropout(p=p)
        )

    model = nn.Sequential(
        nn.Flatten(),
        layer_fn(in_dim, width, *args),
        nn.ReLU(),
        nn.Dropout(p=p),
        *[Block() for i in range(nr_blocks)],
        nn.Linear(width, out_dim)
    )

    return model


# https://arxiv.org/abs/2306.13575
"""
B-MLP, with inverted bottlenecks and skip connections
"""
def B_MLP(in_dim, nr_blocks, width, out_dim, layer_fn, *args, p=0.2):
    
    k = 4
    def Block():
        return layers.SkipConnection(nn.Sequential(
            nn.LayerNorm((width,)), 
            layer_fn(width, k * width, *args), 
            nn.ReLU(),
            nn.Dropout(p=p),
            layer_fn(k * width, width, *args), 
        ))

    model = nn.Sequential(
        nn.Flatten(),
        layer_fn(in_dim, width, *args),
        nn.ReLU(),
        nn.Dropout(p=p),
        *[Block() for i in range(nr_blocks)],
        nn.Linear(width, out_dim)
    )

    return model


"""
MLP with dense embedding layer, skip connections, and no inverted bottlenecks
"""
def B_MLP_noIB(in_dim, nr_blocks, width, out_dim, layer_fn, *args, p=0.2):
    
    def Block():
        return layers.SkipConnection(nn.Sequential(
            nn.LayerNorm((width,)), 
            layer_fn(width, width, *args), 
            nn.ReLU(),
            nn.Dropout(p=p),
            layer_fn(width, width, *args), 
        ))

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, width),
        nn.ReLU(),
        nn.Dropout(p=p),
        *[Block() for i in range(nr_blocks)],
        nn.Linear(width, out_dim)
    )

    return model



"""
ViT without inverse bottleneck (IB), instead use square layers
"""
def VisionTransformer_noIB(embed_dim, patch_size, nr_transformer_blocks, nr_heads, nr_classes, layer_fn, *args, p=0.2):
    nr_patches = int(32*32 / (patch_size * patch_size))
    
    return nn.Sequential(
        ViT.Patchify(patch_size),
        ViT.Embedding(embed_dim),
        ViT.PosEncoding(embed_dim, nr_patches),
        *(ViT.TransformerBlock_noIB(embed_dim, nr_heads, layer_fn, *args, p=p) for _ in range(nr_transformer_blocks)),
        ViT.ClassificationHead(nr_classes)
    )


# https://arxiv.org/abs/2010.11929
"""
a more standart Vision Transformer
"""
def VisionTransformer(embed_dim, patch_size, nr_transformer_blocks, nr_heads, nr_classes, layer_fn, *args, p=0.2):
    nr_patches = int(32*32 / (patch_size * patch_size))
    
    return nn.Sequential(
        ViT.Patchify(patch_size),
        ViT.Embedding(embed_dim),
        ViT.PosEncoding(embed_dim, nr_patches),
        *(ViT.TransformerBlock(embed_dim, nr_heads, layer_fn, *args, p=p) for _ in range(nr_transformer_blocks)),
        ViT.ClassificationHead(nr_classes)
    )