
import torch
from torch import nn
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

import layers
import ViT


device = torch.device("cuda")

# there are models with only square matrices, and models with rectangular weight matrices as well
# you may fail to create a model when the layer type does not support non-square weight matrices 

"""
MLP, with a dense embedding layer (which may do all the work, and may dominate parameter count for sparse weights)
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
def VisionTransformer_noIB(embed_dim, image_dim, patch_size, nr_transformer_blocks, nr_heads, nr_classes, layer_fn, *args, dropout_p=0.2, drop_rate=0.1):
    nr_patches = int(image_dim*image_dim / (patch_size * patch_size))
    
    return nn.Sequential(
        ViT.Patchify(patch_size),
        ViT.Embedding(embed_dim),
        ViT.PosEncoding(embed_dim, nr_patches),
        *(ViT.TransformerBlock_noIB(embed_dim, nr_heads, layer_fn, *args, dropout_p=dropout_p, drop_rate=drop_rate) for _ in range(nr_transformer_blocks)),
        ViT.ClassificationHead(nr_classes)
    )


# https://arxiv.org/abs/2010.11929
"""
a more standart Vision Transformer
"""
def VisionTransformer(embed_dim, image_dim, patch_size, nr_transformer_blocks, nr_heads, nr_classes, layer_fn, *args, dropout_p=0.2, drop_rate=0.1):
    nr_patches = int(image_dim*image_dim / (patch_size * patch_size))
    
    return nn.Sequential(
        ViT.Patchify(patch_size),
        ViT.Embedding(embed_dim),
        ViT.PosEncoding(embed_dim, nr_patches),
        *(ViT.TransformerBlock(embed_dim, nr_heads, layer_fn, *args, dropout_p=dropout_p, drop_rate=drop_rate) for _ in range(nr_transformer_blocks)),
        ViT.ClassificationHead(nr_classes)
    )


# replace Conv1D in gpt models with equivalent nn.linear
def replace_conv1d(model):
    for name, child in list(model.named_children()):
        if isinstance(child, Conv1D):

            in_dim, out_dim = child.weight.shape

            assert child.bias is not None

            l = nn.Linear(in_dim, out_dim, device=device)
            l.weight = nn.Parameter(child.weight.transpose(0, 1).clone().detach())
            l.bias = nn.Parameter(child.bias.clone().detach())

            model._modules[name] = l
        else:
            replace_conv1d(child)


def replace_layers(module: nn.Module, layer_fn, *args):

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name != "lm_head":
            # print(name)
            out_dim, in_dim = child.weight.shape
            l = layer_fn(in_dim, out_dim, *args).to(device)
            l.project(child)

            
            # new_layer = replacement_cls(in_f, out_f, bias=bias)
            module._modules[name] = l
        else:
            replace_layers(child, layer_fn, *args)



def gpt2_model(model_name, layer_fn, *args):

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    replace_conv1d(model)

    if layer_fn is not layers.Dense:
        replace_layers(model, layer_fn, *args)

    return model


"""
pretrained smaller gpt2 model 82M
"""
def distilGPT2(layer_fn, *args):
    return gpt2_model("distilgpt2", layer_fn, *args)

"""
pretrained gpt2-small model 124M
"""
def GPT2(layer_fn, *args):
    return gpt2_model("gpt2", layer_fn, *args)




