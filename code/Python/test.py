
from main import VisionTransformer, nr_parameters
import torch
import layers
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_vit(embed_dim, patch_size, nr_blocks, nr_heads, layer_fn, *args):
    model = VisionTransformer(embed_dim, patch_size, nr_blocks, nr_heads, 10, layer_fn, *args).to(device)
    print("patch-size: ", patch_size)
    
    gc.collect()
    torch.cuda.empty_cache()
    model.train(False)
    bs = 64
    try:
        with torch.no_grad():
            while True:
                x = torch.randn(bs, 3, 32, 32).to(device)
                y = model(x)
                bs += 64
    except:
        print("final eval size: ", bs)
    
    gc.collect()
    torch.cuda.empty_cache()
    model.train(True)
    bs = 16
    try:
        while True:
            x = torch.randn(bs, 3, 32, 32).to(device)
            y = model(x)
            bs += 16
    except:
        print("final train size: ", bs)
    
    print(nr_parameters(model), " parameters")
    

nr_transformer_blocks = 6
nr_heads = 6
head_dim = 64

# test_vit(nr_heads * head_dim, 16, nr_transformer_blocks, nr_heads, layers.Dense)
# test_vit(nr_heads * head_dim, 8,  nr_transformer_blocks, nr_heads, layers.Dense)
# test_vit(nr_heads * head_dim, 4,  nr_transformer_blocks, nr_heads, layers.Dense)
# test_vit(nr_heads * head_dim, 2,  nr_transformer_blocks, nr_heads, layers.Dense)

test_vit(nr_heads * head_dim, 4,  nr_transformer_blocks, nr_heads, layers.Monarch, 4)
test_vit(nr_heads * head_dim, 4,  nr_transformer_blocks, nr_heads, layers.Monarch, 64)
