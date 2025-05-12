
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
nr_heads = 8
head_dim = 128

# test_vit(nr_heads * head_dim, 16, nr_transformer_blocks, nr_heads, layers.Dense)
# test_vit(nr_heads * head_dim, 8,  nr_transformer_blocks, nr_heads, layers.Dense)
# test_vit(nr_heads * head_dim, 4,  nr_transformer_blocks, nr_heads, layers.Dense)
test_vit(nr_heads * head_dim, 4,  nr_transformer_blocks, nr_heads, layers.LowRankLight, 73)
# test_vit(nr_heads * head_dim, 2,  nr_transformer_blocks, nr_heads, layers.Dense)

# test_vit(nr_heads * head_dim, 4,  nr_transformer_blocks, nr_heads, layers.Monarch, 4)
# test_vit(nr_heads * head_dim, 4,  nr_transformer_blocks, nr_heads, layers.Monarch, 64)


nr_transformer_blocks = 6
nr_heads = 6
head_dim = 64

# test_vit(nr_heads * head_dim, 16, nr_transformer_blocks, nr_heads, layers.Dense)
# test_vit(nr_heads * head_dim, 8,  nr_transformer_blocks, nr_heads, layers.Dense)
# test_vit(nr_heads * head_dim, 4,  nr_transformer_blocks, nr_heads, layers.Dense)



# from projection import *

# m = torch.diag(torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))
# m = torch.zeros(9, 9)
# for i in range(9):
#     for j in range(i, 9):
#         m[i, j] = 1
#         m[j, i] = -1

# m[3, 6] = 0
# m[0, 4] = 0
# m[5, 3] = 0

# m[3, 6] = 1

# m[0, 0] = 2
# m[0, 7] = 1
# m[0, 5] = 0.3

# m = torch.randn(9, 9)


# torch.set_printoptions(sci_mode=False, linewidth=200, precision=2)

# for rank in range(1, 4):
    
#     print(rank)
    
#     # target = Dense(9)
#     # target.weight = nn.Parameter(m.T)
#     # target.bias = nn.Parameter(torch.zeros(9))
#     # X = F.normalize(torch.randn(100, 9), p=2, dim=1)
    
#     # model = optimize(target.to(device), 9, X.to(device), 1e-3, 8000, BTT, 2, rank)[1]

#     # rank = 3

#     # print(m)

#     btt = BTT.from_matrix(m, rank)

#     btt.bias = nn.Parameter(torch.zeros(9))

#     out = btt(torch.eye(9, 9))
    
#     # print("optimal cores:")
#     # print(model.cores[0].reshape(3, 3, 3, rank))
#     # print(model.cores[1].reshape(3, 3, 3, rank))
#     # print("projected cores:")
#     # print(btt.cores[0].reshape(3, 3, 3, rank))
#     # print(btt.cores[1].reshape(3, 3, 3, rank))
#     # # print(btt.cores[0].reshape(3, 3, 3, rank), btt.cores[1].reshape(3, 3, 3, rank))
#     # print("target matrix:")
#     # print(m)
#     # print("projected matrix:")
#     # print(out)
#     # print("optimal matrix")
#     # print(model(torch.eye(9, 9).to(device)))

#     print((out - m).reshape(-1).norm())