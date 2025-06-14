import torch
import torch.nn.functional as F
from layers import *
from main import nr_parameters
from matrices import *

import os
import csv
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# s = 64
# x = torch.rand(5, s)
# l = [Dense(s), LowRank(s, 7), Monarch(s, 4), TT(s, 2, 3), BTT(s, 2, 3)]
# y = [f(x) for f in l]
# d = [to_dense(f)[2] for f in l]
# y2 = [f(x) for f in d]
# e = [(y[i] - y2[i]).norm().item() for i in range(len(y))]
# print(e)


def project(matrix, nr_runs, layer_fn, *args, nr_steps):

    out_dim, in_dim = matrix.shape
    assert out_dim == in_dim  # for now

    fn = nn.Linear(in_dim, out_dim)
    fn.weight = nn.Parameter(matrix.clone().detach())
    fn.bias = nn.Parameter(torch.zeros(out_dim))
    fn = fn.to(device)

    def loss(layer_fn, *args):
        l = layer_fn(in_dim, out_dim, *args).to(device)
        l.nr_steps = nr_steps
        l.project(fn)
        x = torch.eye(in_dim, in_dim, device=device)
        return torch.linalg.norm(l(x) - fn(x), ord="fro")

    losses = [
        loss(layer_fn, *args).item()
        for i in range(nr_runs)
    ]
    print(losses)
    return min(losses), nr_parameters(layer_fn(in_dim, out_dim, *args))


def save_results(file_name, columns, *data_rows):
    WD = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    timestamp = datetime.now().strftime("%m%d_%H%M")
    os.makedirs("../../measurements/projection", exist_ok=True)
    path = os.path.join("../../measurements/projection", f"{file_name}_{timestamp}.csv")

    # Write data to CSV
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)  # Write headers
        for rows in data_rows:
            writer.writerows(rows)  # Write data
    
    os.chdir(WD)


def collect256(M, nr_runs, nr_steps):
    size = 256
    
    assert M.shape == (size, size)
    
    args_unstructured = [0.99, 0.97, 0.93, 0.88, 0.83, 0.75, 0.5, 0.3, 0.05]
    args_lowrank = [128, 96, 64, 48, 32, 16, 8, 4]
    args_lowrank_light = [255, 224, 192, 160, 128, 64, 32, 16, 8, 4, 1]
    args_monarch = [2, 4, 8, 16, 32, 64]
    # args_tt = [(2, 16), (2, 32), (2, 48), (2, 64), (2, 96), (2, 128)]
    # args_tt2 = [(4, 8), (4, 16), (4, 32), (4, 44)]
    # args_btt = [(2, 1), (2, 2), (2, 4), (2, 6), (2, 8)]
    # args_bttlight = [1, 4, 8, 12, 14, 15, 16]
    # args_btt2 = [(4, 1), (4, 2), (4, 4), (4, 5)]
    args_blocksparse = [(16, 1), (16, 2), (16, 3), (16, 4), (16, 6), (16, 11), (16, 15)]
    
    def add_label(label, data):
        return [(label, score, nparams) for (score, nparams) in data]
    
    # args_unstructured = [0.95, 0.7, 0.4]
    # args_lowrank = [128, 64]
    # args_monarch = [2, 8]
    # args_blr     = [(4, 2), (4, 1)]
    # args_tt = [(2, 128)]
    # args_tt2 = [(4, 44)]
    # args_btt = [(2, 8), (2, 4)]
    # args_btt2 = [(4, 5)]
    args_blocksparse = [(16, 2), (16, 12)]

    
    # dense = ("dense", *project(M, 1, Dense, nr_steps=nr_steps, X=X, p=p))
    
    
    # blocks = [
    #     project(M, nr_runs, BlockSparse, *(args_blocksparse[i]), nr_steps=nr_steps, X=X, p=p)
    #     for i in range(len(args_blocksparse))
    # ]
    # blocks = add_label("block_sparse", blocks)
    
    
    # def from_block_mag_pruned(size, nr_blocks_per_row, nr_blocks_to_drop):
    #     return BlockSparse.from_mag_pruned(M, nr_blocks_per_row, nr_blocks_to_drop)

    # block_mag_prune = [
    #     project(M, 1, from_block_mag_pruned, *(args_blocksparse[i]), nr_steps=nr_steps, X=X, p=p)
    #     for i in range(len(args_blocksparse))
    # ]
    # block_mag_prune = add_label("block_mag_pruned", block_mag_prune)
    
    
    
    # unstructured = [
    #     project(M, nr_runs, Unstructured, args_unstructured[i], nr_steps=nr_steps, X=X, p=p)
    #     for i in range(len(args_unstructured))
    # ]
    # unstructured = add_label("unstructured", unstructured)
    
    print("unstructured")
    mag_prune = [
        project(M, 1, Unstructured, args_unstructured[i], nr_steps=nr_steps)
        for i in range(len(args_unstructured))
    ]
    mag_prune = add_label("Unstructured, magnitude-pruned", mag_prune)
    
    print("lowrank")
    lowrank = [
        project(M, 1, LowRank, args_lowrank[i], nr_steps=nr_steps)
        for i in range(len(args_lowrank))
    ]
    lowrank = add_label("lowrank", lowrank)
    

    svd = torch.linalg.svdvals(M)
    lowrank_opt = [("LowRankOpt", sum(svd[rank:].pow(2)).sqrt().item(), nr_parameters(LowRank(size, size, rank))) for rank in args_lowrank] # the indeces actually work out like this, because zero based indexing into array, but ranks start at 1
    
    lowranklight_opt = [("lowrank_light_opt", sum(svd[rank:].pow(2)).sqrt().item(), 2*size*rank - rank*rank + size) for rank in args_lowrank_light]
    
    print("lowranklight")
    lowrank_light = [
        project(M, 1, LowRankLight, args_lowrank_light[i], nr_steps=nr_steps)
        for i in range(len(args_lowrank_light))
    ]
    lowrank_light = add_label("lowrank_light", lowrank_light)
    
    print("monarch")
    monarch = [
        project(M, nr_runs, Monarch, args_monarch[i], nr_steps=nr_steps)
        for i in range(len(args_monarch))
    ]
    monarch = add_label("monarch", monarch)
    
    
    # blr = [
    #     project(M, nr_runs, BlockLowRank, *(args_blr[i]), nr_steps=nr_steps, X=X, p=p)
    #     for i in range(len(args_blr))
    # ]
    # blr = add_label("block-lowrank", blr)

    
    # tt1 = [
    #     project(M, nr_runs, TT, *(args_tt[i]), nr_steps=nr_steps, X=X, p=p)
    #     for i in range(len(args_tt))
    # ]
    # tt1 = add_label("tt-2cores", tt1)
    

    
    # tt2 = [
    #     project(M, nr_runs, TT, *(args_tt2[i]), nr_steps=nr_steps, X=X, p=p)
    #     for i in range(len(args_tt2))
    # ]
    # tt2 = add_label("tt-4cores", tt2)

    
    

    # def btt_projected(size, nr_cores, rank):
    #     return BTT.from_matrix(M.T, rank)

    # btt_opt = [
    #     project(M, 1, btt_projected, *(args_btt[i]), nr_steps=0, X=X, p=p)
    #     for i in range(len(args_btt))
    # ]
    # btt_opt = add_label("BTT", btt_opt)


    
    # bttlight = [
    #     project(M, nr_runs, BTTLight, args_bttlight[i], nr_steps=nr_steps, X=X, p=p)
    #     for i in range(len(args_bttlight))
    # ]
    # bttlight = add_label("BTT Light", bttlight)
    
    # btt_uopt = [
    #     project(M, nr_runs, btt_projected, *(args_btt[i]), nr_steps=0, X=X, p=p)
    #     for i in range(len(args_btt))
    # ]
    # btt_uopt = add_label("btt_opt_no_optimization", btt_uopt)
    
    
    # btt1 = [
    #     project(M, nr_runs, BTT, *(args_btt[i]), nr_steps=nr_steps, X=X, p=p)
    #     for i in range(len(args_btt))
    # ]
    # btt1 = add_label("btt-2cores", btt1)
    

    
    # btt2 = [
    #     project(M, nr_runs, BTT, *(args_btt2[i]), nr_steps=nr_steps, X=X, p=p)
    #     for i in range(len(args_btt2))
    # ]
    # btt2 = add_label("btt-4cores", btt2)
    
    save_results("p256", ["layer", "op_norm", "nr_parameters"], mag_prune, monarch, lowrank, lowrank_opt, lowrank_light, lowranklight_opt)
    


# def collect729(M, nr_runs, nr_steps, p=4):
#     size = 729
    
#     assert M.shape == (size, size)
#     args_lowrank = [364, 291, 218, 145, 73, 32, 16, 4]
#     args_monarch = [3, 9, 27, 81, 243]
#     args_tt = [(2, 364), (2, 291), (2, 218), (2, 145), (2, 73), (2, 32), (2, 16), (2, 4)]
#     args_tt2 = [(3, 80), (3, 60), (3, 40), (3, 20)]
#     args_btt = [(2, 13), (2, 10), (2, 7), (2, 5), (2, 3)]
#     args_btt2 = [(3, 8), (3, 6), (3, 4), (3, 2)]
    
#     def add_label(label, data):
#         return [(label, score, nparams) for (score, nparams) in data]

#     X = F.normalize(torch.randn(50 * size, size), p=2, dim=1)

    
#     dense = ("dense", *project(M, 1, Dense, nr_steps=2*nr_steps, X=X, p=p))
    

    
#     lowrank = [
#         project(M, nr_runs, LowRank, args_lowrank[i], nr_steps=nr_steps, X=X, p=p)
#         for i in range(len(args_lowrank))
#     ]
#     lowrank = add_label("lowrank", lowrank)
    
#     monarch = [
#         project(M, nr_runs, Monarch, args_monarch[i], nr_steps=nr_steps, X=X, p=p)
#         for i in range(len(args_monarch))
#     ]
#     monarch = add_label("monarch", monarch)

    
#     tt1 = [
#         project(M, nr_runs, TT, *(args_tt[i]), nr_steps=nr_steps, X=X, p=p)
#         for i in range(len(args_tt))
#     ]
#     tt1 = add_label("tt-2cores", tt1)
    

    
#     tt2 = [
#         project(M, nr_runs, TT, *(args_tt2[i]), nr_steps=nr_steps, X=X, p=p)
#         for i in range(len(args_tt2))
#     ]
#     tt2 = add_label("tt-3cores", tt2)
    
    
    
#     btt1 = [
#         project(M, nr_runs, BTT, *(args_btt[i]), nr_steps=nr_steps, X=X, p=p)
#         for i in range(len(args_btt))
#     ]
#     btt1 = add_label("btt-2cores", btt1)
    

    
#     btt2 = [
#         project(M, nr_runs, BTT, *(args_btt2[i]), nr_steps=nr_steps, X=X, p=p)
#         for i in range(len(args_btt2))
#     ]
#     btt2 = add_label("btt-3cores", btt2)
    
#     save_results("p729", ["layer", "op_norm", "nr_parameters"], [dense], lowrank, monarch, tt1, tt2, btt1, btt2)
    
    
    
collect256(model_matrix(256, 2), 2, 5000)
# collect256(random_matrix_normal(256), 3, 5000)
# collect729(model_matrix(729, 3), 3, 5000)

# project(torch.randn(64, 64), torch.rand(64), 1, LowRank, 48)
# m = to_dense(Monarch(64, 4))[0]
# project(m.T, torch.rand(64), 1, Monarch, 4)
# project(torch.randn(64, 64), 1, Monarch, 4)


# def project(model: torch.nn.Linear, lr, nr_steps, layer_fn, *args):

#     out_dim, in_dim = model.weight.shape
#     assert out_dim == in_dim # for now
#     d = out_dim
#     fn = layer_fn(d, *args).to(device)
#     X = F.normalize(torch.randn(10*d, d), p=2, dim=1).to(device)
#     model = model.to(device)
#     M = model(X).detach()

#     fn[2].bias = nn.Parameter(fn[1](model.bias))
#     fn[2].bias.requires_grad = False

#     def loss(a, b):
#         diff = a - b
#         return (diff * diff).mean()

#     opt = torch.optim.Adam(fn.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nr_steps, eta_min=lr/10)

#     for i in range(nr_steps):
#         opt.zero_grad()

#         L = loss(M, fn(X))
#         if i < 10 or i % 100 == 0:
#             print(i, "; ", L.item())

#         L.backward()

#         opt.step()
#         scheduler.step()


#     print(model.bias)
#     print(fn[3](fn[2].bias))

#     fn[2].bias.requires_grad = True

# l = to_dense(Monarch(64, 2))[2]
# project(l, 1e-2, 1000, Monarch, 64)
