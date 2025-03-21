import torch
import torch.nn.functional as F
from layers import *
from main import nr_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# s = 64
# x = torch.rand(5, s)
# l = [Dense(s), LowRank(s, 7), Monarch(s, 4), TT(s, 2, 3), BTT(s, 2, 3)]
# y = [f(x) for f in l]
# d = [to_dense(f)[2] for f in l]
# y2 = [f(x) for f in d]
# e = [(y[i] - y2[i]).norm().item() for i in range(len(y))]
# print(e)


def spectral_norm(matrix):
    return torch.linalg.svdvals(matrix)[0].item()


def optimize(target, size, X, lr, nr_steps, layer_fn, *args, p=2):

    model = layer_fn(size, *args).to(device)

    model.bias = nn.Parameter(target.bias.detach().clone())
    model.bias.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=nr_steps, eta_min=lr / 100
    )

    def loss(x, y):
        d = x - y
        sing_vs = d.norm(p=2, dim=1)
        # print(sing_vs.max())
        return sing_vs.norm(p=p)

    M = target(X).detach()

    for i in range(nr_steps):
        opt.zero_grad()

        L = loss(M, model(X))

        if i % 200 == 0:
            print(i, "; ", L.item())

        L.backward()

        opt.step()
        scheduler.step()

    model.bias.requires_grad = True

    I = torch.eye(size, size).to(device)

    return spectral_norm(target(I) - model(I))


def project(matrix, nr_runs, layer_fn, *args, nr_steps, X=None, p=2):

    out_dim, in_dim = matrix.shape
    assert out_dim == in_dim  # for now
    d = out_dim
    bias = torch.zeros(d)

    X = (
        F.normalize(torch.randn(10 * d, d), p=2, dim=1).to(device)
        if X is None
        else X.to(device)
    )

    target = Dense(d)
    target.weight = nn.Parameter(matrix)
    target.bias = nn.Parameter(bias)
    target = target.to(device)

    losses = [
        optimize(target, d, X, 1e-3, nr_steps, layer_fn, *args, p=p)
        for i in range(nr_runs)
    ]
    print(losses)
    return min(losses), nr_parameters(layer_fn(d, *args))


def random_matrix_uniform(size):
    return torch.rand(size, size) - 0.5


def random_matrix_normal(size):
    return torch.randn(size, size)


def random_matrix_1(size):
    m = torch.randn(size, size)
    m = m.abs().exp() * m.sign() / 25
    return m


def collect256():
    size = 256
    args_lowrank = [128, 64, 32, 16, 8, 4]
    args_monarch = [2, 4, 8, 16, 32, 64]
    args_tt = [(2, 16), (2, 32), (2, 64), (2, 128)]
    args_tt2 = [(4, 8), (4, 16), (4, 32), (4, 44)]
    args_btt = [(2, 1), (2, 2), (2, 4), (2, 8)]
    args_btt2 = [(4, 1), (4, 2), (4, 4), (4, 5)]

    M = random_matrix_1(256)
    X = F.normalize(torch.randn(10 * size, size), p=2, dim=1)

    nr_runs = 5
    print("dense")
    dense = project(M, nr_runs, Dense, nr_steps=1000, X=X)
    print(dense)

    print("lowrank")
    lowrank = [
        project(M, nr_runs, LowRank, args_lowrank[i], nr_steps=1000, X=X)
        for i in range(len(args_lowrank))
    ]
    print(lowrank)

    print("monarch")
    monarch = [
        project(M, nr_runs, Monarch, args_monarch[i], nr_steps=1000, X=X)
        for i in range(len(args_monarch))
    ]
    print(monarch)

    print("tt")
    tt1 = [
        project(M, nr_runs, TT, *(args_tt[i]), nr_steps=1000, X=X)
        for i in range(len(args_tt))
    ]
    print(tt1)

    print("tt")
    tt2 = [
        project(M, nr_runs, TT, *(args_tt2[i]), nr_steps=1000, X=X)
        for i in range(len(args_tt2))
    ]
    print(tt2)

    print("btt")
    btt1 = [
        project(M, nr_runs, BTT, *(args_btt[i]), nr_steps=1000, X=X)
        for i in range(len(args_btt))
    ]
    print(btt1)

    print("btt")
    btt2 = [
        project(M, nr_runs, BTT, *(args_btt2[i]), nr_steps=1000, X=X)
        for i in range(len(args_btt2))
    ]
    print(btt2)


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
