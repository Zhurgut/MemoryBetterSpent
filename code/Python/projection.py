
import torch
import torch.nn.functional as F
from layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# s = 64
# x = torch.rand(5, s)
# l = [Dense(s), LowRank(s, 7), Monarch(s, 4), TT(s, 2, 3), BTT(s, 2, 3)]
# y = [f(x) for f in l]
# d = [to_dense(f)[2] for f in l]
# y2 = [f(x) for f in d]
# e = [(y[i] - y2[i]).norm().item() for i in range(len(y))]
# print(e)

def optimize(target, size, X, lr, nr_steps, layer_fn, *args, p=2):
    
    model = layer_fn(size, *args).to(device)
    
    model.bias = nn.Parameter(target.bias.detach().clone())
    model.bias.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nr_steps, eta_min=lr/100)
    
    def loss(x, y):
        # d = x - y
        # return (d*d).mean()
        return (x-y).norm(p=2, dim=1).norm(p=p)
    
    M = target(X).detach()
    min_loss = loss(M, model(X))
    
    for i in range(nr_steps):
        opt.zero_grad()
            
        L = loss(M, model(X))
        min_loss = min(min_loss, L.item())
        
        if i < 10 or i % 100 == 0:
            print(i, "; ", L.item())
        
        L.backward()
        
        opt.step()
        scheduler.step()

    model.bias.requires_grad = True
    
    return min_loss




def project(matrix, nr_runs, layer_fn, *args):
        
    out_dim, in_dim = matrix.shape
    assert out_dim == in_dim # for now
    d = out_dim
    bias = torch.zeros(d)
    
    
    X = F.normalize(torch.randn(10*d, d), p=2, dim=1).to(device)
    
    target = Dense(d)
    target.weight = nn.Parameter(matrix)
    target.bias   = nn.Parameter(bias)
    target = target.to(device)

    
    losses = [optimize(target, d, X, 1e-3, 5000, layer_fn, *args) for i in range(nr_runs)]
    print(losses)
    return min(losses)


def spectral_norm(matrix, nr_iters=100, nr_starters=4):
    d = matrix.shape[0]
    x = F.normalize(torch.randn(nr_starters, d, device=device), dim=1)  # Random initial vectors

    for _ in range(nr_iters):
        x = x @ matrix.T
        x = x @ matrix
        x = torch.linalg.qr(x)[0]
        # x = F.normalize(x, dim=1)
        print(torch.norm(x @ matrix.T, dim=1))

    return torch.norm(x @ matrix.T, dim=1)


def random_matrix_uniform(size):
    return torch.rand(size, size) - 0.5

def random_matrix_normal(size):
    return torch.randn(size, size)

def random_matrix_1(size):
    m = torch.randn(size, size)
    m = m.abs().exp() * m.sign() / 25
    return m

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