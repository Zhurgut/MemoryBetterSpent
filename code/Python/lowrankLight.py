

import torch
from torch import nn
from math import prod
import torch.nn.functional as F
import math

class HP_LowRankLight_MVM(torch.autograd.Function):
    """Efficient MVM for vector, dont manifest weight matrix"""
    
    @staticmethod
    def forward(ctx, x, A, B, bias):
        
        M = x.shape[-1]
        BS = prod(x.shape[:-1])
        shape = x.shape
        X = x.reshape(BS, M)

        N, rank = A.shape
        
        r, mr = B.shape
        assert r == rank
        assert mr == M - rank
        
        X_a = X[:, :rank] # BS * k
        X_ab = X[:, rank:] # BS * (N - k)

        # compute:
        # BS * (N - k) * k + BS * k + BS * k * N + BS * N
        # = BS * k * (2N - k + 1) + BS * N
        # memory:
        # BS * N + k * (N - k) + BS * k + N * k + N + BS * N
        # k * (2N - k + BS) + (2BS + 1) * N
        mid = torch.addmm(X_a, X_ab, B.T) # BS * k
        out = torch.addmm(bias, mid, A.T)

        ctx.save_for_backward(x, A, B, mid)
        
        return out.reshape(*(shape[:-1]), N)



    @staticmethod
    def backward(ctx, grad_output):
        # memory:
        # 2*BS*N + 2N*k + 2(N-k)*k + BS*k + BS * (N - k) + N
        # 3*BS*N + 4nk - 2k^2 + N
        # compute: (common case, all require grads)
        # BS*N*k + BS*(N-k)*k + BS*N*k + BS*k*(N-k) + N*BS
        # = 2*BS*(-k^2 + 2Nk) + N*BS -> compute peaks for rank=N/2
        # dense: 2BS*N*N + N*BS < 1.5*BS*N*N tho, so should still be better than dense
        
        x, A, B, mid = ctx.saved_tensors
        M = x.shape[-1]
        BS = prod(x.shape[:-1])
        X = x.reshape(BS, M)
        X_ab = X[:, rank:] # BS * (N - k)
        rank, mr = B.shape
        
        grad_output = grad_output.reshape(BS, -1) # BS * N
        
        grad_input = grad_weightA = grad_weightB = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.empty(BS, M, device=grad_output.device) # BS * N
            torch.matmul(grad_output, A, out=grad_input[:, :rank])
            torch.matmul(grad_input[:, :rank], B, out=grad_input[:, rank:])
            

        if ctx.needs_input_grad[1]:
            grad_weightA = grad_output.T @ mid
        
        if ctx.needs_input_grad[2]:
            if ctx.needs_input_grad[0]:
                grad_weightB = grad_input[:, :rank].T @ X_ab
            else:
                grad_weightB = torch.linalg.multi_dot([A.T, grad_output.T, X_ab])
            
        
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output.reshape(BS, -1).sum(dim=0)
        
        if grad_input is not None:
            grad_input = grad_input.reshape(x.shape)
        
        return grad_input, grad_weightA, grad_weightB, grad_bias




class LowRankLight(nn.Module):    

    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        
        assert rank <= min(out_dim, in_dim)
        
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.rank = rank
        
        bound = 1 / math.sqrt(in_dim)
        self.A = nn.Parameter(nn.init.uniform_(torch.empty(out_dim, rank), a=-bound, b=bound))
        # self.A = nn.Parameter(nn.init.kaiming_normal_(torch.empty(out_dim, rank)))
        
        b = F.normalize(nn.init.normal_(torch.empty(rank, in_dim - rank)), p=2, dim=0)

        self.B = nn.Parameter(b * 0.5*bound / (bound / math.sqrt(3))) # now all values in [A; A*B] roughly from the same distribution as A, which is the same distribution as nn.Linear
        
        self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())

        # l = nn.Linear(in_dim, out_dim)

        # U, S, Vt = torch.linalg.svd(l.weight)
        # s = torch.diag(S)[:rank, :rank]

        # X = U[:, :rank] @ s
        # Y = Vt[:rank, :]
        # Y1 = Y[:, :rank]
        # Y2 = Y[:, rank:]

        # self.A = nn.Parameter(X @ Y1)
        # self.B = nn.Parameter(torch.linalg.pinv(Y1) @ Y2)
        
        # self.bias = nn.Parameter(l.bias)

    
    @staticmethod
    def from_matrix(M, rank):
        out_dim, in_dim = M.shape
  
        U, S, Vt = torch.linalg.svd(M)
        S = torch.diag(S)
        
        A = U[:, :rank] @ S[:rank, :rank]
        B = Vt[:rank, :]
        B1 = B[:, :rank]
        B2 = B[:, rank:]
        
        X = A @ B1
        Y = torch.linalg.pinv(B1) @ B2
        
        out = LowRankLight(in_dim, out_dim, rank)
        out.A = nn.Parameter(X)
        out.B = nn.Parameter(Y)
        
        return out
        

    def forward(self, x):
        # return HP_LowRankLight_MVM.apply(x, self.A, self.B, self.bias)
    
        M = x.shape[-1]
        BS = prod(x.shape[:-1])
        shape = x.shape
        X = x.reshape(BS, M)

        X_a = X[:, :self.rank]
        X_ab = X[:, self.rank:]

        mid = torch.addmm(X_a, X_ab, self.B.T)
        out = torch.addmm(self.bias, mid, self.A.T)
        
        return out.reshape(*(shape[:-1]), -1)
    


class LowRankLight_validation(nn.Module):
    

    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        
        assert rank <= min(out_dim, in_dim)
        
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.rank = rank
        
        k = (1/in_dim) ** 0.5 # just like nn.Linear
        self.A = nn.Parameter(nn.init.uniform_(torch.empty(out_dim, rank), -k, k))
        # self.A = nn.Parameter(nn.init.kaiming_normal_(torch.empty(out_dim, rank)))
        self.B = nn.Parameter(nn.init.kaiming_normal_(torch.empty(rank, in_dim - rank))) 
        
        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), -k, k))



    def forward(self, x):
    
        M = x.shape[-1]
        BS = prod(x.shape[:-1])
        shape = x.shape
        X = x.reshape(BS, M)

        X_a = X[:, :self.rank]
        X_ab = X[:, self.rank:]

        mid = torch.addmm(X_a, X_ab, self.B.T)
        out = torch.addmm(self.bias, mid, self.A.T)
        
        return out.reshape(*(shape[:-1]), -1)


def test(input_requires_grad = True):
    
    x = torch.rand(23, 15, 16)
    x.requires_grad = input_requires_grad
    L = LowRankLight_validation(16, 24, 7)
    
    L2 = LowRankLight(16, 24, 7)
    L2.A = nn.Parameter(L.A.clone())
    L2.B = nn.Parameter(L.B.clone())
    L2.bias = nn.Parameter(L.bias.clone())
    
    loss = L(x).sum()
    loss.backward()
    
    a_grad = L.A.grad.clone()
    b_grad = L.B.grad.clone()
    if x.requires_grad:
        x_grad = x.grad.clone()
        x.grad = None
    
    loss2 = L2(x).sum()
    loss2.backward()
    
    a_grad2 = L2.A.grad.clone()
    b_grad2 = L2.B.grad.clone()
    if x.requires_grad:
        x_grad2 = x.grad.clone()

    assert (a_grad == a_grad2).all()
    assert (b_grad == b_grad2).all()
    if x.requires_grad:
        assert (x_grad == x_grad2).all()


# import time



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# rank = 576
# model = nn.Sequential(LowRankLight(1024, 1024, rank), LowRankLight(1024, 1024, rank), LowRankLight(1024, 1024, rank), LowRankLight(1024, 1024, rank), LowRankLight(1024, 1024, rank), nn.Linear(1024, 1)).to(device)
# # model = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1024), nn.Linear(1024, 1024), nn.Linear(1024, 1024), nn.Linear(1024, 1024), nn.Linear(1024, 1)).to(device)
# x = torch.randn(4096, 1024).to(device)

# timings = []
# for i in range(100):
#     y = model(x)
#     torch.cuda.synchronize()
#     start = time.perf_counter()
#     y.sum().backward()
#     torch.cuda.synchronize()
#     end   = time.perf_counter()
    
#     timings.append(end-start)

# print(f"Elapsed: {sorted(timings)[len(timings)//2]:.9f} seconds")