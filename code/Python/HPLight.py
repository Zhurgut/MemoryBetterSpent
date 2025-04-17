

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HP_Light_MMM(torch.autograd.Function):
    """Manifest the matrix first (store in preallocated 'weight' tensor), then do matrix matrix multiplication"""
    
    @staticmethod
    def forward(ctx, input, weightA, weightB, bias, weight, out):

        N, M = weight.shape # preallocated tensor for weight
        
        BS, m = input.shape
        # assert m == M
        
        # bs, n = out.shape
        # assert bs == BS
        # assert n == N
        
        n, rank = weightA.shape
        # assert n == N
        
        # r, mr = weightB.shape
        # assert r == rank
        # assert mr == M - rank
        
        weight[:, :rank] = weightA
        torch.matmul(weightA, weightB, out=weight[:, rank:])

        ctx.save_for_backward(input, weightA, weightB, weight, bias)
        
        torch.addmm(bias, input, weight.T, out=out)
        
        return out



    @staticmethod
    def backward(ctx, grad_output):
        
        input, weightA, weightB, weight, bias = ctx.saved_tensors
        # N, M = weight.shape
        rank, mr = weightB.shape
        
        grad_input = grad_weightA = grad_weightB = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            mid = torch.addmm(input[:, :rank], input[:, rank:], weightB.T)
            grad_weightA = grad_output.T @ mid
        
        if ctx.needs_input_grad[2]:
            if ctx.needs_input_grad[0]:
                grad_weightB = grad_input[:, :rank].T @ input[:, rank:]
            else:
                grad_weightB = torch.linalg.multi_dot([weightA.T, grad_output.T, input[:, rank:]])
        
        if ctx.needs_input_grad[3]:
            # Gradient with respect to bias is the sum along the batch (first dimension).
            grad_bias = grad_output.sum(dim=0)
        
        return grad_input, grad_weightA, grad_weightB, grad_bias, None, None



class HP_Light_MVM(torch.autograd.Function):
    """Efficient MVM for vector, dont manifest weight matrix"""
    
    @staticmethod
    def forward(ctx, input, weightA, weightB, bias, out):
        
        BS, M = input.shape
        
        bs, N = out.shape
        # assert bs == BS
        
        n, rank = weightA.shape
        # assert n == N
        
        # r, mr = weightB.shape
        # assert r == rank
        # assert mr == M - rank
        
        # mid = input[:, rank:] @ weightB.T

        # torch.addmm(bias, mid, weightA.T, out=out)
        # torch.addmm(out, input[:, :rank], weightA.T, out=out)


        mid = torch.addmm(input[:, :rank], input[:, rank:], weightB.T)
        out2 = torch.addmm(bias, mid, weightA.T)

        ctx.save_for_backward(input, weightA, weightB, mid)
        
        return out2



    @staticmethod
    def backward(ctx, grad_output):
        
        input, weightA, weightB, mid = ctx.saved_tensors
        BS, M = input.shape
        rank, mr = weightB.shape
        
        grad_input = grad_weightA = grad_weightB = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.empty(BS, M, device=grad_output.device)
            torch.matmul(grad_output, weightA, out=grad_input[:, :rank])
            torch.matmul(grad_input[:, :rank], weightB, out=grad_input[:, rank:])

        if ctx.needs_input_grad[1]:
            grad_weightA = grad_output.T @ mid
        
        if ctx.needs_input_grad[2]:
            if ctx.needs_input_grad[0]:
                grad_weightB = grad_input[:, :rank].T @ input[:, rank:]
            else:
                grad_weightB = torch.linalg.multi_dot([weightA.T, grad_output.T, input[:, rank:]])
            
        
        if ctx.needs_input_grad[3]:
            # Gradient with respect to bias is the sum along the batch (first dimension).
            grad_bias = grad_output.sum(dim=0)
        
        return grad_input, grad_weightA, grad_weightB, grad_bias, None, None





# class HP_BTTLight_MMM(torch.autograd.Function):
#     """Manifest the matrix first (store in preallocated 'weight' tensor), then do matrix matrix multiplication"""
    
#     @staticmethod
#     def forward(ctx, input, weightA, weightB, bias, weight, out):

#         N, M = weight.shape # preallocated tensor for weight
        
#         B, rank, dr = weightB.shape
        
#         b, D, r = weightA.shape
#         assert r == rank
#         assert dr == D - rank
#         assert b == B
        
#         BS, m = input.shape
#         assert m == M
        
#         bs, n = out.shape
#         assert bs == BS
#         assert n == N
        
#         weight.reshape((B, D, D))[:, :, :rank] = weightA
#         torch.matmul(weightA, weightB, out=weight.reshape((B, D, D))[:, :, rank:]) # should do bmm
        
#         ctx.save_for_backward(input, weightA, weightB, weight, bias)
        
#         torch.addmm(bias, input, weight.t(), out=out)
        
#         return out



    # @staticmethod
    # def backward(ctx, grad_output):
        
    #     input, weightA, weightB, weight, bias = ctx.saved_tensors
    #     N, M = weight.shape
    #     rank, mr = weightB.shape
        
    #     grad_input = grad_weightA = grad_weightB = grad_bias = None

    #     if ctx.needs_input_grad[0]:
    #         grad_input = grad_output.matmul(weight)
         
    #     if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
    #         grad_weight = grad_output.t().matmul(input)
    #         G2 = grad_weight[:, rank:]
            
    #         if ctx.needs_input_grad[1]:
    #             G1 = grad_weight[:, :rank]
    #             grad_weightA = torch.addmm(G1, G2, weightB.t())
            
    #         if ctx.needs_input_grad[2]:
    #             grad_weightB = weightA.t().matmul(G2)
        
    #     if ctx.needs_input_grad[3]:
    #         # Gradient with respect to bias is the sum along the batch (first dimension).
    #         grad_bias = grad_output.sum(dim=0)
        
    #     return grad_input, grad_weightA, grad_weightB, grad_bias, None, None





# def HP_BTTLight_MVM(input, weightA, weightB, bias):

        
#         BS, M = input.shape
        
#         B, rank, dr = weightB.shape
        
#         b, D, r = weightA.shape
#         assert r == rank
#         assert dr == D - rank
#         assert b == B
        
#         N = B * D * D // M
        
#         mid = torch.bmm(input.reshape(BS, M // D, D)[:, :, rank:].transpose(0, 1), weightB.reshape(M // D, D - rank, rank * N // D)).transpose(0, 1).reshape(-1, rank) + input.reshape(BS, M//D, D)[:, :, :rank].reshape(-1, rank)
#         # BS * M/D * N/D, rank
#         out = torch.bmm(mid.reshape(BS, N//D, -1).transpose(0, 1), weightA.reshape(N // D, -1, D)).transpose(0, 1) # BS, N // D, D
 
#         return out.reshape(BS, N) + bias





# class BTT_init(nn.Module):
     
#     def __init__(self, out_dim, in_dim, block_dim, rank):
#         super().__init__()
        
#         assert out_dim % block_dim == 0
#         assert in_dim  % block_dim == 0
#         assert rank <= block_dim
        
#         self.out_dim = out_dim
#         self.in_dim = in_dim
#         self.rank = rank
        
#         nr_blocks = (out_dim * in_dim) // (block_dim * block_dim)
        
#         self.A = nn.Parameter(nn.init.kaiming_normal_(torch.empty(nr_blocks, block_dim, rank)))
#         self.B = nn.Parameter(nn.init.kaiming_normal_(torch.empty(nr_blocks, rank, block_dim - rank))) 
        
#         self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())
        
#         print("bias: ", self.bias.min(), " < -> < ", self.bias.max())




# class BTTLight1(BTT_init):    

#     def __init__(self, out_dim, in_dim, block_dim, rank):
#         super().__init__(out_dim, in_dim, block_dim, rank)
        
#     def forward(self, x):
#         BS, _ = x.shape
#         return HP_BTTLight_MMM.apply(x, self.A, self.B, self.bias, torch.empty(self.out_dim, self.in_dim), torch.empty(BS, self.out_dim))
    

# class BTTLight2(BTT_init):
    
#     def __init__(self, out_dim, in_dim, block_dim, rank):
#         super().__init__(out_dim, in_dim, block_dim, rank)
        
#     def forward(self, x):
#         HP_BTTLight_MVM(x, self.A, self.B, self.bias)


class LRL_init(nn.Module):
     
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        
        assert rank <= min(out_dim, in_dim)
        
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.rank = rank
        
        self.A = nn.Parameter(nn.init.kaiming_normal_(torch.empty(out_dim, rank)))
        self.B = nn.Parameter(nn.init.kaiming_normal_(torch.empty(rank, in_dim - rank))) 
        
        self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, out_dim)).squeeze())

    
    
class LRLight0(LRL_init):    

    def __init__(self, in_dim, out_dim, rank):
        super().__init__(in_dim, out_dim, rank)

    def forward(self, x):

        W = torch.cat((self.A, self.A @ self.B), dim=1)
        return torch.addmm(self.bias, x, W.T)


class LRLight1(LRL_init):    

    def __init__(self, in_dim, out_dim, rank):
        super().__init__(in_dim, out_dim, rank)
        self.register_buffer("out", torch.empty(1, self.out_dim))
        self.register_buffer("weight", torch.empty(self.out_dim, self.in_dim))
        self.bs = 1

    def forward(self, x):
        BS, _ = x.shape
        if BS != self.bs:
            self.out.resize_(BS, self.out_dim)
            self.bs = BS

        return HP_Light_MMM.apply(x, self.A, self.B, self.bias, self.weight, self.out)
    

class LRLight2(LRL_init):

    def __init__(self, in_dim, out_dim, rank):
        super().__init__(in_dim, out_dim, rank)
        self.register_buffer("out", torch.empty(1, self.out_dim))
        self.bs = 1

    def forward(self, x):
        BS, _ = x.shape
        if BS != self.bs:
            self.out.resize_(BS, self.out_dim)
            self.bs = BS
        return HP_Light_MVM.apply(x, self.A, self.B, self.bias, self.out)


class LRLight3(LRL_init):

    def __init__(self, in_dim, out_dim, rank):
        super().__init__(in_dim, out_dim, rank)

    def forward(self, x):
        
        # mid = x[:, self.rank:] @ self.B.T

        # preout = torch.addmm(self.bias, mid, self.A.T)
        # out = torch.addmm(preout, x[:, :self.rank], self.A.T)
        
        r1 = torch.addmm(x[:, :self.rank], x[:, self.rank:], self.B.T)
        out = torch.addmm(self.bias, r1, self.A.T)

        return out


from layers import LowRankLight


BS = 53
in_dim = 64
out_dim = 48
rank=13
x = torch.randn(BS, in_dim)

l1 = LRLight1(in_dim, out_dim, rank)
l2 = LRLight2(in_dim, out_dim, rank)

l1.A = l2.A
l1.B = l2.B
l1.bias = l2.bias

with torch.no_grad():
    print((l1(x) - l2(x)).norm().item(), " < small?")

l1 = LRLight1(in_dim, out_dim, rank)
l2 = LRLight2(in_dim, out_dim, rank)

print(l1(x).shape, " =?= ", l2(x).shape, " =?= ", (BS, out_dim))

y = torch.rand(BS, 1)


model = nn.Sequential(l1, nn.ReLU(), nn.Linear(out_dim, 1))

loss = nn.MSELoss()
opt = opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(50):
    L = loss(y, model(x))
    if i%5 == 0:
        print(i, ": ", L.item())
    opt.zero_grad()
    L.backward()
    opt.step()

model = nn.Sequential(l2, nn.ReLU(), nn.Linear(out_dim, 1))

loss = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(50):
    L = loss(y, model(x))
    if i%5 == 0:
        print(i, ": ", L.item())
    opt.zero_grad()
    L.backward()
    opt.step()


def train(model, opt, x, y):
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA
    #     ],
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:

    for i in range(2001):
        
        L = loss(y, model(x))
        
        # if i%100 == 0:
        #     print(i, ": ", L.item())
        opt.zero_grad()
        L.backward()
        opt.step()
            
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

import timeit

def dense(in_dim, out_dim, rank):
    # return nn.Sequential(nn.Linear(in_dim, rank, bias=False), nn.Linear(rank, out_dim))
    return nn.Linear(in_dim, out_dim)

def time(fn):
    in_dim = 1024
    out_dim = 960
    BSs = [1, 32, 512, 4096, 16384]
    rank = 768
    for BS in BSs:
        model = nn.Sequential(fn(in_dim, in_dim, rank), nn.ReLU(), fn(in_dim, out_dim, rank), nn.ReLU(), fn(out_dim, out_dim, rank), nn.ReLU(), nn.Linear(out_dim, 1))
        x = torch.randn(BS, in_dim).to(device)
        y = torch.rand(BS, 1).to(device)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        time_taken = timeit.timeit(lambda: train(model, opt, x, y), number=1)
        print(f"Time taken: {time_taken:.6f} seconds")

# print("MM_df")
# time(LRLight0)
# print("MM")
# time(LRLight1)
print("MVM")
time(LRLight2)
# print("MVM_df")
# time(LRLight3)
# print("dense")
# time(dense)