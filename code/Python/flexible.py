from layers import *


def flex():
    A = Dense(64, 64)
    B = Dense(64, 64)
    A.bias.requires_grad = False
    A.bias *= 0
    B.bias.requires_grad = False
    B.bias *= 0

    l = LowRankLight(64, 64, 64)
    # l = Blast(64, 64, 8, 20)
    l.project(A)
    l.bias.requires_grad = False

    I = torch.eye(64, 64)
    a = A(I)
    b = B(I)
    m = l(I)


    x = alpha(a, b, m)
    print(x)

    opt = torch.optim.Adam(l.parameters(), 1e-3)


    for i in range(1000):
        opt.zero_grad()

        loss = torch.linalg.norm(l(I) - B(I), ord="fro")
        loss.backward()

        opt.step()

        x = alpha(a, b, l(I))

        print(i, ": ", x)

        if x <= 0.01:
            print("nr_steps: ", i)
            break




def alpha(A, B, M):
    return -torch.trace((A-B).T @ (B-M)) / torch.trace((A-B).T @ (A-B))


# flex()


m = torch.Tensor([
    [2, 2, 4], 
    [1, 1, 2], 
    [2, 2.0, -1]])

l = nn.Linear(3, 3)
with torch.no_grad():
    l.bias *= 0

f = LowRankLight(3, 3, 2)
l.weight = nn.Parameter(m)
f.project(l)

print(m.T)
print(f(torch.eye(3, 3)))
print(f.A)
print(f.B)
