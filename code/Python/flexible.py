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

torch.manual_seed(18)

l = nn.Linear(1024, 1024)
with torch.no_grad():
    l.bias *= 0
r = LowRankLight(1024, 1024, 256)
# l.weight = nn.Parameter(1e-3 * l.weight)


r.project_regularized(l, 1e-0)
print((l.weight.T - r(torch.eye(1024, 1024))).norm())

r.project_regularized(l, 1e-1)
print((l.weight.T - r(torch.eye(1024, 1024))).norm())

r.project_regularized(l, 1e-2)
print((l.weight.T - r(torch.eye(1024, 1024))).norm())

r.project_regularized(l, 1e-3)
print((l.weight.T - r(torch.eye(1024, 1024))).norm())

r.project_regularized(l, 1e-4)
print((l.weight.T - r(torch.eye(1024, 1024))).norm())

r.project_regularized(l, 1e-5)
print((l.weight.T - r(torch.eye(1024, 1024))).norm())

r.project_regularized(l, 1e-6)
print((l.weight.T - r(torch.eye(1024, 1024))).norm())

r.project_regularized(l, 1e-7)
print((l.weight.T - r(torch.eye(1024, 1024))).norm())

r.project_precise(l)
print((l.weight.T - r(torch.eye(1024, 1024))).norm())

# from layers import *
# l = Blast(200, 250, 50, 50)
# m = nn.Linear(200, 250)
# # l.project(m)
# l = Blast(200, 250, 50, 50)
# l.precGD(m.weight)

