
import torch
import operators

device = torch.device("cuda")


def validate():
    M, N, S, P = 96, 80, 112, 64

    ref_A = torch.rand(M, N, device=device)
    ref_B = torch.rand(N, P, device=device)
    ref_C = torch.rand(M, S, device=device)
    ref_X = torch.zeros(M, P, device=device)
    ref_Y = torch.zeros(N, S, device=device)

    operators.versions["reference"](ref_A, ref_B, ref_C, ref_X, ref_Y)

    for ver in operators.versions.keys():

        if ver != "reference":

            A = ref_A.clone()
            B = ref_B.clone()
            C = ref_C.clone()
            X = torch.zeros(M, P, device=device)
            Y = torch.zeros(N, S, device=device)

            operators.versions[ver](A, B, C, X, Y)

            try:
                torch.testing.assert_close(A, ref_A)
                torch.testing.assert_close(B, ref_B)
                torch.testing.assert_close(C, ref_C)
                torch.testing.assert_close(X, ref_X)
                torch.testing.assert_close(Y, ref_Y)
            except:
                print("validation FAILED for ", ver)
                print("A - ref_A = ", torch.norm(A - ref_A).item())
                print("B - ref_B = ", torch.norm(B - ref_B).item())
                print("C - ref_C = ", torch.norm(C - ref_C).item())
                print("X - ref_X = ", torch.norm(X - ref_X).item())
                print("Y - ref_Y = ", torch.norm(Y - ref_Y).item())
            else: 
                print("validation SUCCESS for ", ver)
