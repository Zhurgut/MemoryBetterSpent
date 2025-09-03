import torch
from torchvision.io import read_image, write_png
from layers import *
from main import nr_parameters


file = open("out/info.txt", "w")


centered = False



def load_image_split_channels(img_path):
    """
    Loads an RGB image, normalizes to [0,1], and returns 3 separate channel tensors.
    
    Returns:
        r, g, b: tensors of shape (H, W), dtype=float32, range [0,1]
    """
    img = read_image(img_path)  # Shape: (3, H, W), dtype=uint8
    img = img.float() / 255.0   # Normalize to [0,1]

    # c, h, w = img.shape
    # print(c, ", ", h, ", ", w)

    r,g,b = None, None, None
    

    if centered:
    
        r = img[0] * 2 - 1
        g = img[1] * 2 - 1
        b = img[2] * 2 - 1
    
    else:

        r = img[0]
        g = img[1]
        b = img[2]

    # gray = (r + g + b) / 3
    # return gray, gray, gray 

    return r, g, b


def save_image_from_channels(r, g, b, out_path):
    """
    Takes 3 channel tensors in [0,1], assembles into RGB image, saves as PNG.
    """
    img = torch.stack([r, g, b], dim=0)  # (3, H, W)

    if centered:
        img = torch.clamp(img, -1.0, 1.0)     # Safety clamp
        img = (img + 1) * 0.5
    else:
        img = torch.clamp(img, 0.0, 1.0)

    img = (img * 255.0).byte()
    write_png(img, out_path)

def compress_channel(img, layer, print_info=False):

    h,w = layer.out_dim, layer.in_dim

    img = img[:h, :w]

    l = nn.Linear(w, h)

    if print_info:
        dense_pms = nr_parameters(l) - h
        layer_pms = nr_parameters(layer) - h
        pc = layer_pms / dense_pms * 100
        pc = round(pc, ndigits=2)
        file.write(f"{pc}%\n")

    l.bias = nn.Parameter(torch.zeros(h))
    l.weight = nn.Parameter(img)

    layer.project(l)

    return layer(torch.eye(w, w)).T


def compress(layer, img_path, out_img_name):

    file.write(f"{out_img_name}, ")
    
    r,g,b = load_image_split_channels(img_path) 

    rc = compress_channel(r, layer, True)
    gc = compress_channel(g, layer)
    bc = compress_channel(b, layer)

    save_image_from_channels(rc, gc, bc, f"./out/{out_img_name}.png")

    

# load_image_split_channels("chevy.jpg") # size 776 - 1156

# compress(Unstructured(1156, 776, 7), "chevy.jpg", "Unstructured 7%")
# compress(Unstructured(1156, 776, 20), "chevy.jpg", "Unstructured 20%")


# for i in [2, 15]:
#     compress(LowRank(1156, 776, i), "chevy.jpg", f"LowRank k={i}")

#     lrl = LowRankLight(1156, 776, i)

#     lrl.regularization = 1e-7
#     compress(lrl, "chevy.jpg", f"LowRank-Light k={i}")

# for i in [40]:
#     compress(LowRank(1156, 776, i), "chevy.jpg", f"LowRank k={i}")

#     lrl = LowRankLight(1156, 776, i)

#     lrl.regularization = 5e-7
#     compress(lrl, "chevy.jpg", f"LowRank-Light k={i}")


# lrl = LowRankLight(1156, 776, 12)
# lrl.regularization = 0.0
# compress(lrl, "chevy.jpg", f"LowRank-Light k=12 no regularization")

# lrl = LowRankLight(1156, 776, 12)
# lrl.regularization = 1e-4
# compress(lrl, "chevy.jpg", f"LowRank-Light k=12 too much regularization")


# compress(Monarch(1155, 774, 3), "chevy.jpg", "Monarch b=3")
# compress(Monarch(1152, 768, 32), "chevy.jpg", "Monarch b=32")


# compress(BTT2(1156, 776, 1), "chevy.jpg", "BTT'' k=1")
# compress(BTT2(1156, 776, 3), "chevy.jpg", "BTT'' k=3")



# compress(BlastPaper(1152, 768, 8, 4), "chevy.jpg", "BLAST-8x8 k=4")
# compress(BlastPaper(1152, 768, 8, 12), "chevy.jpg", "BLAST-8x8 k=12")
# compress(BlastPaper(1152, 768, 8, 32), "chevy.jpg", "BLAST-8x8 k=32")


# compress(Kronecker(1156, 776, 4), "chevy.jpg", "Kronecker 4")
# compress(Kronecker(1152, 768, 32), "chevy.jpg", "Kronecker 32")

# compress(TT(1156, 729, 2, 1), "chevy.jpg", "TT d=2 k=1")
compress(TT(1000, 729, 3, 1), "chevy.jpg", "TT d=3 k=1")
compress(TT(1000, 729, 3, 4), "chevy.jpg", "TT d=3 k=4")
# compress(TT(625, 625, 2, 1), "chevy.jpg", "TT d=4 k=1")

# compress(TT(1156, 729, 2, 20), "chevy.jpg", "TT d=2 k=20")
compress(TT(1000, 729, 3, 20), "chevy.jpg", "TT d=3 k=20")


file.close()


# perm = torch.arange(0, 3072)
# for i in range(3072):
#     r = (torch.rand(1) * 3072).int()
#     perm[i], perm[r.item()] = perm[r.item()], perm[i]



# M = torch.load("cifar10weight.pt")
# torch.save(M[:, perm], "cifar10weightSHUFFLED.pt")

# m = M.to(torch.device("cpu"))
# m = (m - m.min()) / (m.max() - m.min())
# m = m * 255
# m = m.byte()
# m = m[:, perm]
# m = m.reshape(1, *m.shape)

# write_png(m, "cifar10weight.png")