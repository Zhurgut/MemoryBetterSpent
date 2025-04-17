from transformers import AutoModel
import torch


model_name = "bert-base-uncased"  # Change to your desired model
model = AutoModel.from_pretrained(model_name)

l1 = dict(model.named_modules())["encoder.layer.4.attention.self.query"].weight
l2 = dict(model.named_modules())["encoder.layer.5.attention.self.key"].weight
l3 = dict(model.named_modules())["encoder.layer.6.attention.self.value"].weight
l4 = dict(model.named_modules())["encoder.layer.7.attention.output.dense"].weight

# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         print(module.weight.shape, ", ", name)


def random_matrix_uniform(size):
    return torch.rand(size, size) - 0.5


def random_matrix_normal(size):
    return torch.randn(size, size)


def random_matrix_1(size):
    m = torch.randn(size, size)
    m = m.abs().exp() * m.sign() / 25
    return m

# from a real model
def model_matrix(size, index):
    return [l1, l2, l3, l4][index][:size, :size]