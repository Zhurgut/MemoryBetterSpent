include("run.jl")


collect_measurements(
    layer=dense,
    model=vit2,
    dataset=tiny_imagenet,
    width=192,
    depth=9, # huge
    lr=[2e-3, 1e-3],
    bs=256,
    max_epochs=200,
    weight_decay=[0.01, 0.02, 0.04],
    lr_decay=true,
    early_stopping=false,
    dropout=[0.1, 0.2],
    max_bs=1000,
    id=20,
    patch_size=8,
    nr_heads=12
)

# collect_measurements(
#     layer=lowrank,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=768,
#     depth=9, # huge
#     lr=5e-4,
#     bs=256,
#     max_epochs=50,
#     weight_decay=0.01,
#     lr_decay=false,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     id=15,
#     patch_size=8,
#     nr_heads=12,
#     rank=384
# )

# collect_measurements(
#     layer=lowranklight,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=768,
#     depth=9, # huge
#     lr=5e-4,
#     bs=256,
#     max_epochs=50,
#     weight_decay=0.01,
#     lr_decay=false,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     id=16,
#     patch_size=8,
#     nr_heads=12,
#     rank=384
# )
