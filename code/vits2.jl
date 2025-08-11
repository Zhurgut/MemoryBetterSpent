include("run.jl")


base_id = 40


nr_runs=2


# collect_measurements(
#     layer=dense,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=192,
#     depth=9,
#     lr=1e-3,
#     bs=256,
#     max_epochs=1600,
#     weight_decay=0.05,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     nr_runs=nr_runs,
#     id=base_id,
#     patch_size=8,
#     nr_heads=12
# )







# collect_measurements(
#     layer=lowrank,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=192,
#     depth=9,
#     lr=1e-3,
#     bs=256,
#     max_epochs=1600,
#     weight_decay=0.03,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     nr_runs=nr_runs,
#     id=base_id + 1,
#     patch_size=8,
#     nr_heads=12,
#     rank=128
# )


# collect_measurements(
#     layer=lowrank,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=192,
#     depth=9,
#     lr=1e-3,
#     bs=256,
#     max_epochs=1600,
#     weight_decay=0.02,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     nr_runs=nr_runs,
#     id=base_id + 1,
#     patch_size=8,
#     nr_heads=12,
#     rank=96
# )


# collect_measurements(
#     layer=lowrank,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=192,
#     depth=9,
#     lr=1e-3,
#     bs=256,
#     max_epochs=1600,
#     weight_decay=0.03,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     nr_runs=nr_runs,
#     id=base_id + 1,
#     patch_size=8,
#     nr_heads=12,
#     rank=48
# )


# collect_measurements(
#     layer=lowrank,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=192,
#     depth=9,
#     lr=1e-3,
#     bs=256,
#     max_epochs=1600,
#     weight_decay=0.03,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     nr_runs=nr_runs,
#     id=base_id + 3,
#     patch_size=8,
#     nr_heads=12,
#     rank=24
# )






# collect_measurements(
#     layer=lowranklight,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=192,
#     depth=9,
#     lr=1e-3,
#     bs=256,
#     max_epochs=1600,
#     weight_decay=0.02,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     nr_runs=nr_runs,
#     id=base_id + 2,
#     patch_size=8,
#     nr_heads=12,
#     rank=96
# )

# collect_measurements(
#     layer=lowranklight,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=192,
#     depth=9,
#     lr=1e-3,
#     bs=256,
#     max_epochs=1600,
#     weight_decay=0.02,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     nr_runs=nr_runs,
#     id=base_id + 2,
#     patch_size=8,
#     nr_heads=12,
#     rank=48
# )

collect_measurements(
    layer=lowranklight,
    model=vit2,
    dataset=tiny_imagenet,
    width=192,
    depth=9,
    lr=1e-3,
    bs=256,
    max_epochs=1600,
    weight_decay=0.02,
    lr_decay=true,
    early_stopping=false,
    dropout=0.1,
    max_bs=1000,
    nr_runs=nr_runs,
    id=base_id + 3,
    patch_size=8,
    nr_heads=12,
    rank=24
)










# collect_measurements(
#     layer=blast,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=192,
#     depth=9,
#     lr=3e-3,
#     bs=256,
#     max_epochs=1600,
#     weight_decay=0.02,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     nr_runs=nr_runs,
#     id=base_id + 7,
#     patch_size=8,
#     nr_heads=12,
#     block_size=48,
#     rank=80,
# )

# collect_measurements(
#     layer=blast,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=192,
#     depth=9,
#     lr=3e-3,
#     bs=256,
#     max_epochs=1600,
#     weight_decay=0.02,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     nr_runs=nr_runs,
#     id=base_id + 7,
#     patch_size=8,
#     nr_heads=12,
#     block_size=48,
#     rank=56,
# )

# collect_measurements(
#     layer=blast,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=192,
#     depth=9,
#     lr=3e-3,
#     bs=256,
#     max_epochs=1600,
#     weight_decay=0.02,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     nr_runs=nr_runs,
#     id=base_id + 7,
#     patch_size=8,
#     nr_heads=12,
#     block_size=48,
#     rank=32,
# )



