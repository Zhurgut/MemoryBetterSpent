include("run.jl")


collect_measurements(
    layer=dense,
    model=vit2,
    dataset=tiny_imagenet,
    width=192,
    depth=9,
    lr=5e-4,
    bs=256,
    max_epochs=1200,
    weight_decay=0.02,
    lr_decay=false,
    early_stopping=false,
    dropout=0.1,
    max_bs=1000,
    id=30,
    patch_size=8,
    nr_heads=12
)


collect_measurements(
    layer=lowrank,
    model=vit2,
    dataset=tiny_imagenet,
    width=192,
    depth=9,
    lr=1e-3,
    bs=256,
    max_epochs=900,
    weight_decay=0.02,
    lr_decay=true,
    early_stopping=false,
    dropout=0.1,
    max_bs=1000,
    id=31,
    patch_size=8,
    nr_heads=12,
    rank=96
)

begin
    id=32

    collect_measurements(
        layer=lowranklight,
        model=vit2,
        dataset=tiny_imagenet,
        width=192,
        depth=9,
        lr=1e-3,
        bs=256,
        max_epochs=900,
        weight_decay=0.02,
        lr_decay=true,
        early_stopping=false,
        dropout=0.1,
        max_bs=1000,
        id=32,
        patch_size=8,
        nr_heads=12,
        rank=96
    )
    
    collect_measurements(
        layer=lowranklight,
        model=vit2,
        dataset=tiny_imagenet,
        width=192,
        depth=9,
        lr=1e-3,
        bs=256,
        max_epochs=900,
        weight_decay=0.02,
        lr_decay=true,
        early_stopping=false,
        dropout=0.1,
        max_bs=1000,
        id=32,
        patch_size=8,
        nr_heads=12,
        rank=48
    )

    # collect_measurements(
    #     layer=lowranklight,
    #     model=vit2,
    #     dataset=tiny_imagenet,
    #     width=192,
    #     depth=9,
    #     lr=5e-4,
    #     bs=256,
    #     max_epochs=1100,
    #     weight_decay=0.02,
    #     lr_decay=false,
    #     early_stopping=false,
    #     dropout=0.1,
    #     max_bs=1000,
    #     id=32,
    #     patch_size=8,
    #     nr_heads=12,
    #     rank=24
    # )



end