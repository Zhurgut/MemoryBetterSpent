include("run.jl")

begin
    collect_measurements(
        layer=dense,
        model=mlp,
        dataset=simple,
        width=13,
        depth=3,
        lr=8e-4,
        bs=10000,
        max_epochs=10000,
        weight_decay=0.0,
        lr_decay=true,
        early_stopping=false,
        dropout=0.0,
        id=60
    )

    collect_measurements(
        layer=lowrank,
        model=mlp,
        dataset=simple,
        width=20,
        depth=3,
        lr=8e-4,
        bs=10000,
        max_epochs=10001,
        weight_decay=0.0,
        lr_decay=true,
        early_stopping=false,
        dropout=0.0,
        id=61,
        rank=4
    )

    collect_measurements(
        layer=lowranklight,
        model=mlp,
        dataset=simple,
        width=14,
        depth=3,
        lr=6e-4,
        bs=10000,
        max_epochs=10002,
        weight_decay=0.0,
        lr_decay=true,
        early_stopping=false,
        dropout=0.0,
        id=62,
        rank=7
    )

end





collect_measurements(dense, mlp2, simple, collect(2:32:1024), 0, [1e-3, 3e-3, 1e-4], 100000, 100, 0.0, true, false, 1.0, 0.0, 1, 1000000, id=0)
collect_measurements(dense, mlp2, simple, collect(1024:128:4096), 0, [3e-3, 5e-3, 1e-4], 100000, 100, 0.0, true, false, 1.0, 0.0, 1, 1000000, id=0)

collect_measurements(dense, mlp2, simple, collect(2:32:1024), 0, [1e-3, 3e-3, 1e-4], 100000, 400, 0.0, true, false, 1.0, 0.0, 1, 1000000, id=1)
collect_measurements(dense, mlp2, simple, collect(1024:128:4096), 0, [3e-3, 5e-3, 1e-4], 100000, 400, 0.0, true, false, 1.0, 0.0, 1, 1000000, id=1)

collect_measurements(dense, mlp2, simple, collect(2:32:1024), 0, [1e-3, 3e-3, 1e-4], 100000, 100, 0.0, false, false, 1.0, 0.0, 1, 1000000, id=2)
collect_measurements(dense, mlp2, simple, collect(1024:128:4096), 0, [3e-3, 5e-3, 1e-4], 100000, 100, 0.0, false, false, 1.0, 0.0, 1, 1000000, id=2)

collect_measurements(
    layer=lowranklight,
    model=mlp,
    dataset=simple,
    width=1024,
    depth=4,
    lr=2e-4,
    bs=10000,
    max_epochs=200,
    weight_decay=0.05,
    lr_decay=false,
    dropout=0.0,
    id=3,
    rank=64
)


collect_measurements(
    layer=lowranklight,
    model=mlp,
    dataset=cifar10,
    width=1024,
    depth=4,
    lr=1e-3,
    bs=1000,
    max_epochs=700,
    weight_decay=0.02,
    lr_decay=true,
    early_stopping=false,
    dropout=0.2,
    rank=64
)



begin
    widths = collect(2:16:220)
    nr_runs = 3

    lrs = [3e-3, 1.2e-3, 1.6e-3, 8e-4, 5e-4, 3e-4]

    id = 11
    collect_measurements( layer=dense, model=mlp, dataset=simple, width=widths,
        depth=1,
        lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
    )

    id = 12
    collect_measurements( layer=dense, model=mlp, dataset=simple, width=widths,
        depth=2,
        lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
    )

    d = 1
    id += 1
    for w in widths
        collect_measurements( layer=lowrank, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
            rank=Int(round(0.5*w))
        )
    end

    id += 1
    for w in widths
        collect_measurements( layer=lowranklight, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
            rank=Int(round(0.5*w))
        )
    end

    
    id += 1
    for w in widths
        collect_measurements( layer=lowrank, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
            rank=Int(round(0.25*w))
        )
    end

    id += 1
    for w in widths
        collect_measurements( layer=lowranklight, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
            rank=Int(round(0.25*w))
        )
    end

    
    id += 1
    for w in widths
        collect_measurements( layer=lowrank, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
            rank=Int(round(0.125*w))
        )
    end

    id += 1
    for w in widths
        collect_measurements( layer=lowranklight, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
            rank=Int(round(0.125*w))
        )
    end

    # d += 1
    # id += 1
    # for w in widths
    #     collect_measurements( layer=lowrank, model=mlp, dataset=simple, 
    #         width=w,
    #         depth=d,
    #         lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
    #         rank=Int(round(0.5*w))
    #     )
    # end

    # id += 1
    # for w in widths
    #     collect_measurements( layer=lowranklight, model=mlp, dataset=simple, 
    #         width=w,
    #         depth=d,
    #         lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
    #         rank=Int(round(0.5*w))
    #     )
    # end

    
    # id += 1
    # for w in widths
    #     collect_measurements( layer=lowrank, model=mlp, dataset=simple, 
    #         width=w,
    #         depth=d,
    #         lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
    #         rank=Int(round(0.25*w))
    #     )
    # end

    # id += 1
    # for w in widths
    #     collect_measurements( layer=lowranklight, model=mlp, dataset=simple, 
    #         width=w,
    #         depth=d,
    #         lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
    #         rank=Int(round(0.25*w))
    #     )
    # end

    
    # id += 1
    # for w in widths
    #     collect_measurements( layer=lowrank, model=mlp, dataset=simple, 
    #         width=w,
    #         depth=d,
    #         lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
    #         rank=Int(round(0.125*w))
    #     )
    # end

    # id += 1
    # for w in widths
    #     collect_measurements( layer=lowranklight, model=mlp, dataset=simple, 
    #         width=w,
    #         depth=d,
    #         lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=500, weight_decay=0.0, lr_decay=true, early_stopping=false, dropout=0.0, id=id,
    #         rank=Int(round(0.125*w))
    #     )
    # end

end



begin
    widths = collect(2:2:18)
    nr_runs = 2

    lrs = [1.5e-2, 1e-2, 8e-3, 6e-3]
    epochs = 10000

    id = 40
    collect_measurements( layer=dense, model=mlp, dataset=simple, width=widths,
        depth=1,
        lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=epochs, weight_decay=0.0, lr_decay=true, early_stopping=true, dropout=0.0, id=id,
    )

    widths = collect(2:2:18)
    d = 1
    id += 1
    for w in widths
        collect_measurements( layer=lowrank, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=epochs, weight_decay=0.0, lr_decay=true, early_stopping=true, dropout=0.0, id=id,
            rank=Int(round(0.5*w))
        )
    end

    id += 1
    for w in widths
        collect_measurements( layer=lowranklight, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=epochs, weight_decay=0.0, lr_decay=true, early_stopping=true, dropout=0.0, id=id,
            rank=Int(round(0.5*w))
        )
    end

    widths = collect(3:3:24)
    id += 1
    for w in widths
        collect_measurements( layer=lowrank, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=epochs, weight_decay=0.0, lr_decay=true, early_stopping=true, dropout=0.0, id=id,
            rank=Int(round(0.25*w))
        )
    end

    id += 1
    for w in widths
        collect_measurements( layer=lowranklight, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=epochs, weight_decay=0.0, lr_decay=true, early_stopping=true, dropout=0.0, id=id,
            rank=Int(round(0.25*w))
        )
    end

    widths = collect(6:3:28)
    id += 1
    for w in widths
        collect_measurements( layer=lowrank, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=epochs, weight_decay=0.0, lr_decay=true, early_stopping=true, dropout=0.0, id=id,
            rank=Int(round(0.125*w))
        )
    end

    id += 1
    for w in widths
        collect_measurements( layer=lowranklight, model=mlp, dataset=simple, 
            width=w,
            depth=d,
            lr=lrs, nr_runs = nr_runs, bs=100000, max_epochs=epochs, weight_decay=0.0, lr_decay=true, early_stopping=true, dropout=0.0, id=id,
            rank=Int(round(0.125*w))
        )
    end

end



collect_measurements(
    layer=dense,
    model=vit2,
    dataset=tiny_imagenet,
    width=768,
    depth=4,
    lr=5e-4,
    bs=128,
    max_bs=1000,
    max_epochs=2,
    weight_decay=0.01,
    lr_decay=true,
    dropout=0.2,
    id=55,
    patch_size=8,
    nr_heads=12
)

collect_measurements(
    layer=dense,
    model=mlp,
    dataset=cifar10,
    width=1024,
    depth=[3, 4],
    lr=[1e-4, 5e-4, 1e-3],
    bs=1000,
    max_bs=5000,
    max_epochs=4,
    weight_decay=[0.01, 0.02, 0.05],
    lr_decay=true,
    dropout=[0.0, 0.1, 0.2, 0.3],
    id=100,
)

collect_measurements(
    layer=dense,
    model=mlp,
    dataset=cifar10,
    width=1024,
    depth=4,
    lr=1e-4,
    bs=1000,
    max_bs=5000,
    max_epochs=4,
    weight_decay=0.022,
    lr_decay=true,
    dropout=0.1,
    id=100,
)




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
#     id=base_id + 3,
#     patch_size=8,
#     nr_heads=12,
#     rank=24
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

collect_measurements(
    layer=blast,
    model=vit2,
    dataset=tiny_imagenet,
    width=192,
    depth=9,
    lr=3e-3,
    bs=256,
    max_epochs=1600,
    weight_decay=0.02,
    lr_decay=true,
    early_stopping=false,
    dropout=0.1,
    max_bs=1000,
    nr_runs=nr_runs,
    id=base_id + 7,
    patch_size=8,
    nr_heads=12,
    block_size=48,
    rank=32,
)



# collect_measurements(
#     layer=lowrank,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=384,
#     depth=10,
#     lr=[8e-4],
#     bs=256,
#     max_epochs=1600,
#     weight_decay=[0.04],
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     id=base_id+9,
#     patch_size=8,
#     nr_heads=16,
#     rank=48,
#     nr_runs=2
# )

# collect_measurements(
#     layer=lowrank,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=288,
#     depth=9,
#     lr=[8e-4],
#     bs=256,
#     max_epochs=1600,
#     weight_decay=[0.04],
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     id=base_id+9,
#     patch_size=8,
#     nr_heads=16,
#     rank=72
# )

# collect_measurements(
#     layer=lowranklight,
#     model=vit2,
#     dataset=tiny_imagenet,
#     width=224,
#     depth=9,
#     lr=[6e-4],
#     bs=256,
#     max_epochs=1600,
#     weight_decay=[0.03],
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     id=base_id+9,
#     patch_size=8,
#     nr_heads=14,
#     rank=123
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
#     id=base_id + 1,
#     patch_size=8,
#     nr_heads=12,
#     rank=128
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
#     weight_decay=wdecay,
#     lr_decay=true,
#     early_stopping=false,
#     dropout=0.1,
#     max_bs=1000,
#     id=base_id + 2,
#     patch_size=8,
#     nr_heads=12,
#     rank=96
# )


# for rank in [32, 56, 80]
# for rank in [80]

#     collect_measurements(
#         layer=blast,
#         model=vit2,
#         dataset=tiny_imagenet,
#         width=192,
#         depth=9,
#         lr=3e-3,
#         bs=256,
#         max_epochs=1600,
#         weight_decay=0.02,
#         lr_decay=true,
#         early_stopping=false,
#         dropout=0.1,
#         max_bs=1000,
#         id=base_id + 7,
#         # id=53,
#         patch_size=8,
#         nr_heads=12,
#         block_size=48,
#         rank=rank,

#     )

# end



# 22 old initialization
# 23 new my initialization
# 24 paper initialization




collect_measurements(
    layer=btt,
    model=vit2,
    dataset=tiny_imagenet,
    width=192,
    depth=9,
    lr=[3e-3],
    bs=256,
    max_epochs=1600,
    weight_decay=[0.02],
    lr_decay=true,
    early_stopping=false,
    dropout=0.1,
    max_bs=1000,
    id=39,
    patch_size=8,
    nr_heads=12,
    rank=6,
    nr_runs=2
)





wdecays = [0.0, 0.01]
epochs = 25
decay = true
base_id = 70

collect_measurements(
    layer=dense,
    model=gpt2,
    dataset=wikitext2,
    width=0,
    depth=0,
    lr=2e-5,
    bs=8,
    max_epochs=15,
    max_bs=4,
    lr_decay=false,
    weight_decay=0.01,
    id=base_id,
)



for rank in [704, 640, 512, 384]
# for rank in [128]

    collect_measurements(
        layer=lowrank,
        model=gpt2,
        dataset=wikitext2,
        width=0,
        depth=0,
        lr=1e-5,
        bs=8,
        max_epochs=25,
        max_bs=4,
        lr_decay=true,
        weight_decay=0.01,
        id=base_id+1,
        rank=rank
    )

    collect_measurements(
        layer=lowranklight,
        model=gpt2,
        dataset=wikitext2,
        width=0,
        depth=0,
        lr=[1e-5, 5e-6, 2e-6, 1e-6],
        bs=8,
        max_epochs=25,
        max_bs=4,
        lr_decay=true,
        weight_decay=0.0,
        # id=base_id+2,
        id=base_id+10,
        rank=rank
    )
    
end


for rank in [128, 256, 384]

    # collect_measurements(
    #     layer=lowrank,
    #     model=gpt2,
    #     dataset=wikitext2,
    #     width=0,
    #     depth=0,
    #     lr=5e-5,
    #     bs=8,
    #     max_epochs=40,
    #     max_bs=4,
    #     lr_decay=true,
    #     weight_decay=[0.0, 0.01],
    #     id=base_id+1,
    #     rank=rank
    # )

    collect_measurements(
        layer=lowranklight,
        model=gpt2,
        dataset=wikitext2,
        width=0,
        depth=0,
        lr=[2e-5, 1e-5, 7e-6, 5e-6],
        bs=8,
        max_epochs=25,
        max_bs=4,
        lr_decay=false,
        weight_decay=0.0,
        # id=base_id+7,
        id=base_id+12,
        rank=rank
    )

    # +7 precise projection
    # +10 new best great improved regularized projection 1e-3
    # +11 new best great improved regularized projection 1e-2
    # +12 new best great improved regularized projection 3e-3
end


for n=[2, 4]

    collect_measurements(
        layer=monarch,
        model=gpt2,
        dataset=wikitext2,
        width=0,
        depth=0,
        lr=[4e-4, 2e-4, 1e-4, 5e-5],
        bs=8,
        max_epochs=30,
        max_bs=4,
        lr_decay=false,
        weight_decay=0.0,
        id=base_id+9,
        nr_blocks=n
    )
end




collect_measurements(
    layer=unstructured,
    model=gpt2,
    dataset=wikitext2,
    width=0,
    depth=0,
    lr=1e-5,
    bs=8,
    max_epochs=10,
    max_bs=4,
    lr_decay=true,
    weight_decay=0.0,
    id=base_id+5,
    density=[50, 70]
)



collect_measurements(
    layer=unstructured,
    model=gpt2,
    dataset=wikitext2,
    width=0,
    depth=0,
    lr=[2e-4, 1e-4, 5e-5],
    bs=8,
    max_epochs=40,
    max_bs=4,
    lr_decay=true,
    weight_decay=0.005,
    id=base_id+5,
    density=30
)



collect_measurements(
    layer=blast,
    model=gpt2,
    dataset=wikitext2,
    width=0,
    depth=0,
    lr=[6e-6, 3e-6],
    bs=8,
    max_epochs=40,
    max_bs=4,
    lr_decay=false,
    weight_decay=0.0,
    id=base_id+8,
    block_size=[128, 128],
    rank=[256, 128]
)


