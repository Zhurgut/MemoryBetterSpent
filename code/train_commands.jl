
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
