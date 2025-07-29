include("run.jl")


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


