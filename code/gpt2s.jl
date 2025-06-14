include("run.jl")

# zero shot perplexities 90 - 92

collect_measurements(
    layer=dense,
    model=gpt2,
    dataset=wikitext2,
    width=0,
    depth=0,
    lr=1e-3,
    bs=8,
    max_epochs=0,
    max_bs=16,
    id=90,
)

for rank in [768, 764, 760, 752, 736, 704, 640, 576, 512]

    collect_measurements(
        layer=lowrank,
        model=gpt2,
        dataset=wikitext2,
        width=0,
        depth=0,
        lr=1e-3,
        bs=8,
        max_epochs=0,
        max_bs=16,
        id=91,
        rank=rank
    )

    collect_measurements(
        layer=lowranklight,
        model=gpt2,
        dataset=wikitext2,
        width=0,
        depth=0,
        lr=1e-3,
        bs=8,
        max_epochs=0,
        max_bs=16,
        id=92,
        rank=rank
    )
    
end