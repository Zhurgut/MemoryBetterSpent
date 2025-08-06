include("plot.jl")

using Latexify

function sidewaystable(file_path)
    content = read(file_path, String)
    write(file_path, replace(content, "{table}" => "{sidewaystable}"))
end

function plot_all_unique(x, y, labels, title, xlabel, ylabel)

    markers=[:circle, :square, :diamond, :utriangle, :dtriangle, :hexagon, :star5, :xcross, :cross]
    markersize = 8

    @assert length(x) == length(labels) == length(y) "Points and labels must be the same length"

    P = plot(title=title, xlabel=xlabel, ylabel=ylabel)
    n = length(x)

    for i in 1:n
        m = markers[(i - 1) % length(markers) + 1]
        scatter!(P, [x[i]], [y[i]], label=labels[i], marker=m, ms=markersize)
    end

    return P
end


function latex_table(header, caption, label, columns::AbstractVector...; adjustment=:c, format=nothing)
    n = length(columns)
    len = length(columns[1])
    @assert all(length(col)==len for col in columns) "All columns must have equal length"

    data = [columns[j][i] for i in 1:len, j in 1:n]
    
    tbl = if isnothing(format)
            latexify(data; env=:tabular, head = header, adjustment=adjustment)
        else
            latexify(data; env=:tabular, head = header, adjustment=adjustment, fmt=format)
        end
    return """
\\begin{table}[h!]
\\centering
\\tiny
\\caption{$caption}
\\label{tab:$label}
$(String(tbl))
\\end{table}
"""
end

function verbatim_string(s) # verbatim
    s = replace(s, "%"=>"\\%")
    s = replace(s, "_"=>"\\_")
    return Latexify.LaTeXString("\\text{$s}")
end

function projection_stuff(plot_out, table_out_file)

    file = "p256_0724_1913.csv"

    P = plot_projection_results(file)
    savefig(joinpath(@__DIR__, "plots_and_tables", plot_out * ".pdf"))

    df = CSV.read(joinpath(@__DIR__, "..", "measurements/projection", file), DataFrame)

    header = verbatim_string.(["matrix type", "parameters", "nr_parameters", "error (frobenius norm)"])

    table = latex_table(
        header, 
        "Projection Results",
        "projection_table",
        verbatim_string.(df.layer), 
        verbatim_string.(df.spec), 
        df.nr_parameters, 
        df.norm,
        adjustment = [:l, :l, :r, :r],
        format="%.1f"
    )

    open(joinpath(@__DIR__, "plots_and_tables", table_out_file * ".tex"), "w") do file
        write(file, table)
    end
end 

function fine_tuning_stuff(plot_out, table_out)
    ids = [70, 71, 72, 74, 75, 77, 78, 79, 80, 82]
    # lrl with precise projection in 77
    # regularized projection in 80

    root_path = joinpath(ROOT_DIR, "cscs_measurements/")

    P = plot_best(ids, root_path, small_legend=true)
    plot!(P, title="Fine-Tuning GPT-2 (124M) on wikitext-2")
    savefig(joinpath(@__DIR__, "plots_and_tables", plot_out * ".pdf"))

    P = plot_best(ids, root_path, small_legend=false)
    plot!(P, title="Fine-Tuning GPT-2 (124M) on wikitext-2")
    savefig(joinpath(@__DIR__, "plots_and_tables", plot_out * "full.pdf"))

    infos = load_measurements_infos(ids, root_path) 
    infos = infos[infos.done, :]

    @assert all(infos.model .== infos.model[1])
    @assert all(infos.dataset .== infos.dataset[1])

    ginfos = groupby(infos, [:layer])

    header = verbatim_string.(["layer type", "layer parameters", "learning rate", "weight decay", "lr-decay", "nr of epochs", "nr of parameters", "perplexity"])

    type = []
    pms = []
    lr = []
    wd = []
    lrdc = []
    epochs = []
    nr_parameters = []
    perplexity = []



    for (il, layer_infos) in enumerate(ginfos)

        np_infos = [g[argmin(g.best_test), :] for g in groupby(layer_infos, :nr_parameters)]

        vs = [:type, :pms, :lr, :wd, :lrdc, :epochs, :nr_parameters, :perplexity]
        as = [:layer, :kwargs, :lr, :wdecay, :lr_decay, :max_epochs, :nr_parameters, :best_test]

        append!(type         , [x.layer for x in np_infos])
        append!(pms          , [x.kwargs for x in np_infos])
        append!(lr           , [x.lr for x in np_infos])
        append!(wd           , [x.wdecay for x in np_infos])
        append!(lrdc         , [x.lr_decay for x in np_infos])
        append!(epochs       , [x.max_epochs for x in np_infos])
        append!(nr_parameters, [x.nr_parameters for x in np_infos])
        append!(perplexity   , [x.best_test for x in np_infos])

    end

    replace!(pms, NamedTuple()=>"")

    table = latex_table(
        header,
        "FineTuning GPT-2 on wikitext-2",
        "ft_results",
        type .|> string .|> verbatim_string,
        pms .|> string .|> verbatim_string, 
        lr,
        wd,
        lrdc .|> string,
        epochs,
        nr_parameters, 
        perplexity,
        adjustment = [:l, :l, :c, :c, :c, :r, :r, :l]
    )

    path = joinpath(@__DIR__, "plots_and_tables", table_out * ".tex")
    open(path, "w") do file
        write(file, table)
    end

    sidewaystable(path)
end


function pretraining_stuff(plot_out, table_out)
    ids = [40, 41, 42, 43, 47]

    P = plot_best(ids, joinpath(ROOT_DIR, "cscs_measurements/"), small_legend=true)
    plot!(P, title="PreTraining ViTs on Tiny-Imagenet")
    savefig(joinpath(@__DIR__, "plots_and_tables", plot_out * ".pdf"))

    P = plot_best(ids, joinpath(ROOT_DIR, "cscs_measurements/"), small_legend=false)
    plot!(P, title="PreTraining ViTs on Tiny-Imagenet")
    savefig(joinpath(@__DIR__, "plots_and_tables", plot_out * "full.pdf"))


    infos = load_measurements_infos(ids, joinpath(ROOT_DIR, "cscs_measurements/")) 
    infos = infos[infos.done, :]

    @assert all(infos.model .== infos.model[1])
    @assert all(infos.dataset .== infos.dataset[1])

    ginfos = groupby(infos, [:layer])

    header = verbatim_string.(["layer type", "layer parameters", "learning rate", "weight decay", "lr-decay", "nr of epochs", "nr of parameters", "accuracy"])

    type = []
    pms = []
    lr = []
    wd = []
    lrdc = []
    epochs = []
    nr_parameters = []
    perplexity = []



    for (il, layer_infos) in enumerate(ginfos)

        np_infos = [g[argmax(g.best_test), :] for g in groupby(layer_infos, :nr_parameters)]

        vs = [:type, :pms, :lr, :wd, :lrdc, :epochs, :nr_parameters, :perplexity]
        as = [:layer, :kwargs, :lr, :wdecay, :lr_decay, :max_epochs, :nr_parameters, :best_test]

        append!(type         , [x.layer for x in np_infos])
        append!(pms          , [x.kwargs for x in np_infos])
        append!(lr           , [x.lr for x in np_infos])
        append!(wd           , [x.wdecay for x in np_infos])
        append!(lrdc         , [x.lr_decay for x in np_infos])
        append!(epochs       , [x.max_epochs for x in np_infos])
        append!(nr_parameters, [x.nr_parameters for x in np_infos])
        append!(perplexity   , [x.best_test for x in np_infos])

    end

    replace!(pms, NamedTuple()=>"")

    table = latex_table(
        header,
        "PreTraining ViTs on Tiny-Imagenet",
        "pt_results",
        type .|> string .|> verbatim_string,
        pms .|> string .|> verbatim_string, 
        lr,
        wd,
        lrdc .|> string,
        epochs,
        nr_parameters, 
        perplexity,
        adjustment = [:l, :l, :c, :c, :c, :r, :r, :l]
    )

    path = joinpath(@__DIR__, "plots_and_tables", table_out * ".tex")
    open(path, "w") do file
        write(file, table)
    end

    sidewaystable(path)



end



begin # tensor train gpt
    table4 = [
        ("dense", "GPT-2 small", 124439808, 100, 17.55),
        ("TT, 4 cores, rank=16", "GPT-2 small TTM-16", 68085504, 54, 21.33),
        ("TT, 4 cores, rank=32", "GPT-2 small TTM-32", 71756544, 57, 21.06),
        ("TT, 4 cores, rank=64", "GPT-2 small TTM-64", 83606784, 67, 18.08),
        ("TT, 4 cores, rank=80", "GPT-2 small TTM-80", 107698944, 86, 17.61)
    ]

    table5 = [
        ("dense", "GPT-2 medium", 354823168, 100, 20.56),
        ("TT, rank=72", "GPT-2 medium TTM-72", 218303488, 61, 30.85),
        ("LowRank, rank=50", "GPT-2 medium SVD-50", 220920832, 62, 55.46),
        ("Distil GPT-2, dense", "Distil GPT-2", 81912576, 23, 51.45)
    ]

    p1 = plot_all_unique([t[3] for t in table4], [t[end] for t in table4], [t[1] for t in table4], "GPT-2 small, Training from scratch", "Nr of Parameters", "In-Domain Perplexity")
    savefig(joinpath(@__DIR__, "plots_and_tables", "tt_paper_tab4.pdf"))

    p2 = plot_all_unique([t[3] for t in table5], [t[end] for t in table5], [t[1] for t in table5], "GPT-2 medium, Training from scratch", "Nr of Parameters", "Out-Domain Perplexity")
    savefig(joinpath(@__DIR__, "plots_and_tables", "tt_paper_tab5.pdf"))



end



projection_stuff("projection", "projection_table")
fine_tuning_stuff("gpt2_results", "gpt2_results_table")
pretraining_stuff("vit_results", "vit_results_table")


plot_best([70, 71, 72, 74, 75, 77, 78, 79], joinpath(ROOT_DIR, "cscs_measurements/"))
plot_best([70, 71, 72, 74, 75, 77, 78, 79, 80, 81, 82], joinpath(ROOT_DIR, "cscs_measurements/"))

plot_best([40, 41, 42, 43, 47], joinpath(ROOT_DIR, "cscs_measurements/"))


