
import CSV
using DataFrames, Plots

include("run.jl")
include("utils.jl")

PLOTTING = true

function density(layer::Layer, width, kwargs)
    if width == 0
        width=768 # hack for gpt2
    end
    dense = width * width
    if layer == lowrank
        r = kwargs.rank
        return round(100 * 2 * r * width / dense)
    elseif layer == monarch
        n = kwargs.nr_blocks
        bs = width / n
        return round(100 * (n * bs * bs + n * n * bs) / dense)
    elseif layer == tt
        n = kwargs.nr_cores
        r = kwargs.rank
        s = width^(1/n)
        ps = (r * s * s) * (2 + (n-2)*r)
        return round(100 * ps / dense)
    elseif layer == kronecker
        return round(100 * 2 * width / dense)
    elseif layer == btt 
        n = 2
        r = kwargs.rank
        s = width^(1/n)
        ps = (r * s^(n+1)) * (2 + (n-2)*r)
        return round(100 * ps / dense)
    elseif layer == lowranklight
        k = kwargs.rank
        n = width
        return round(100 * (n*k + (n-k)*k) / dense)
    elseif layer == unstructured
        return kwargs.density
    elseif layer == blast
        bs = kwargs.block_size
        b = width ÷ bs
        r = kwargs.rank
        return round(100 * ((2width + b*b)*r) / dense)
    end
    100

end



function layer_name(layer::Layer)
    if layer == dense return "Dense" end
    if layer == lowrank return "Low-Rank" end
    if layer == lowranklight return "Low-Rank Light" end
    if layer == monarch return "Monarch" end
    if layer == kronecker return "Kronecker" end
    if layer == tt return "TT" end
    if layer == btt return "BTT" end
    if layer == blast return "BLAST" end
    if layer == unstructured return "Unstructured" end
end



function make_label(df_row, arch=false)
    io = IOBuffer()
    cols = names(df_row)

    # write(io, "($(df_row.id))") # measurement id
    # write(io, "($(df_row.id)[[$(df_row.nr_runs)]])") # measurement id

    if "layer" in cols
        write(io, " $(layer_name(df_row.layer))")

        if df_row.layer != dense && "width" in cols && "kwargs" in cols
            write(io, " [$(density(df_row.layer, df_row.width, df_row.kwargs) |> Int)%]")
        end
    end

    if arch && "width" in cols && "depth" in cols
        write(io, ", $(df_row.width)-$(df_row.depth)")
    end

    # if "kwargs" in cols && df_row.kwargs != NamedTuple()
    #     write(io, ", $(df_row.kwargs)")
    # end

    # if "lr" in cols
    #     write(io, ", lr=$(df_row.lr)")
    # end

    # if "wdecay" in cols && df_row.wdecay != 0.0
    #     write(io, ", wd=$(df_row.wdecay)")
    # end

    # if "init_scale" in cols && df_row.init_scale != 1.0
    #     write(io, ", is=$(df_row.init_scale)")
    # end

    # if "lr_decay" in cols
    #     write(io, ", lrdc=$(df_row.lr_decay)")
    # end

    # if "max_epochs" in cols
    #     write(io, ", epochs=$(df_row.max_epochs)")
    # end


    return String(take!(io))
end


# max over nr_parameters
# look at all measurements in ids for this, fix architecture (width and depth), take the best over all hyperparameters
function plot_best(ids, root_path = ROOT_DIR; small_legend=false)

    infos = load_measurements_infos(ids, root_path) 
    infos = infos[infos.done, :]

    @assert all(infos.model .== infos.model[1])
    @assert all(infos.dataset .== infos.dataset[1])

    ginfos = groupby(infos, [:layer])

    shapes =  [:rect, :circle, :star5, :diamond, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star8, :+, :x]

    gpt = infos.model[1] == gpt2 || infos.model[1] == distil_gpt2

    P = if gpt
        if small_legend
            plot(xlabel="# Parameters\n ", ylabel="\nPerplexity", legend_position=:topright, size=(800, 600))
        else
            # plot(xlabel="# Parameters\n ", ylabel="Perplexity", legend_position=:outerleft, size=(1400, 800))
            plot(xlabel="# Parameters\n ", ylabel="\nPerplexity", legend_position=:topright, size=(800, 600))
        end
    else
        if small_legend
            plot(xlabel="# Parameters\n ", ylabel="\nTest-Accuracy", legend_position=:bottomright, size=(800, 600))
        else
            # plot(xlabel="# Parameters\n ", ylabel="Test-Accuracy", legend_position=:outerleft, size=(1400, 800))
            plot(xlabel="# Parameters\n ", ylabel="\nTest-Accuracy", legend_position=:bottomright, size=(800, 600))
        end
    end

    
    for (il, layer_infos) in enumerate(ginfos)


        np_infos = if gpt
            [g[argmin(g.best_test), :] for g in groupby(layer_infos, :nr_parameters)]
        else
            [g[argmax(g.best_test), :] for g in groupby(layer_infos, :nr_parameters)]
        end

        X = [x.nr_parameters for x in np_infos]
        y = [x.best_test for x in np_infos]
        perm = sortperm(X)

        if small_legend
            plot!(P, X[perm], y[perm], label=layer_name(layer_infos.layer[1]), color=palette(:tab10)[il])
        else
            plot!(P, X[perm], y[perm], label=nothing, color=palette(:tab10)[il])
        end

        perm = sortperm(X, rev=true)

        for (ir, r) in enumerate(np_infos[perm])
            
            if small_legend
                scatter!(P, [r.nr_parameters], [r.best_test], label=nothing, color=palette(:tab10)[il], markershape=shapes[(ir-1) % length(shapes) + 1], markersize=8)
            else
                scatter!(P, [r.nr_parameters], [r.best_test], label=make_label(r), color=palette(:tab10)[il], markershape=shapes[(ir-1) % length(shapes) + 1], markersize=8)
            end
        end

    end

    P

end



function plot_projection_results(filename)
    # Load the data from the CSV file
    df = CSV.read(joinpath(@__DIR__, "..", "measurements/projection", filename), DataFrame)
    df = df[df.layer .!= "dense", :]
    # Group by label (no aggregation, just to separate the data for each label)
    grouped_df = groupby(df, :layer)

    p = plot(size=(800, 600), title="Projection")  # Initialize the plot

    # Loop through each group (each label) and plot it
    for group in grouped_df
        label_name = group[1, :layer]  # Get the label for the group
        plot!(p, group.nr_parameters, group.norm, label=label_name, xlabel="# Parameters\n", ylabel="\nApproximation Error ‖ W - M ‖")
    end

    # Display the plot
    # savefig(filename * ".svg")

    p
end
