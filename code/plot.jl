
import CSV
using DataFrames, Plots

include("run.jl")
include("utils.jl")

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
        n = kwargs.nr_cores
        r = kwargs.rank
        s = width^(1/n)
        ps = (r * s^(n+1)) * (2 + (n-2)*r)
        return round(100 * ps / dense)
    elseif layer == lowranklight
        k = kwargs.rank
        n = width
        return round(100 * (n*k + (n-k)*k) / (n*n))
    elseif layer == unstructured
        return kwargs.density
    end
    100

end


function plot_the_stuff(ids)
    # every id should include only one layer, model type and dataset, and depth. every id will give one series in the plot
    dfs = [load_measurements_info(id) for id in ids]
    for df in dfs
        @assert all(
            [row == df[1, [:layer, :model, :dataset, :depth]] for row in eachrow( df[:, [:layer, :model, :dataset, :depth]] )]
        )
    end
    series = [
        (df[1, :layer], df[1, :depth], df[end, :kwargs], df[end, :width], combine(groupby(df, :nr_parameters), :best_test => maximum)) for df in dfs
    ]

    Ps = [plot(title="depth=$i", ylims=(0.7, 1.0), xlims=(0,1000)) for i=0:2]
    for s in series 
        layer, d, kwargs, w, df = s
        plot!(Ps[d+1], df.nr_parameters, df.best_test_maximum, label="$layer $(kwargs==NamedTuple() ? "" : round(100*kwargs.rank/w))")

    end

    for P in Ps display(P) end
end


function make_label(df_row, arch=false)
    io = IOBuffer()
    cols = names(df_row)

    if "layer" in cols
        write(io, "$(df_row.layer)")

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

    if "lr" in cols
        write(io, ", lr=$(df_row.lr)")
    end

    if "wdecay" in cols && df_row.wdecay != 0.0
        write(io, ", wd=$(df_row.wdecay)")
    end

    if "init_scale" in cols && df_row.init_scale != 1.0
        write(io, ", is=$(df_row.init_scale)")
    end

    return String(take!(io))
end

general_plot_against_best(id::Int, x_col::Symbol, max_over_cols=[]) = general_plot_against_best([id], x_col, max_over_cols)

function general_plot_against_best(ids, x_col::Symbol, max_over_cols=[])
    all_cols = [:layer, :kwargs, :width, :depth, :lr, :bs, :wdecay, :init_scale, :max_epochs]
    infos = vcat([load_measurements_info(id) for id in ids]...)
    infos = infos[infos.done, :]
    gcols = [c for c in all_cols if c != x_col && c ∉ max_over_cols]
    groups = groupby(infos, gcols)
    P1 = plot(title="train", xlabel="$x_col\n", ylabel="accuracy", legend=:outerleft, size=(1600, 800))
    P2 = plot(title="test",  xlabel="$x_col\n", ylabel="accuracy", legend=:outerleft, size=(1600, 800))
    for g in groups
        rgcols = [c for c in all_cols if c ∉ max_over_cols]
        reduce_gs = groupby(g, rgcols)
        best_trains, best_tests = [], []
        display(reduce_gs)
        for rg in reduce_gs
            push!(best_trains, (rg[1, x_col], maximum(rg.best_train)))
            push!(best_tests,  (rg[1, x_col], maximum(rg.best_test)))
        end
        x, y = Plots.unzip(best_trains)
        p = sortperm(x)
        plot!(P1, x[p], y[p], label=make_label(g[1, gcols]))

        x, y = Plots.unzip(best_tests)
        p = sortperm(x)
        plot!(P2, x[p], y[p], label=make_label(g[1, gcols]))
    end
    display(P1)
    display(P2)
end

function plot_everything(id)
    WD = pwd()
    cd(joinpath(@__DIR__, "../measurements/$id"))

    info = CSV.read("measurements_info.csv", DataFrame)

    # @assert all(info.done)

    display(info)

    N = size(info, 1)
    for i=1:N
        P = plot(titlefontsize=9, ylims=(0, 2), title="$(info.layer[i]): w=$(info.width[i]), d=$(info.depth[i]), lr=$(info.lr[i]), bs=$(info.bs[i]), nr_parameters=$(info.nr_parameters[i])")
        data = groupby(CSV.read("data/$i.csv", DataFrame), :run_id)

        for g in data
            plot!(g.time, [g.train_loss, g.test_loss, g.train_acc, g.test_acc], label=["train_l$(g.run_id[1])" "test_l$(g.run_id[1])" "train_acc$(g.run_id[1])" "test_acc$(g.run_id[1])"])
        end

        display(P)
    end

    cd(WD)
end

function plot_wd(id)
    WD = pwd()
    cd(joinpath(@__DIR__, "../measurements/$id"))

    info = CSV.read("measurements_info.csv", DataFrame)

    @assert all(info.done)

    info.row = 1:size(info, 1)
    display(info)

    ginfo = groupby(info, [:layer, :kwargs, :width, :depth, :bs, :lr])

    display(ginfo)

    for g in ginfo
        P1 = plot(titlefontsize=9, size=(1200, 800), title="LOSS - $(g.layer[1]): lr=$(g.lr[1]), w=$(g.width[1]), d=$(g.depth[1]), bs=$(g.bs[1]), nr_parameters=$(g.nr_parameters[1]), $(g.kwargs[1])", xlabel="epochs")
        P2 = plot(titlefontsize=9, size=(1200, 800), title="ACC  - $(g.layer[1]): lr=$(g.lr[1]), w=$(g.width[1]), d=$(g.depth[1]), bs=$(g.bs[1]), nr_parameters=$(g.nr_parameters[1]), $(g.kwargs[1])", xlabel="epochs")
        for (i, mid) in enumerate(g.row)
            df = CSV.read("data/$mid.csv", DataFrame) |> average_over_runs # get the average over runs or something, 
            # then plot everything with the same color for one mid
            N = size(df, 1)
            plot!(P1, 1:N, df.train_loss, label="wd=$(g.wdecay[i])", color=palette(:tab10)[(i-1) % 10 + 1], alpha=0.2)
            plot!(P1, 1:N, df.test_loss,  label="wd=$(g.wdecay[i])", color=palette(:tab10)[(i-1) % 10 + 1])
            plot!(P2, 1:N, [df.train_acc, df.test_acc],   label="wd=$(g.wdecay[i])", color=palette(:tab10)[(i-1) % 10 + 1])
        end
        display(P1)
        display(P2)

        

    end

    cd(WD)
end

function plot_lr(id)
    WD = pwd()
    cd(joinpath(@__DIR__, "measurements/$id"))

    info = CSV.read("measurements_info.csv", DataFrame)

    @assert all(info.done)

    info.row = 1:size(info, 1)
    display(info)

    ginfo = groupby(info, [:layer, :kwargs, :width, :depth, :bs])

    display(ginfo)

    for g in ginfo
        P1 = plot(titlefontsize=9, size=(1200, 800), title="LOSS - $(g.layer[1]): w=$(g.width[1]), d=$(g.depth[1]), bs=$(g.bs[1]), nr_parameters=$(g.nr_parameters[1]), $(g.kwargs[1])", xlabel="epochs")
        P2 = plot(titlefontsize=9, size=(1200, 800), title="ACC  - $(g.layer[1]): w=$(g.width[1]), d=$(g.depth[1]), bs=$(g.bs[1]), nr_parameters=$(g.nr_parameters[1]), $(g.kwargs[1])", xlabel="epochs")
        for (i, mid) in enumerate(g.row)
            df = CSV.read("data/$mid.csv", DataFrame) |> average_over_runs # get the average over runs or something, 
            # then plot everything with the same color for one mid
            N = size(df, 1)
            plot!(P1, 1:N, df.train_loss, label="lr=$(g.lr[i])", color=palette(:tab10)[(i-1) % 10 + 1], alpha=0.2)
            plot!(P1, 1:N, df.test_loss,  label="lr=$(g.lr[i])", color=palette(:tab10)[(i-1) % 10 + 1])
            plot!(P2, 1:N, [df.train_acc, df.test_acc],   label="lr=$(g.lr[i])", color=palette(:tab10)[(i-1) % 10 + 1])
        end
        # display(P1)
        display(P2)

        

    end

    cd(WD)
end


function plot_bs(id)
    WD = pwd()
    cd(joinpath(@__DIR__, "measurements/$id"))

    info = CSV.read("measurements_info.csv", DataFrame)

    @assert all(info.done)

    info.row = 1:size(info, 1)
    display(info)

    ginfo = groupby(info, [:layer, :kwargs, :width, :depth, :lr])

    display(ginfo)

    for g in ginfo
        P1 = plot(titlefontsize=9, title="LOSS - $(g.layer[1]): w=$(g.width[1]), d=$(g.depth[1]), lr=$(g.lr[1]), nr_parameters=$(g.nr_parameters[1])", xlabel="time[s]")
        P2 = plot(titlefontsize=9, title="ACC  - $(g.layer[1]): w=$(g.width[1]), d=$(g.depth[1]), lr=$(g.lr[1]), nr_parameters=$(g.nr_parameters[1])", xlabel="time[s]")
        for (i, mid) in enumerate(g.row)
            df = CSV.read("data/$mid.csv", DataFrame) |> average_over_runs # get the average over runs or something, 
            # then plot everything with the same color for one mid
            N = size(df, 1)
            plot!(P1, df.time, df.train_loss, label="bs=$(g.bs[i])", color=palette(:tab10)[i], alpha=0.5)
            plot!(P1, df.time, df.test_loss,  label="bs=$(g.bs[i])", color=palette(:tab10)[i])
            plot!(P2, df.time, [df.train_acc, df.test_acc],   label="bs=$(g.bs[i])", color=palette(:tab10)[i])
        end
        display(P1)
        display(P2)

        

    end

    cd(WD)
end

function plot_interpolation_threshold(id)
    WD = pwd()
    cd(joinpath(@__DIR__, "measurements/$id"))

    info = CSV.read("measurements_info.csv", DataFrame)
    info = info[info.done, :]

    @assert all(info.done)

    info.row = 1:size(info, 1)
    display(info)

    data = DataFrame([[], [], [], [], [], [], []], [:layer, :lr, :bs, :nr_parameters, :train_acc, :test_acc, :time])

    for row in eachrow(info)
        df = CSV.read("data/$(row.row).csv", DataFrame) |> average_over_runs

        idx = argmax(df.train_acc)
        tra = df.train_acc[idx]
        tsa = df.test_acc[idx]
        time = idx

        push!(data, (layer=row.layer, lr=row.lr, bs=row.bs, nr_parameters=row.nr_parameters, train_acc=tra, test_acc=tsa, time=time))
    end
    # display(data)
    data.time *= (1/maximum(data.time))

    

    gs = groupby(data, [:lr, :bs])

    for g in gs
        P = plot(title="lr=$(g.lr[1]), bs=$(g.bs[1])", size=(1000, 600))
        ls = groupby(g, :layer)
        for (i, l) in enumerate(ls)
            plot!(P, l.nr_parameters, [l.train_acc, l.test_acc, l.time], 
                label=["$(l.layer[1]) train_acc" "test_acc" "epochs"], 
                color=palette(:tab10)[i],
                linestyle=[:solid :dash :dot],
                legend_position=:outerleft,
                ylims = (0,1)
            )
        end
        display(P)
    end

    cd(WD)

end

# max over nr_parameters
# look at all measurements in ids for this, fix architecture (width and depth), take the best over all hyperparameters
function plot_best(ids, root_path = ROOT_DIR)

    infos = load_measurements_infos(ids, root_path) 
    infos = infos[infos.done, :]

    @assert all(infos.model .== infos.model[1])
    @assert all(infos.dataset .== infos.dataset[1])

    ginfos = groupby(infos, [:layer])

    shapes =  [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star8, :+, :x]

    gpt = infos.model[1] == gpt2 || infos.model[1] == distil_gpt2

    P = if gpt
        plot(xlabel="nr parameters\n ", ylabel="test perplexity", legend_position=:outerleft, size=(1400, 800))
    else
        plot(xlabel="nr parameters\n ", ylabel="test accuracy", legend_position=:outerleft, size=(1400, 800))
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
        plot!(P, X[perm], y[perm], label=nothing, color=palette(:tab10)[il])

        perm = sortperm(X, rev=true)

        for (ir, r) in enumerate(np_infos[perm])

            scatter!(P, [r.nr_parameters], [r.best_test], label=make_label(r), color=palette(:tab10)[il], markershape=shapes[(ir-1) % length(shapes) + 1], markersize=8)

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

    # Create the plot
    p = plot(size=(800, 600))  # Initialize the plot

    # Loop through each group (each label) and plot it
    for group in grouped_df
        label_name = group[1, :layer]  # Get the label for the group
        plot!(p, group.nr_parameters, group.op_norm, label=label_name, xlabel="nr parameters\n", ylabel="\nFrobenius Norm || W - W' ||")
    end

    # Display the plot
    savefig(filename * ".svg")

    p
end


function plot_runtimes(id::Int)
    info = load_measurements_info(id)
    by_bs = groupby(info, :bs)
    by_bs = [by_bs[key] for key in keys(by_bs)]
    P = plot(title="runtimes", size(800, 600), ylims=(0, 600))
    
    for df in by_bs
        by_lr = groupby(df, :layer)
        dense = by_lr[(vit_dense,)]
        t = get_time(id, dense[1, :row])
        hline!(P, [t], label="dense bs=$(dense.bs[1])")
        times = get_time.(id, by_lr[(vit_lowranklight,)].row)
        nr_params = by_lr[(vit_lowranklight,)].nr_parameters
        plot!(P, nr_params, times, label="bs=$(dense.bs[1])")
    end
    P
end


function plot_profiling(filename)
    df = CSV.read(joinpath(@__DIR__, "..", "measurements/profiling", filename), DataFrame)
    by_bs = groupby(df, :batch_size)
    by_model = groupby(df, [:model, :nr_parameters])
    for g in by_bs
        println(g[findall(==("lowrank_light"), g.model), :])
        l = g[findall(==("lowrank_light"), g.model), :]
        P = plot(l.nr_parameters, [l.cpu_forward_us l.cpu_backward_us l.gpu_forward_us], label=["lrl cpu fw" "lrl bw" "lrl gpu fw"], title="$(g[1, :batch_size])", 
            ylims=(0, max(maximum.((df.cpu_forward_us, df.cpu_backward_us, df.gpu_forward_us))...))
        )
        display(g[1, :])
        d = g[findfirst(==("dense"), g.model), :]
        hline!(P, [d.cpu_forward_us d.cpu_backward_us d.gpu_forward_us], label=["dense cpu fw" "dense bw" "dense gpu fw"])
        display(P)
    end
end