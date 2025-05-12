import CSV
using DataFrames

parse_eval(x) = eval(Meta.parse(x))

function load_measurements_info(id)
    WD = pwd()
    cd(joinpath(ROOT_DIR, "measurements/$id"))

    df = CSV.read("measurements_info.csv", DataFrame)
    df.layer = parse_eval.(df.layer)
    df.kwargs = parse_eval.(df.kwargs)

    cols = [:nr_parameters, :best_train, :best_test, :epoch_of_best_train, :epoch_of_best_test]
    Ts   = [Int,            Float64,      Float64,   Int,                  Int]

    for i in eachindex(cols)
        df[!, cols[i]] = Vector{Union{Missing, Ts[i]}}(df[:, cols[i]])
    end

    if "row" ∉ names(df)
        df.row = 1:size(df, 1)
    end

    if "wdecay" ∉ names(df)
        df.wdecay = zeros(Float64, size(df, 1))
    end

    if "init_scale" ∉ names(df)
        df.init_scale = ones(Float64, size(df, 1))
    end

    if "max_bs" ∉ names(df)
        df.max_bs = fill(50000, size(df, 1))
    end

    cd(WD)

    df
end

# "resize" and initialize to zero
function resize_to_zero(A, m, n)
    out = zeros(eltype(A), m, n)
    r, c = size(A)
    out[1:r, 1:c] .= A
    return out
end

function average_over_runs(df)
    if maximum(df.run_id) == 1 return df end # only one run

    gs = groupby(df, :run_id)
    sizes = [size(g, 1) for g in gs]
    N = maximum(sizes)
    matrices = [resize_to_zero(Matrix(g[:, [:train_loss, :test_loss, :train_acc, :test_acc, :time]]), N, 5) for g in gs]
    averaging_constants = [m .!= 0 for m in matrices] |> sum

    averages = sum(matrices) ./ averaging_constants

    return DataFrame([averages[:, 1], averages[:, 2], averages[:, 3], averages[:, 4], averages[:, 5]], [:train_loss, :test_loss, :train_acc, :test_acc, :time])

end


function make_dataframe(info::Measurements)

    layer = [first(x) for x in info.layers]
    kwargs = [last(x) for x in info.layers]
    
    layers = DataFrame([layer, kwargs], [:layer, :kwargs])
    widths = DataFrame([info.width], [:width])
    depths = DataFrame([info.depth], [:depth])
    lrs    = DataFrame([info.lr], [:lr])
    bss    = DataFrame([info.bs], [:bs])
    eps    = DataFrame([[info.max_epochs]], [:max_epochs])
    wdecay = DataFrame([info.weight_decay], [:wdecay])
    init_scale = DataFrame([info.init_scale], [:init_scale])
    max_bs = DataFrame([[info.max_bs]], [:max_bs])

    df = crossjoin(layers, widths, depths, lrs, bss, eps, wdecay, init_scale, max_bs)

    N = size(df, 1)

    df.row = 1:N
    df.nr_runs = fill(info.nr_runs, N)
    df.done    = fill(false, N)
    df.nr_parameters = missings(Int, N)
    df.best_train = missings(Float64, N)
    df.best_test  = missings(Float64, N)
    df.epoch_of_best_train = missings(Int, N)
    df.epoch_of_best_test  = missings(Int, N)

    return df
end


function filter_out_bad_ones(df)
    is_better_train(r1, r2) = r1.best_train > r2.best_train
    is_better_test(r1, r2)  = r1.best_test  > r2.best_test

    df.row = 1:size(df, 1)
    gs = groupby(df, [:layer, :kwargs])
    train_discards = []
    test_discards  = []
    for g in gs
        N = size(g, 1)
        for i=1:N-1, j=i:N
            if is_better_train(g[i, :], g[j, :]) push!(train_discards, g[j, :row]) end
            if is_better_train(g[j, :], g[i, :]) push!(train_discards, g[i, :row]) end

            if is_better_test(g[i, :], g[j, :]) push!(test_discards, g[j, :row]) end
            if is_better_test(g[j, :], g[i, :]) push!(test_discards, g[i, :row]) end
        end
    end

    train_ids = (x->x ∉ train_discards).(1:size(df, 1))
    test_ids  = (x->x ∉ test_discards).(1:size(df, 1))

    return df[train_ids, :], df[test_ids, :]
end

# add all measurements from df which have not been done yet to old df
# take the max of nr runs
function augment_old_measuements_info(old_df, df)
    completely_done  = antijoin(old_df, df, on=[:layer, :kwargs, :width, :depth, :lr, :bs, :wdecay, :init_scale, :max_bs]) # done, and not up for any redoing
    new_measurements = antijoin(df, old_df, on=[:layer, :kwargs, :width, :depth, :lr, :bs, :wdecay, :init_scale, :max_bs]) # need to be done no matter what
    done     = semijoin(old_df, df, on=[:layer, :kwargs, :width, :depth, :lr, :bs, :wdecay, :init_scale, :max_bs]) 
    not_done = semijoin(df, old_df, on=[:layer, :kwargs, :width, :depth, :lr, :bs, :wdecay, :init_scale, :max_bs]) 
    max_nr_runs = max.(done.nr_runs, not_done.nr_runs)
    done.done .= done.done .&& (done.nr_runs .>= not_done.nr_runs)
    done.nr_runs .= max_nr_runs

    not_new = sort(vcat(completely_done, done), :row)
    @assert all(not_new.row .== 1:size(not_new, 1))

    N = size(not_new, 1)
    new_measurements.row = N+1:N+size(new_measurements, 1)
    
    # println("\n")
    # println("old")
    # display(old_df)
    # println("new")
    # display(df)
    # println("new_measurements")
    # display(new_measurements)
    # println("not_new")
    # display(not_new)
    # println("final")
    # display(vcat(not_new, new_measurements))
    # println("\n")
    
    return vcat(not_new, new_measurements)
end


function get_time(id, row)
    WD = pwd()
    cd(joinpath(@__DIR__, "..", "measurements", "$id", "data"))
    csv = CSV.read("$row.csv", DataFrame)
    out = csv[end, :time]
    cd(WD)
    return out
end

