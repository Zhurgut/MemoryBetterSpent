import CSV
using DataFrames

parse_eval(x) = eval(Meta.parse(x))

PLOTTING = false

function load_measurements_info(id, root_path=ROOT_DIR)
    WD = pwd()
    cd(joinpath(root_path, "measurements/$id"))

    df = CSV.read("measurements_info.csv", DataFrame)
    df.layer = parse_eval.(df.layer)
    df.model = parse_eval.(df.model)
    df.dataset = parse_eval.(df.dataset)
    df.kwargs = parse_eval.(df.kwargs)

    if PLOTTING
        df.id = fill(id, nrow(df))
    end

    cols = [:nr_parameters, :best_train, :best_test, :epoch_of_best_train, :epoch_of_best_test]
    Ts   = [Int,            Float64,      Float64,   Int,                  Int]

    for i in eachindex(cols)
        df[!, cols[i]] = Vector{Union{Missing, Ts[i]}}(df[:, cols[i]])
    end

    cd(WD)

    return df
end

function load_measurements_infos(ids, root_path)
    dfs = [load_measurements_info(id, root_path) for id in ids]
    df = vcat(dfs...)
    df
end

function load_data(id, measurement_id, root_path; run_id=1)
    WD = pwd()

    try
        cd(joinpath(root_path, "measurements/$id"))

        df = CSV.read("data/$measurement_id.csv", DataFrame)
        df = df[df.run_id .== run_id, :]

        cd(WD)

        return df
    catch e
        cd(WD)
    end
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
    models = DataFrame([info.models], [:model])

    widths = DataFrame([info.width], [:width])
    depths = DataFrame([info.depth], [:depth])
    
    lrs    = DataFrame([info.lr], [:lr])
    bss    = DataFrame([info.bs], [:bs])
    
    eps    = DataFrame([info.max_epochs], [:max_epochs])
    wdecay = DataFrame([info.weight_decay], [:wdecay])

    init_scale = DataFrame([info.init_scale], [:init_scale])
    dropout = DataFrame([info.dropout], [:dropout])


    df = crossjoin(layers, models, widths, depths, lrs, bss, eps, wdecay, init_scale, dropout)

    N = size(df, 1)

    df.measurement_id = 1:N

    df.dataset = fill(info.dataset, N)
    df.lr_decay = fill(info.lr_decay, N)
    df.early_stop = fill(info.early_stopping, N)
    df.max_bs = fill(info.max_bs, N)
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

    df.run_id = 1:size(df, 1)
    gs = groupby(df, [:layer, :kwargs])
    train_discards = []
    test_discards  = []
    for g in gs
        N = size(g, 1)
        for i=1:N-1, j=i:N
            if is_better_train(g[i, :], g[j, :]) push!(train_discards, g[j, :run_id]) end
            if is_better_train(g[j, :], g[i, :]) push!(train_discards, g[i, :run_id]) end

            if is_better_test(g[i, :], g[j, :]) push!(test_discards, g[j, :run_id]) end
            if is_better_test(g[j, :], g[i, :]) push!(test_discards, g[i, :run_id]) end
        end
    end

    train_ids = (x->x ∉ train_discards).(1:size(df, 1))
    test_ids  = (x->x ∉ test_discards).(1:size(df, 1))

    return df[train_ids, :], df[test_ids, :]
end

# add all measurements from df which have not been done yet to old df
# take the max of nr runs
function augment_old_measuements_info(old_df, df)
    all_cols = [:layer, :kwargs, :model, :dataset, :width, :depth, :lr, :bs, :max_epochs, :wdecay, :init_scale, :dropout, :max_bs, :lr_decay, :early_stop]
    completely_done  = antijoin(old_df, df, on=all_cols) # all in old which are not in new, done and dusted
    new_measurements = antijoin(df, old_df, on=all_cols) # all in new which are not in old, need to be done no matter what
    
    # the only difference in these two is the nr_runs, how many runs were registered the first time, how many runs were done, maybe we want to do more runs now...
    done     = semijoin(old_df, df, on=all_cols) # all the ones we potentially already did
    not_done = semijoin(df, old_df, on=all_cols) # all the ones we want to again, or want to do more of
    
    max_nr_runs = max.(done.nr_runs, not_done.nr_runs)
    done.done .= done.done .&& (done.nr_runs .>= not_done.nr_runs)
    done.nr_runs .= max_nr_runs 

    not_new = sort(vcat(completely_done, done), :measurement_id) # all the ones we already registered, should be equal to old_df, except .done and .nr_runs columns
    @assert all(not_new.measurement_id .== 1:size(not_new, 1))

    N = size(not_new, 1)
    new_measurements.measurement_id = N+1:N+size(new_measurements, 1)
    
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


function get_time(id, run_id)
    csv = CSV.read(joinpath(@__DIR__, "..", "measurements", "$id", "data", "$run_id.csv"), DataFrame)
    out = csv[end, :time]
    return out
end

