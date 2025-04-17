
using JSON3
import CSV
using DataFrames
using Statistics: mean

const ROOT_DIR = joinpath(@__DIR__, "..")

@enum Layer dense lowrank lowranklight monarch kronecker tt btt vit_dense vit_lowrank vit_lowranklight vit_monarch vit_kronecker vit_tt vit_btt

struct Measurements
    layers::Vector{Tuple{Layer, NamedTuple}}

    width::Vector{Int}
    depth::Vector{Int}

    lr::Vector{Float64}
    bs::Vector{Int}

    init_scale::Vector{Float64}

    max_epochs::Int
    weight_decay::Vector{Float64}

    max_bs::Int
    
    nr_runs::Int
end




include("utils.jl")





function train(layer::Layer, width::Int, depth::Int, lr::Float64, bs::Int, max_epochs::Int, weight_decay::Float64=0.0, init_scale=1.0, max_bs=50000; kwargs...)
    WD = pwd()
    cd(joinpath(@__DIR__, "Python"))

    to_string(;kwargs...) = isempty(kwargs) ? nothing : string(values(kwargs))[2:end-1] |> x->split(x, ",") .|> filter(!isspace) |> filter(!isempty)

    params = to_string(;kwargs...)

    if params |> isnothing
        cmd = `python main.py -l $(layer) -w $width -d $depth -lr $lr -bs $bs -e $max_epochs -wd $weight_decay -s $init_scale --max_bs $max_bs`
    else
        cmd = `python main.py -l $(layer) -w $width -d $depth -lr $lr -bs $bs -e $max_epochs -wd $weight_decay -s $init_scale --max_bs $max_bs -p $params`
    end

    println(cmd)

    local output
    try
        output = read(cmd, String)
    catch e 
        cd(WD)
        println("probably out of gpu memory...")
        rethrow(e)
    end
    # println(output)

    data = JSON3.read(output)

    cd(WD)

    return data
end



function collect_measurements(
        layer::Union{Layer, Vector{Layer}}, 
        width::Union{Int, Vector{Int}}, 
        depth::Union{Int, Vector{Int}}, 
        lr::Union{Float64, Vector{Float64}}, 
        bs::Union{Int, Vector{Int}}, 
        max_epochs::Int, 
        nr_runs::Int, # how many runs to average over
        kwargs::Union{NamedTuple, Vector{T}} = NamedTuple();
        weight_decay::Union{Float64, Vector{Float64}}=0.0,
        init_scale::Union{Float64, Vector{Float64}}=1.0,
        max_bs::Int=50000,
        id=nothing # associate measurements with some id that already exists
    ) where T <: NamedTuple



    vec(x) = x isa Vector ? x : [x]

    collect_measurements(Measurements(
        collect(zip(vec(layer), vec(kwargs))), # layers and kwargs need to have the same lengths
        vec(width), vec(depth), 
        vec(lr), vec(bs), vec(init_scale),
        max_epochs, vec(weight_decay), max_bs, nr_runs
    ), id)

end




function collect_measurements(info::Measurements, id=nothing)
    WD = pwd()
    cd(ROOT_DIR)

    # initialize measurements directory if does not exist
    if !isdir("measurements") mkdir("measurements") end

    if isnothing(id)
        ids = parse.(Int, cd(readdir, "measurements"))
        if length(ids) == 0
            id = 0
        else
            id = maximum(ids) + 1
        end
    end

    if !isdir("measurements/$id") mkdir("measurements/$id") end # make a folder for new measurements

    cd("measurements/$id")

    # make a dataframe with all the train commands we have to run, based on the inputs
    df = make_dataframe(info)

    display(df)

    if isfile("measurements_info.csv") 
        old_df = load_measurements_info(id)
        df = augment_old_measuements_info(old_df, df)

        # println(df)
        # error("hello")
    end

    CSV.write("measurements_info.csv", df)

    collect_measurements(df, id)

    cd(WD)
end

function save_results(path, data, run)
    N = data.nr_epochs
    times = collect(LinRange(0, data.time, N+1))[2:end]
    out = DataFrame(
        [fill(run, N), data.train_losses, data.test_losses, data.train_accuracies, data.test_accuracies, times], 
        [:run_id, :train_loss, :test_loss, :train_acc, :test_acc, :time]
    )

    results = if isfile(path)
        CSV.read(path, DataFrame)
    else 
        DataFrame([[], [], [], [], [], []], [:run_id, :train_loss, :test_loss, :train_acc, :test_acc, :time])
    end

    append!(results, out)
    CSV.write(path, results)
end


# assumes we are in a folder with a measurements_info.csv and a folder data
function update_measurements_info!(df, idx, nr_parameters=nothing)
    
    if !(nr_parameters |> isnothing)
        df.nr_parameters[idx] = nr_parameters
    end

    results = CSV.read("data/$idx.csv", DataFrame)
    nr_runs = maximum(results.run_id)

    best_train = []
    best_test = []
    epoch_of_best_test = []
    epoch_of_best_train = []

    g = groupby(results, :run_id)

    for run = 1:nr_runs
        push!(best_train, maximum(g[(run,)].train_acc))
        push!(best_test, maximum(g[(run,)].test_acc))
        push!(epoch_of_best_train, argmax(g[(run,)].train_acc))
        push!(epoch_of_best_test, argmax(g[(run,)].test_acc))
    end

    df.best_train[idx]          = mean(best_train)
    df.best_test[idx]           = mean(best_test)
    df.epoch_of_best_train[idx] = mean(epoch_of_best_test)  |> round |> Int
    df.epoch_of_best_test[idx]  = mean(epoch_of_best_train) |> round |> Int

end



function collect_measurements(df, id)
    WD = pwd()
    cd(joinpath(ROOT_DIR, "measurements/$id"))


    if !isdir("data") mkdir("data") end

    # then work through all the train commands we have to do, and store measurements in data folder
    current_row_idx = 1
    while current_row_idx <= size(df, 1)
        
        if df.done[current_row_idx] 
            current_row_idx += 1
            continue 
        end
        
        row = df.row[current_row_idx]
        
        try
            run = 1
            if isfile("data/$row.csv")
                results = CSV.read("data/$row.csv", DataFrame)
                run = maximum(results.run_id) + 1
            end

            while run <= df.nr_runs[row]

                # do the training
                data = train(df.layer[row], df.width[row], df.depth[row], df.lr[row], df.bs[row], df.max_epochs[row], df.wdecay[row], df.init_scale[row], df.max_bs[row]; df.kwargs[row]...)

                save_results("data/$row.csv", data, run)

                update_measurements_info!(df, row, data.nr_parameters)

                CSV.write("measurements_info.csv", df)

                run += 1
            end

            df.done[row] = true

            CSV.write("measurements_info.csv", df)

            
        catch e 
            if e isa InterruptException
                cd(WD)
                rethrow(e)
            else
                println("failed to collect measurements for:\n", df[current_row_idx, :])
                println(e)
                # continue to next one
                
                # rethrow(e)
            end
        end
        current_row_idx += 1
    end

    cd(WD)
end

# when there was some issue, or out of time or whatever, can just continue on from where we left off
function resume_collecting(id)
    df = load_measurements_info(id)
    collect_measurements(df, id)
end
