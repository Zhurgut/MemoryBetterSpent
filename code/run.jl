
using JSON3
import CSV
using DataFrames
using Statistics: mean

const ROOT_DIR = joinpath(@__DIR__, "..")

@enum Layer dense lowrank lowranklight monarch kronecker tt btt
@enum Model mlp mlp2 b_mlp b_mlp2 vit vit2 
@enum Dataset simple cifar10 tiny_imagenet

struct Measurements
    layers::Vector{Tuple{Layer, NamedTuple}}
    models::Vector{Model}
    dataset::Dataset

    width::Vector{Int}
    depth::Vector{Int}

    lr::Vector{Float64}
    bs::Vector{Int}

    max_epochs::Vector{Int}
    weight_decay::Vector{Float64}

    lr_decay::Bool
    early_stopping::Bool

    init_scale::Vector{Float64}
    dropout::Vector{Float64}

    max_bs::Int
    
    nr_runs::Int
end




include("utils.jl")





function train(
        layer::Layer, model::Model, dataset::Dataset, 
        width::Int, depth::Int, 
        lr::Float64, bs::Int, max_epochs::Int, weight_decay::Float64,
        lr_decay::Bool, early_stopping::Bool, 
        init_scale::Float64, max_bs::Int, dropout::Float64
        ; kwargs...)
    WD = pwd()
    cd(joinpath(@__DIR__, "Python"))

    to_string(;kwargs...) = isempty(kwargs) ? nothing : string(values(kwargs))[2:end-1] |> x->split(x, ",") .|> filter(!isspace) |> filter(!isempty)

    params = to_string(;kwargs...)

    if params |> isnothing
        cmd = `python main.py -l $(layer) -m $(model) -ds $(dataset) -w $width -d $depth -lr $lr -bs $bs -e $max_epochs -wd $weight_decay --lr_decay $(lr_decay) --early_stopping $(early_stopping) -s $init_scale --dropout $(dropout) --max_bs $max_bs`
    else
        cmd = `python main.py -l $(layer) -m $(model) -ds $(dataset) -w $width -d $depth -lr $lr -bs $bs -e $max_epochs -wd $weight_decay --lr_decay $(lr_decay) --early_stopping $(early_stopping) -s $init_scale --dropout $(dropout) --max_bs $max_bs -p $params`
    end

    println(cmd)

    local output
    try
        output = read(cmd, String)
    catch e 
        cd(WD)
        println("something went wrong during training, sorry :(")
        rethrow(e)
    end
    # println(output)

    local data
    try
        data = JSON3.read(output)
    catch e
        cd(WD)
        println("couldnt read json")
        println(output)
        rethrow(e)
    end

    cd(WD)

    return data
end


function collect_measurements(;
        layer=dense,
        model=mlp,
        dataset=cifar10,
        width=1024,
        depth=4,
        lr=1e-4,
        bs=1000,
        max_epochs=500,
        weight_decay=0.0,
        lr_decay=true,
        early_stopping=true,
        init_scale=1.0,
        dropout=0.2,
        nr_runs=1,
        max_bs=50000,
        id=nothing,
        kwargs... 
    )

    if isnothing(id)
        if !isdir(joinpath(ROOT_DIR, "measurements"))
            mkdir(joinpath(ROOT_DIR, "measurements"))
        end
        
        ids = parse.(Int, cd(readdir, joinpath(ROOT_DIR, "measurements")))
        if length(ids) == 0
            id = 0
        else
            id = maximum(ids) + 1
        end
    end

    collect_measurements(
        layer, model, dataset,
        width, depth,
        lr, bs, max_epochs, weight_decay,
        lr_decay, early_stopping,
        init_scale,
        dropout,
        nr_runs,
        max_bs,
        NamedTuple{keys(kwargs)}(values(kwargs)),
        id=id
    )
end


function collect_measurements(
        layer::Union{Layer, Vector{Layer}}, 
        model::Union{Model, Vector{Model}},
        dataset::Dataset,
        width::Union{Int, Vector{Int}}, 
        depth::Union{Int, Vector{Int}}, 
        lr::Union{Float64, Vector{Float64}}, 
        bs::Union{Int, Vector{Int}}, 
        max_epochs::Union{Int, Vector{Int}}, 
        weight_decay::Union{Float64, Vector{Float64}},
        lr_decay::Bool,
        early_stopping::Bool,
        init_scale::Union{Float64, Vector{Float64}},
        dropout::Union{Float64, Vector{Float64}},
        nr_runs::Int, # how many runs to average over
        max_bs::Int,
        kwargs::Union{NamedTuple, Vector{T}} = NamedTuple();
        id::Union{Int, Nothing}=nothing # associate measurements with some id that maybe already exists
    ) where T <: NamedTuple

    vec(x) = x isa Vector ? x : [x]

    @assert length(vec(layer)) == length(vec(kwargs))

    collect_measurements(Measurements(
        collect(zip(vec(layer), vec(kwargs))), 
        vec(model),
        dataset,
        vec(width), vec(depth), 
        vec(lr), vec(bs), 
        vec(max_epochs), vec(weight_decay), 
        lr_decay, early_stopping,
        vec(init_scale), vec(dropout), 
        max_bs, nr_runs
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
    out = DataFrame(
        [fill(run, N), data.train_losses, data.test_losses, data.train_accuracies, data.test_accuracies, data.time], 
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

    # work through all the train commands we have to do, and store measurements in data folder
    current_measurement_id = 1
    while current_measurement_id <= size(df, 1)
        
        if df.done[current_measurement_id] 
            current_measurement_id += 1
            continue 
        end
        
        measurement_id = df.measurement_id[current_measurement_id]
        @assert measurement_id == current_measurement_id # should be the same thing?
        
        try
            run = 1

            if isfile("data/$measurement_id.csv")
                results = CSV.read("data/$measurement_id.csv", DataFrame)
                run = maximum(results.run_id) + 1
            end

            df_row = df[measurement_id, :]

            while run <= df.nr_runs[measurement_id]

                # do the training
                data = train(
                    df_row.layer, 
                    df_row.model, 
                    df_row.dataset,
                    df_row.width, 
                    df_row.depth, 
                    df_row.lr, 
                    df_row.bs, 
                    df_row.max_epochs, 
                    df_row.wdecay, 
                    df_row.lr_decay, 
                    df_row.early_stop,
                    df_row.init_scale, 
                    df_row.max_bs, 
                    df_row.dropout
                    ;df_row.kwargs...)

                save_results("data/$measurement_id.csv", data, run)

                update_measurements_info!(df, measurement_id, data.nr_parameters)

                CSV.write("measurements_info.csv", df)

                run += 1
            end

            df.done[measurement_id] = true

            CSV.write("measurements_info.csv", df)

            
        catch e 
            if e isa InterruptException
                cd(WD)
                rethrow(e)
            else
                println("failed to collect measurements for:\n", df[current_measurement_id, :])
                println(e)
                # continue to next one
                
                # rethrow(e)
            end
        end
        current_measurement_id += 1
    end

    cd(WD)
end

# # when there was some issue, or out of time or whatever, can just continue on from where we left off
# function resume_collecting(id)
#     df = load_measurements_info(id)
#     collect_measurements(df, id)
# end
