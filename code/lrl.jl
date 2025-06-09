using LinearAlgebra
using Plots
using CSV, DataFrames


function approximate(W, k, λ)
    N, M = size(W)
    X = W[:, 1:k]
    Y = W[:, k+1:end]

    A = W[:, 1:k] + 0.0000001*randn(N, k)
    B = inv(A' * A + λ * I) * A' * Y


    A = (X - Y*B') * inv(I - B*B')
    # println("$i: $(norm(W - [A A*B]))")
    B = inv(A' * A + λ * I) * A' * Y


    display(W)
    display([A A*B])
    display(W - [A A*B])

    display(A)
    display(B)

end


function generate_ds()

    root_dir = joinpath(@__DIR__, "..")
    if !isdir(joinpath(root_dir, "datasets"))
        mkdir(joinpath(root_dir, "datasets"))
    end

    function in_spiral(x, y)
        d = x*x + y*y
        if d > 0.9 return false end
        d = 5*d

        x, y = x*sin(-d) + y*cos(-d), x*cos(-d) - y*sin(-d)
        
        return x > 0 && -0.5x < y < x
    end

    in_circle1(x, y) = (x + 0.5)^2 + (y - 0.25)^2 < 0.2
    in_circle2(x, y) = (x + 0.1)^2 + (y - 0.25)^2 < 0.3

    in_bar(x, y) = -0.3x - 0.5 < y < -0.3x - 0.2

    points_per_class = 250
    points_per_class_test = 100

    N = 5000
    all_points = 2*rand(N, 2) .- 1
    classes = zeros(Int, N)

    classes[in_bar.(all_points[:, 1], all_points[:, 2])] .= 4
    classes[in_circle2.(all_points[:, 1], all_points[:, 2])] .= 1
    classes[in_circle1.(all_points[:, 1], all_points[:, 2])] .= 2
    
    classes[in_spiral.(all_points[:, 1], all_points[:, 2])] .= 3

    # P = scatter(all_points[classes .== 1, 1], all_points[classes .== 1, 2])
    # scatter!(P, all_points[classes .== 2, 1], all_points[classes .== 2, 2])
    # scatter!(P, all_points[classes .== 3, 1], all_points[classes .== 3, 2])
    # scatter!(P, all_points[classes .== 4, 1], all_points[classes .== 4, 2])
    # scatter!(P, all_points[classes .== 0, 1], all_points[classes .== 0, 2])

    points1 = all_points[classes .== 1, :][1:points_per_class, :]
    points2 = all_points[classes .== 2, :][1:points_per_class, :]
    points3 = all_points[classes .== 3, :][1:points_per_class, :]
    points4 = all_points[classes .== 4, :][1:points_per_class, :]

    points1_test = all_points[classes .== 1, :][points_per_class+1:points_per_class+points_per_class_test, :]
    points2_test = all_points[classes .== 2, :][points_per_class+1:points_per_class+points_per_class_test, :]
    points3_test = all_points[classes .== 3, :][points_per_class+1:points_per_class+points_per_class_test, :]
    points4_test = all_points[classes .== 4, :][points_per_class+1:points_per_class+points_per_class_test, :]

    df_train = DataFrame([points1[:, 1], points1[:, 2], fill(0, points_per_class)], [:x, :y, :label])
    append!(df_train, DataFrame([points2[:, 1], points2[:, 2], fill(1, points_per_class)], [:x, :y, :label]))
    append!(df_train, DataFrame([points3[:, 1], points3[:, 2], fill(2, points_per_class)], [:x, :y, :label]))
    append!(df_train, DataFrame([points4[:, 1], points4[:, 2], fill(3, points_per_class)], [:x, :y, :label]))
    
    df_test = DataFrame([points1_test[:, 1], points1_test[:, 2], fill(0, points_per_class_test)], [:x, :y, :label])
    append!(df_test, DataFrame([points2_test[:, 1], points2_test[:, 2], fill(1, points_per_class_test)], [:x, :y, :label]))
    append!(df_test, DataFrame([points3_test[:, 1], points3_test[:, 2], fill(2, points_per_class_test)], [:x, :y, :label]))
    append!(df_test, DataFrame([points4_test[:, 1], points4_test[:, 2], fill(3, points_per_class_test)], [:x, :y, :label]))

    CSV.write(joinpath(root_dir, "datasets", "simple_train.csv"), df_train)
    CSV.write(joinpath(root_dir, "datasets", "simple_test.csv"), df_test)

    P = scatter(points1[:, 1], points1[:, 2])
    scatter!(P, points2[:, 1], points2[:, 2])
    scatter!(P, points3[:, 1], points3[:, 2])
    scatter!(P, points4[:, 1], points4[:, 2])

    display(P)

    P = scatter(points1_test[:, 1], points1_test[:, 2])
    scatter!(P, points2_test[:, 1], points2_test[:, 2])
    scatter!(P, points3_test[:, 1], points3_test[:, 2])
    scatter!(P, points4_test[:, 1], points4_test[:, 2])

    return P
end