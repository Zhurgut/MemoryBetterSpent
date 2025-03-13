using Plots
using Roots

# Define the baseline function f(x)
function f(x)
    if x < 0 || x > 1
        return NaN  # Undefined outside [0,1]
    end
    sqrt2 = sqrt(2)
    return 1 - 0.5 * (1 / (2 * x + sqrt2 - 1) - sqrt2 + 1)
end

# Function to compute c such that f(s - c) = a
function compute_c(s, a)
    # Solve f(s - c) - a = 0 for c
    # Let t = s - c, so f(t) = a, then c = s - t
    t = find_zero(x -> f(x) - a, (0, 1))  # Find where f(x) = a
    c = s - t
    return c
end

# Fake datapoints: (s, a) pairs
models = [
    (0.2, 1.0, "Model 1 -> 5"),
    (0.5, 1.0, "Model 2 -> 2"),
    (0.8, 0.8, "Model 3 -> bad"),
    (0.9, 0.8, "Model 4 -> worse"),
    (0.03, 0.4, "Model 5 similar to 6"),
    (0.04, 0.4, "Model 6 similar to 5"),
]

# Generate x values for the original curve
x_vals = 0:0.01:1
y_vals = [f(x) for x in x_vals]

function score(x, y)
    c = compute_c(x, y)
    z = c + 1
    1/z
end

# Create a plot for each model
for (s, a, label) in models
    # Compute the shift c
    c = compute_c(s, a)
    
    # Define x range for shifted curve
    # Since f(x - c) requires x - c in [0,1], x should be in [c, c+1]
    # Limit to [0,1.5] for visualization consistency
    x_shifted = range(max(0, c), min(1.5, c + 1), length=100)
    y_shifted = [f(x - c) for x in x_shifted]
    
    # Find where shifted curve hits y = 1
    z = c + 1  # Since f(1) = 1, f(z - c) = 1 when z - c = 1
    
    # Create the initial plot
    p = plot(x_vals, y_vals, label="f(x)", xlabel="Proportion of Parameters", ylabel="Accuracy", 
             title="$label (s=$s, a=$a)", linewidth=2, xlims=(0, 1.5), ylims=(0, 1.1))
    
    # Add the shifted curve to the same plot
    plot!(p, x_shifted, y_shifted, label="f(x - $c)", linestyle=:dash, linewidth=2)
    
    # Mark the model's point (s, a)
    scatter!(p, [s], [a], label="($s, $a)", markersize=6, color=:red)
    
    # Mark where shifted curve hits y=1, if within plot range
    if 0 <= z <= 1.5
        scatter!(p, [z], [1], label="z=$z", markershape=:star, markersize=8, color=:green)
    end
    
    # Display the plot
    display(p)
    
    # Print some info
    println("$label: c = $c, z = $z, score = $(1/z)")
end