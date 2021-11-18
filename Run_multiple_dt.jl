import Pkg
Pkg.add("Turing")
Pkg.add("DifferentialEquations")
Pkg.add("Distributions")
Pkg.add("StatsPlots")
Pkg.add("Plots")
Pkg.add("ReverseDiff")
Pkg.add("Memoization")
Pkg.add("DelimitedFiles")

###################################################

using Turing, ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
using Distributions
using LinearAlgebra

# Ornstein-Uhlenbeck process
@model ou(rn,T,delta_t) = begin
    ampl ~ Uniform(0.0,5.0)
    tau ~ Uniform(0.0,5.0)

    b = exp(-delta_t/tau)

    rn[1] ~ Normal(0,sqrt(ampl))

    for i=2:T
        rn[i] ~ Normal(rn[i-1]*b,sqrt(ampl*(1-b^2)))
    end
end

# Ornstein-Uhlenbeck process with added Gaussian noise
@model oupn(rn,T,delta_t,::Type{R}=Vector{Float64}) where {R} = begin
    ampl ~ Uniform(0.0,2.0)
    tau ~ Uniform(0.1,2.0)
    noise_ampl ~ Uniform(0.0,0.5)

    b = exp(-delta_t/tau)
    r = R(undef, T)

    r[1] ~ Normal(0,sqrt(ampl))

    for i=2:T
        r[i] ~ Normal(r[i-1]*b,sqrt(ampl*(1-b^2)))
    end
    rn ~ MvNormal(r,sqrt.(abs.(r)).*noise_ampl)
end

###################################################

using DifferentialEquations
using Plots
using StatsPlots
using Turing, ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

using Distributions, Random
using LinearAlgebra

# Sets up ou process, all of these values are held constant
μ = 0.0
σ = sqrt(2)
Θ = 1.0
W = OrnsteinUhlenbeckProcess(Θ,μ,σ,0.0,1.0)
prob = NoiseProblem(W,(0.0,100.0))

# Number of dt values the loop will go through
n_dt_vals = 30
# This creates a linspace array between 0.05 and 2 with the number of elements that was specified above
dt_vals = range(0.05, 2, length = n_dt_vals)

# Holds: dt, means the std deviation of ampl, tau, noise_ampl of them in that order
parameter_data = zeros(n_dt_vals, 7)

# For loop that iterates through each value for dt
for i in 1:n_dt_vals
    # Sets dt to the correct value and stores it in the array
    dt = dt_vals[i]
    parameter_data[i, 1] = dt

    # Returns the ou process with this dt value
    sol = solve(prob;dt=dt)

    # Stores the true ou data and creates noise
    ou_data = sol.u
    noise = rand.(Normal.(0,0.2*sqrt.(abs.(ou_data))))

    # Adds the true data and noise together, then samples the distribtution for the parameters
    data = ou_data .+ noise
    @time chnpn = sample(oupn(data,length(data),dt), NUTS(0.65), 2000)

    parameter_data[i, 2] = mean(chnpn[:ampl][:,1,1])
    parameter_data[i, 3] = std(chnpn[:ampl][:,1,1])
    parameter_data[i, 4] = mean(chnpn[:tau][:,1,1])
    parameter_data[i, 5] = std(chnpn[:tau][:,1,1])
    parameter_data[i, 6] = mean(chnpn[:noise_ampl][:,1,1])
    parameter_data[i, 7] = std(chnpn[:noise_ampl][:,1,1])
end


using DelimitedFiles

writedlm("parameter_data.csv", parameter_data, ",")
