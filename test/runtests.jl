using DirectIteration
using Test

using SSMProblems
using Random
using Distributions
using AbstractMCMC
using MCMCChains
using AdvancedHMC

import ForwardDiff

# factor model definition
include("model.jl")

# to compare the Turing implementation (not yet though...)
# include("turing.jl")

NX = 2
T = 250

rng = MersenneTwister(1234);
true_λs = rand(rng, λ_prior(NX, NX));
true_ssm = factor_model(0.2, true_λs, (NX, NX));
x0, xs, ys = SSMProblems.sample(rng, true_ssm, T);

@testset "direct iteration" begin
    # create the model
    ssmprob = StateSpaceProblem(ys, (1, NX), θ -> factor_model(0.2, θ, (NX, NX)))

    # sample a random draw from the "prior"
    initial_params = begin
        λs = rand(λ_prior(NX, NX))
        x0, xs, _ = sample(rng, ssmprob(λs), T)
        [λs; x0; vec(hcat(xs...))]
    end

    # run the sampler
    rng = MersenneTwister(1234)
    chain = AbstractMCMC.sample(
        rng,
        LogDensityModel(ssmprob),
        AdvancedHMC.NUTS(0.8),
        250 + 500;
        n_adapts = 250,
        initial_params,
        chain_type = Chains
    )
end
