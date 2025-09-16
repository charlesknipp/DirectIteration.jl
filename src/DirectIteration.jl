module DirectIteration

# for SSM model definition we require the following
using SSMProblems, GeneralisedFilters
using Random
using OffsetArrays
const GF = GeneralisedFilters

# for more efficient HMC integration, use LogDensityProblems
using LogDensityProblems, LogDensityProblemsAD
using AbstractMCMC, MCMCChains

# overload for conveneince
using AbstractMCMC: LogDensityModel

export StateSpaceProblem, LogDensityModel

## STATE SPACE PROBLEM #####################################################################

"""
    StateSpaceProblem(model, obs, dims)

A simple object used primarily for dispatch on a state space model constructor. The two
required arguments define the observation vector and a tuple of dimensions (nθ, nx) where nx
is the size of the state vector, T is the number of periods, and nθ is the size of the
parameter vector of the state space.
"""
struct StateSpaceProblem{YT<:AbstractVector,MT}
    model::MT
    observations::YT
    dimensions::NTuple{2,Int}
end

(p::StateSpaceProblem)(θ) = p.model(θ)

# just stick to ForwardDiff for now, we can play around with this later
function AbstractMCMC.LogDensityModel(p::StateSpaceProblem)
    return AbstractMCMC.LogDensityModel(
        LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), p)
    )
end

## DIRECT ITERATION ALGORITHM ##############################################################

function logjoint(
    model::StateSpaceModel, step::Integer, state, prev_state, observation; kwargs...
)
    logjoint = SSMProblems.logdensity(model.obs, step, state, observation; kwargs...)
    logjoint += SSMProblems.logdensity(model.dyn, step, prev_state, state; kwargs...)
    return logjoint
end

function LogDensityProblems.dimension(p::StateSpaceProblem)
    return p.dimensions[1] + p.dimensions[2] * (length(p.observations) + 1)
end

# offset vectors are nice, but I could reduce the allocations here
function LogDensityProblems.logdensity(p::StateSpaceProblem, θ)
    nθ, nx = p.dimensions
    params = θ[1:nθ]
    states = OffsetVector(eachcol(reshape(θ[(nθ + 1):end], nx, :)), -1)
    return LogDensityProblems.logdensity(p, params, states, p.observations)
end

# nicer, but could induce more allocations than the for loop
function LogDensityProblems.logdensity(
    p::StateSpaceProblem, params, states, observations; kwargs...
)
    return mapreduce(
        t -> logjoint(p(params), t, states[t], states[t - 1], observations[t]; kwargs...),
        +,
        eachindex(observations)
    )
end

## OLD VERSION #############################################################################

# function LogDensityProblems.logdensity(p::StateSpaceProblem, θ)
#     nθ, nx = p.dimensions
#     params, states = θ[1:nθ], θ[(nθ + 1):end]

#     x0, xs = states[1:nx], states[(nx + 1):end]
#     xs = reshape(xs, nx, :)

#     return LogDensityProblems.logdensity(
#         p, params, x0, eachcol(xs), p.observations
#     )
# end

# function LogDensityProblems.logdensity(
#     p::StateSpaceProblem, params, init_state, states, observations; kwargs...
# )
#     logevidence = logjoint(p(params), 1, states[1], init_state, observations[1]; kwargs...)
#     for t in 2:length(observations)
#         logevidence += logjoint(
#             p(params), t, states[t], states[t - 1], observations[t]; kwargs...
#         )
#     end
#     return logevidence
# end

end
