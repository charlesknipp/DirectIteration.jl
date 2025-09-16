using OffsetArrays
using Turing

@model function turing_ssm(ny, nx)
    λs ~ λ_prior(ny, nx)
    return factor_model(0.2, λs, (ny, nx))
end

@model function direct_iteration(state_space, data)
    ssm ~ to_submodel(state_space, false)
    x0 ~ SSMProblems.distribution(ssm.prior)
    x = OffsetVector(fill(x0, length(data) + 1), -1)
    for t in eachindex(data)
        x[t] ~ SSMProblems.distribution(ssm.dyn, t, x[t-1])
        data[t] ~ SSMProblems.distribution(ssm.obs, t, x[t])
    end
end

state_space = turing_ssm(2, 2)
dppl_chain = sample(rng, direct_iteration(state_space, ys), Turing.NUTS(0.8), 500)
