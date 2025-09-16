# DirectIteration

[![Build Status](https://github.com/charlesknipp/DirectIteration.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/charlesknipp/DirectIteration.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Marginal Likelihood Approach
Traditional likelihood evaluation of state space models is done via marginalizing the log-likelihood with a particle filter (for nonlinear models). Using Bayes theorem we evaluate the following likelihood:

```math
\begin{align*}
\log p \left( x^{T}, \theta | y^{T} \right) &\propto \log p \left( y^{T}, \theta \right) \\
&\propto \log p \left( \theta \right) + \log p \left( y_{T} | \theta \right) \\
&\propto \log p \left( \theta \right) + \sum \log p \left( y_{t} | y_{t-1}, \theta \right)
\end{align*}
```

where filter yields an expression for $\log p \left( y_{t} | y_{t-1}, \theta \right)$ at every time step of the algorithm.

In Julia, we can use `GeneralizedFilters` to return the log marginal likelihood for a user provided state space ala `SSMProblems`. Given the user provided model `state_space` we can compute the posterior likelihood as follows:

```julia
function logpost(θ, data, ::Marginalization)
	model = state_space(θ)
	_, logmarginal = GeneralisedFilters.filter(model, BF(1024), data)
	return logmarginal
end
```

## Direct Iteration Approach
The algorithm proposed in (Childers, 2022) uses a different approach to evaluating likelihoods like so:

```math
\begin{align*}
	\log p \left( \theta, x^{T} | y^{T} \right) &\propto \log p \left( x^{T}, \theta \right) + \log p \left( y^{T} | x^{T}, \theta \right) \\
	 &\propto \log p \left( \theta \right) + \log p \left( x^{T} | \theta \right) + \sum \log p \left( y_{t} | x_{t}, \theta \right) \\
	 &\propto \log p \left( \theta \right) + \log p \left( x_{0} | \theta \right) + \sum \log p \left( x_{t} | x_{t-1}, \theta \right) + \sum \log p \left( y_{t} | x_{t}, \theta \right)
\end{align*}
```

With this approach, computing the posterior is *filter free* in the sense that predictions are made in bulk. We specifically ignore the Markovian structure of the problem in favor of potential gains from differentiable sampling.

Using `Turing.jl` we can very easily define this problem in a self contained model.

```julia
@model function direct_iteration(θ, data)
    model = state_space(θ)
    x[0] ~ SSMProblems.distribution(model.prior)
    for t in eachindex(data)
        x[t] ~ SSMProblems.distribution(model.dyn, t, x[t-1])
        data[t] ~ SSMProblems.distribution(model.obs, t, x[t])
    end
end
```

Unfortunately, this simple implementation leaves a lot to be desired in terms of performance and flexibility. For starters, the overhead required to initialize a set of `VarInfo`s for both the states and parameters is quite costly. Furthermore, support for static arrays, CUDA, and single precision floats are not yet available in a fully featured Turing model. If we want to optimize to the fullest extent, the best course is to rewrite this likelihood evaluation from scratch.

Luckily, we can still use the APIs and samplers provided in the Turing ecosystem (`AbstractMCMC`, `MCMCChains`, and `AdvancedHMC` to name a few). Instead of using `DynamicPPL` to define the likelihood evaluation, we can try the following:

```julia
function logpost(θ, x, data, ::DirectIteration)
	logjoint = 0
	for t in eachindex(data)
		logjoint += SSMProblems.logdensity(model.dyn, t, x[t-1], x[t])
		logjoint += SSMProblems.logdensity(model.obs, t, x[t], data[t])
	end
	return logjoint
end
```

Where `x` is determined by the MCMC algorithm, and transformed by a Bijector if necessary.