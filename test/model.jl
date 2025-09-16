function factor_matrix(λs::AbstractVector{T}, ny::Int, nx::Int) where {T}
    Λ = diagm(ny, nx, ones(T, min(nx, ny)))
    iter = 1
    for i in 1:ny, j in 1:nx
        if i > j
            Λ[i, j] = λs[iter]
            iter += 1
        end
    end
    return Λ
end

num_factors(ny::Int, nx::Int) = ny * nx - sum(1:nx);

function factor_model(σ::ΣT, λs::Vector{ΛT}, dims::NTuple{2,Int}) where {ΣT, ΛT}
    T = Base.promote_type(ΣT, ΛT)
    ny, nx = dims

    # transition process is a dampened iid random walk
    A = T(0.85)I(nx)

    # add noise to identify mixed signals
    Q = T(0.4)I(nx)

    # factor loading normalized on the diagonals
    Λ = factor_matrix(λs, ny, nx)
    Σ = Diagonal(σ * ones(ΣT, ny))

    # return the homogeneous linear Gaussian state space model
    return SSMProblems.StateSpaceModel(
        GF.HomogeneousGaussianPrior(zeros(T, nx), lyapd(A, Q)),
        GF.HomogeneousLinearGaussianLatentDynamics(A, zeros(T, nx), Q),
        GF.HomogeneousLinearGaussianObservationProcess(Λ, zeros(T, ny), Σ)
    )
end

## PRIORS ##################################################################################

# should be type stable even under the main branch of Distributions
λ_prior(::Type{T}, ny::Int, nx::Int) where {T<:Real} = MvNormal(T(0.1)I(num_factors(ny, nx)));
λ_prior(ny::Int, nx::Int) = λ_prior(Float64, ny, nx);

# only use with Distributions#dw/rand_multiple_consistent
σ_prior(::Type{T}) where {T} = Beta(T(1), T(1));
σ_prior() = σ_prior(Float64);