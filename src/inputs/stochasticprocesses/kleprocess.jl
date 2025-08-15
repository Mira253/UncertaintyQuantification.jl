abstract type AbstractStochasticProcess <: RandomUQInput end

struct KLEProcess <: AbstractStochasticProcess
    cov::Function  # covariance function K(t, s)
    time::Vector{Float64}  # time vector t_1, ..., t_N
    eigvals::Vector{Float64}  # Eigenvaluese λ_n
    eigfuncs::Matrix{Float64}  # Eigenfunctions ξ_n(t)
    name::Symbol
    ξnames::Vector{Symbol}  # names of the random variables
end

"""
    KLEProcess(cov::Function, time::Vector, name::Symbol, num_terms::Int) -> KLEProcess

Constructs a KLE model for a given time discrete covariance operator.


# Arguments
- `cov(t, s)`: covariance function K(t, s)
- `time::Vector{Float64}`: time discretisation
- `name::Symbol`: name of the process
- `num_terms`: number of eigenfunctions (KLE-terms)
"""

function KLEProcess(cov::Function, time::Vector{Float64}, name::Symbol, num_terms::Int)
    N = length(time)
    K = [cov(ti, tj) for ti in time, tj in time]  # covariance matrix
    vals, vecs = eigen(Symmetric(K))  # creating of eigenvalues and eigenfunctions

    eigvals = reverse(vals[1:num_terms])
    eigfuncs = reverse(vecs[:, 1:num_terms], dims=2)

    return KLEProcess(
        cov,
        time,
        eigvals,
        eigfuncs,
        name,
        [Symbol("$(name)_ξ_$(i)") for i in 1:num_terms]
    )
end

"""
    sample(proc::KLEProcess, n::Integer=1) -> DataFrame

Creates random values for the KLE random coefficients (standard normally distributed).
"""
function sample(proc::KLEProcess, n::Integer=1)
    return DataFrame(proc.ξnames .=> eachcol(randn(n, length(proc.eigvals))))
end

"""
    evaluate(proc::KLEProcess, ξ::AbstractVector) -> Vector

Calculates x(t) = sum_n sqrt(λ_n) * ξ_n * ϕ_n(t)
"""
function evaluate(proc::KLEProcess, ξ::AbstractVector)
    return proc.eigfuncs * (sqrt.(proc.eigvals) .* ξ)
end

function (proc::KLEProcess)(ξ::AbstractVector)
    return evaluate(proc, ξ)
end

function dimensions(proc::KLEProcess)
    return length(proc.eigvals)
end

function names(proc::KLEProcess)
    return proc.ξnames
end