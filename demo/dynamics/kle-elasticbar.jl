using UncertaintyQuantification
using Statistics
using LinearAlgebra


# Karhunen–Loève Expansion (KLE) for a 1D Random Elastic Modulus Field

# Reference:
# Kazemi & Steeves (2022) "Uncertainty Quantification in Material Properties 
# of Additively Manufactured Materials for Application in Topology Optimization"

# Description:
# A stationary random field E(x) (Elastic modulus) is represented using
# the Karhunen–Loève Expansion (KLE) with an exponential covariance kernel:
#
#     C(x₁, x₂) = σ² * exp(-|x₁ - x₂| / ℓ)
#
# The resulting field represents spatial material uncertainty in a 1D steel bar.
# The KLE is used to reconstruct the 1D random field and compute representative
# engineering statistics (mean, standard variation, explained variance).

# material parameters
L = 1.0                            # bar length [m]
N = 100                            # number of discrete grid points on the bar
x = collect(range(0, L; length=N))

mean_E = 210                       # mean elastic modulus [GPa] (steel)
σ² = 0.0025                        # variance of normalized field
l = 0.2                            # correlation length [m]

# Exponential covariance function (stationary)
cov_E(x1, x2) = σ² * exp(-abs(x1 - x2) / l)  # equation (3.3) & (3.4) in Kazemi & Steeves (2022)

M = 20                             # number of KLE terms
kle = KLEProcess(cov_E, x, :Efield, M)

# random realizations
n_samples = 500
df = sample(kle, n_samples)
ξ = collect(df[1, names(kle)])
E_fluct = evaluate(kle, ξ)
E_realization = mean_E .* (1 .+ E_fluct)

# Basic field statistics
E_mean = mean(E_realization)
E_std  = std(E_realization)
E_min  = minimum(E_realization)
E_max  = maximum(E_realization)

println("------------------------------------------------------------")
println("Karhunen-Loève Expansion: 1D Random Elastic Modulus Field")
println("------------------------------------------------------------")
println("Bar length L = $(L) m")
println("Correlation length b = $(b) m")
println("Number of KLE terms M = $(M)")
println()
println("Mean modulus      E_mean = $(round(E_mean, digits=3)) GPa")
println("Std. deviation    E_std  = $(round(E_std, digits=4)) GPa")
println("Min / Max         = $(round(E_min, digits=3)) / $(round(E_max, digits=3)) GPa")
println("------------------------------------------------------------")
println("Interpretation:")
println("• The KLE models spatial fluctuations in the elastic modulus field.")
println("• The explained variance quantifies how much of the total field variance is captured by M terms.")
println("• The effective modulus represents the equivalent homogeneous stiffness of the random bar.")
println("------------------------------------------------------------")
