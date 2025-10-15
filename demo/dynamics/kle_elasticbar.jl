using UncertaintyQuantification
using Statistics
using LinearAlgebra

# ------------------------------------------------------------
# Demo: Karhunen–Loève Expansion (KLE) for a 1D Random Elastic Modulus Field
# ------------------------------------------------------------
#
# References:
# - Ghanem & Spanos (1991), "Stochastic Finite Elements: A Spectral Approach"
#
# Description:
# A stationary random field E(x) (Elastic modulus) is represented using
# the Karhunen–Loève Expansion (KLE) with an exponential covariance kernel:
#
#     C(x₁, x₂) = σ² * exp(-|x₁ - x₂| / ℓ)
#
# The resulting field represents spatial material uncertainty in a 1D steel bar.
# The KLE is used to reconstruct the random field and compute representative
# engineering statistics (mean, std, effective modulus, explained variance).
# ------------------------------------------------------------

# --- 1. Define problem domain and material parameters ---
L = 1.0                            # bar length [m]
N = 100                            # number of discrete grid points on the bar
x = collect(range(0, L; length=N))

mean_E = 210                       # mean elastic modulus [Pa] (steel)
σ² = 0.0025                        # variance of normalized field
b = 0.2                            # correlation length [m]

# Exponential covariance function (stationary)
cov_E(x1, x2) = σ² * exp(-abs(x1 - x2) / b)

# --- 2. Construct KLE process ---
M = 20                             # number of KLE terms
kle = KLEProcess(cov_E, x, :Efield, M)

# --- 3. Generate one random realization ---
n_samples = 1
df = sample(kle, n_samples)
ξ = collect(df[1, names(kle)])     # extract ξ₁,...,ξ_M from DataFrame
E_fluct = evaluate(kle, ξ)
E_realization = mean_E .* (1 .+ E_fluct)

# --- 4. Compute representative engineering quantities ---

# Effective modulus (harmonic mean → physically representative for serial stiffness)
E_eff = L / mean(1.0 ./ E_realization)

# Basic field statistics
E_mean = mean(E_realization)
E_std  = std(E_realization)
E_min  = minimum(E_realization)
E_max  = maximum(E_realization)

# Energy ratio (explained variance by truncated KLE)
kle_full = KLEProcess(cov_E, x, :Efull, length(x))
energy_ratio = sum(kle.eigvals) / sum(kle_full.eigvals)

# --- 5. Print summary results ---

println("------------------------------------------------------------")
println("Karhunen-Loève Expansion: 1D Random Elastic Modulus Field")
println("------------------------------------------------------------")
println("Bar length L = $(L) m")
println("Correlation length b = $(b) m")
println("Number of KLE terms M = $(M)")
println("Explained variance (energy ratio): $(round(energy_ratio*100, digits=2)) %")
println()
println("Mean modulus      E_mean = $(round(E_mean, digits=3)) GPa")
println("Std. deviation    E_std  = $(round(E_std, digits=4)) GPa")
println("Min / Max         = $(round(E_min, digits=3)) / $(round(E_max, digits=3)) GPa")
println("Effective modulus E_eff = $(round(E_eff, digits=3)) GPa")
println("------------------------------------------------------------")
println("Interpretation:")
println("• The KLE models spatial fluctuations in the elastic modulus field.")
println("• The explained variance quantifies how much of the total field variance is captured by M terms.")
println("• The effective modulus represents the equivalent homogeneous stiffness of the random bar.")
println("------------------------------------------------------------")
