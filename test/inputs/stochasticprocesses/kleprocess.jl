
@testset "Karhunen-Loève Expansion (stationary)" begin
    # --- Setup ---
    # define time steps and covariance function
    t = collect(0:0.1:10)
    cov_stat(ti, tj) = exp(-abs(ti - tj))  # exponential covariance kernel

    # build KLE process with M terms
    M = 30
    kle_stat = KLEProcess(cov_stat, t, :klestatprocess, M)

    # sample realizations
    n_samples = 500
    df = sample(kle_stat, n_samples)

    # --- Test 1: Evaluate consistency ---
    ξ = collect(df[1, names(kle_stat)])
    x1 = evaluate(kle_stat, ξ)
    x2 = kle_stat(ξ)

    @test length(x1) == length(t)   # check correct output dimension
    @test x1 ≈ x2                   # consistency of evaluate vs. call

    # --- Test 2: Covariance reproduction ---
    # Empirical covariance from samples
    X = [evaluate(kle_stat, collect(df[i, names(kle_stat)])) for i in 1:n_samples]
    Xmat = reduce(hcat, X)  # (length(t), n_samples)
    K_empirical = cov(Matrix(Xmat'))

    # Theoretical covariance matrix
    K_kle = kle_stat.eigfuncs * Diagonal(kle_stat.eigvals) * kle_stat.eigfuncs'

    # Difference between theoretical and empirical covariance
    diff = abs.(K_empirical - K_kle)
    max_diff = maximum(diff)

    @test max_diff ≤ 0.4

    # --- Test 3: Energy ratio ---
    M_full = length(t)  # maximum number of eigenvalues
    kle_full = KLEProcess(cov_stat, t, :klefull, M_full)

    total_variance = sum(kle_full.eigvals)          # sum of all eigenvalues of full KLE
    captured_variance = sum(kle_stat.eigvals)       # sum of used eigenvalues in original KLE

    energy_ratio = captured_variance / total_variance

    @test energy_ratio ≥ 0.9  # at least 90% of variance captured
end


@testset "Karhunen-Loève Expansion (non-stationary)" begin
    # --- Setup ---
    # define time steps and non-stationary covariance function
    t = collect(0:0.1:10)
    cov_n_stat(ti, tj) = exp(-abs(ti - tj)) * sqrt(ti * tj + 1)  # non-stationary covariance kernel

    # build KLE process with M terms
    M = 30
    kle_n_stat = KLEProcess(cov_n_stat, t, :klenstatprocess, M)

    # sample realizations
    n_samples = 500
    df = sample(kle_n_stat, n_samples)

    # --- Test 1: Evaluate consistency ---
    ξ = collect(df[1, names(kle_n_stat)])
    x1 = evaluate(kle_n_stat, ξ)
    x2 = kle_n_stat(ξ)

    @test length(x1) == length(t)   # check correct output dimension
    @test x1 ≈ x2                   # consistency of evaluate vs. call

    # --- Test 2: Covariance reproduction ---
    # Empirical covariance from samples
    X = [evaluate(kle_n_stat, collect(df[i, names(kle_n_stat)])) for i in 1:n_samples]
    Xmat = reduce(hcat, X)  # (length(t), n_samples)
    K_n_empirical = cov(Matrix(Xmat'))

    # Theoretical covariance matrix
    K_n_kle = kle_n_stat.eigfuncs * Diagonal(kle_n_stat.eigvals) * kle_n_stat.eigfuncs'

    # Difference between theoretical and empirical covariance
    diff = abs.(K_n_empirical - K_n_kle)
    max_diff = maximum(diff)

    @test max_diff ≤ 4

    # --- Test 3: Energy ratio ---
    M_full = length(t)  # maximum number of eigenvalues
    kle_n_full = KLEProcess(cov_n_stat, t, :klenfull, M_full)

    total_variance = sum(kle_n_full.eigvals)          # sum of all eigenvalues of full KLE
    captured_variance = sum(kle_n_stat.eigvals)       # sum of used eigenvalues in original KLE

    energy_ratio = captured_variance / total_variance

    @test energy_ratio ≥ 0.9  # at least 90% of variance captured
end
