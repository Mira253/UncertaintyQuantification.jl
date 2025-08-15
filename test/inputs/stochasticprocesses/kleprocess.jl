
@testset "Karhunen-Loève Expansion (stationary)" begin

    # 1. define time steps and covariance function
    t = collect(0:0.1:10)  # time vector
    cov_stat(ti, tj) = exp(-abs(ti - tj))  # exponential grading covariance function

    # 2. KLE-Process
    kle_stat = KLEProcess(cov_stat, t, :klestatprocess, 20)  # 20 eigenfunctions

    # 3. generate random values (ξ-values)
    Random.seed!(1234)
    df = sample(kle_stat, 1) # Dataframe
    ξ = collect(df[1, names(kle_stat)])

    # 4. Evaluating
    x1 = evaluate(kle_stat, ξ)
    x2 = kle_stat(ξ)

    # 5. Test
    @test length(x1) == length(t)
    @test x1 ≈ x2

end

@testset "Karhunen-Loève Expansion (non-stationary)" begin

    # 1. define time steps and non-stationary covariance function
    t = collect(0:0.1:10)  # time vector
    cov_nstat(ti, tj) = exp(-abs(ti - tj)) * sqrt(ti * tj + 1)

    # 2. KLE-Process with non-stationary covariance
    kle_nstat = KLEProcess(cov_nstat, t, :kle_nonstat_process, 20)

    # 3. generate random ξ-values
    Random.seed!(5678)
    df = sample(kle_nstat, 1)
    ξ = collect(df[1, names(kle_nstat)])

    # 4. Evaluate
    x1 = evaluate(kle_nstat, ξ)
    x2 = kle_nstat(ξ)

    # 5. Tests
    @test length(x1) == length(t)
    @test x1 ≈ x2

end