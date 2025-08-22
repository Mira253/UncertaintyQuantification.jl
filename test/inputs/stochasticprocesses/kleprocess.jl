
@testset "Karhunen-Loève Expansion (stationary)" begin

    # 1. define time steps and covariance function
    t = collect(0:0.1:10)
    cov_stat(ti, tj) = exp(-abs(ti - tj))  # exponential grading covariance function

    # 2. KLE-Process
    kle_stat = KLEProcess(cov_stat, t, :klestatprocess, 20)  # 20 eigenfunctions

    # 3. generate random values (ξ-values)
    Random.seed!(1234)
    n_samples = 500
    df = sample(kle_stat, n_samples)

    # 4. Evaluating
    ξ = collect(df[1, names(kle_stat)])
        x1 = evaluate(kle_stat, ξ)
        x2 = kle_stat(ξ)


    ## making the empirical covariance matrix 

    # 1. Evaluate the time series X(t) for each sample ξ
    X = [evaluate(kle_stat, collect(df[i, names(kle_stat)])) for i in 1:n_samples]

    # 2. Construct a matrix: each column = one realization X^(k)(t)
    Xmat = reduce(hcat, X)  # size: (length(t), n_samples)
    

    # 3. empirical matrix
    K_empirical = cov(Matrix(Xmat'))

    K_kle = (kle_stat.eigfuncs .* kle_stat.eigvals') * kle_stat.eigfuncs'

    ## Test
    @test length(x1) == length(t)
    @test x1 ≈ x2
    @test K_empirical ≈ K_kle atol=0.1

end

@testset "Karhunen-Loève Expansion (non-stationary)" begin

    # 1. define time steps and non-stationary covariance function
    t = collect(0:0.1:10)  # time vector
    cov_nstat(ti, tj) = exp(-abs(ti - tj)) * sqrt(ti * tj + 1)

    # 2. KLE-Process
    kle_nstat = KLEProcess(cov_nstat, t, :klestatprocess, 50)  # 20 eigenfunctions

    # 3. generate random values (ξ-values)
    Random.seed!(1234)
    n_samples = 500
    df = sample(kle_nstat, n_samples)

    # 4. Evaluating
    ξ = collect(df[1, names(kle_nstat)])
        x1 = evaluate(kle_nstat, ξ)
        x2 = kle_nstat(ξ)


    ## making the empirical covariance matrix 

    # 1. Evaluate the time series X(t) for each sample ξ
    X = [evaluate(kle_nstat, collect(df[i, names(kle_nstat)])) for i in 1:n_samples]

    # 2. Construct a matrix: each column = one realization X^(k)(t)
    Xmat = reduce(hcat, X)  # size: (length(t), n_samples)
    

    # 3. empirical matrix
    K_empirical = cov(Matrix(Xmat'))
    
    K_kle = (kle_nstat.eigfuncs .* kle_nstat.eigvals') * kle_nstat.eigfuncs'

    ## Test
    @test length(x1) == length(t)
    @test x1 ≈ x2
    @test K_empirical ≈ K_kle atol=0.1

end