
@testset "Karhunen-Loève Expansion" begin

    # 1. define time steps and covariance function
    t = collect(0:0.1:10)  # time vector
    cov(ti, tj) = exp(-abs(ti - tj))  # exponential grading covariance function

    # 2. KLE-Process
    kle = KLEProcess(cov, t, :kleprocess, 20)  # 20 eigenfunctions

    # 3. generate random values (ξ-values)
    Random.seed!(1234)
    df = sample(kle, 1) # Dataframe
    ξ = collect(df[1, names(kle)])

    # 4. Evaluating
    x1 = evaluate(kle, ξ)
    x2 = kle(ξ)

    # 5. Test
    @test length(x1) == length(t)
    @test x1 ≈ x2

end
