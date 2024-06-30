include("SparseAutoDiff.jl")

function gradient_descent(f,n,N,γ)
    𝐱 = rand(n)
    for i=1:N
        t, p = pullback_diagonal(f, 𝐱)
        𝐱 -= γ*p(𝐱)
    end
    𝐱
end

function square_offset(𝐱)
    (𝐱 .- [1, 0, 1]).^2
end

gradient_descent(square_offset,3,1000,0.1)