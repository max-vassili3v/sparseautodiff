include("sparseautodiff.jl")
include("gradientdescent.jl")
using .SparseAutoDiff, LinearAlgebra, Statistics, Plots

elemwise_relu(𝐛,p) = relu.(𝐛 .+ p)

#Entire process of obtaining ∇f(x) for gradient descent, as well as the loss.
function differentiate_model(𝐛,x)
    l = length(𝐛)
    #relu pullback
    y, t = pullback_diagonal(elemwise_relu, 𝐛; 𝐩 = x)
    #summation pullback
    y2, t2 = sum(y), ones(l)
    #loss pullback
    y3, t3 = (y2 - tan(x))^2, 2*(y2-tan(x))

    t(t2*t3), y3
end

#Simple gradient descent, 'batching' over the interval [0,1]
function gradient_descent(n,N,γ)
    𝐱 = rand(n)
    for i=1:N
        δ = []
        loss = []
        for j in LinRange(0,1,10)
            #Find losses and gradients at individual points
            d,l = differentiate_model(𝐱,j)
            push!(δ,d)
            push!(loss,l)
        end
        #Compute and apply means.
        δ_final, loss_final = mean(δ), mean(loss)
        𝐱 -= γ*δ_final
        println("iteration $i loss $loss_final")
    end
    𝐱
end

𝐛 = gradient_descent(4,200,0.1)
x = LinRange(0,1,10)
y = [sum(elemwise_relu(𝐛,j)) for j in x]

plot(x,tan.(x))
plot!(x,y)