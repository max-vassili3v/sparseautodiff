include("SparseAutoDiff.jl")
using .SparseAutoDiff

elemwise_relu(𝐛,x) = relu.(𝐛 .+ x)

function model_loss(𝐛,x)
    (sum(elemwise_relu(𝐛,x))-tan(x))^2
end

function differentiate_model(𝐛)
    
end