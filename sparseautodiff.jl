#Utility module for the structures we need
module SparseAutoDiff

import Base: ^, *, +, -, sin, size

export pullback_diagonal, relu

#Implementation of Dual numbers
struct Dual
    a::Number
    b::Number
end

+(x::Dual,y::Dual) = Dual(x.a+y.a,x.b+y.b)
+(x::Dual,y::Number) = Dual(x.a+y,x.b)
+(x::Number,y::Dual) = Dual(y.a+x,y.b)
-(x::Dual,y::Number) = Dual(x.a-y,x.b)
-(x::Dual,y::Dual) = Dual(x.a-y.a,x.b-y.b)
-(x::Number,y::Dual) = Dual(x-y.a,-y.b)
^(x::Dual,y::Number) = Dual(x.a^y,x.b*y*x.a^(y-1))
*(x::Dual,y::Dual) = Dual(x.a*y.a, x.a*y.b+x.b*y.a)
*(x::Number,y::Dual) = Dual(x*y.a,x*y.b)
*(x::Dual,y::Number) = Dual(x.a*y,x.b*y)
sin(x::Dual) = Dual(sin(x.a),x.b*cos(x.a))
relu(x::Real) = max(0,x)
relu(x::Dual) = Dual(x.a,x.b * (x.a >= 0 ? 1 : 0))

#Implementation of Diagonal Matrix
struct DiagonalMatrix{T<:Number}
    x::AbstractVector{T}
end

function size(a::DiagonalMatrix)
    length(a.x)
end
*(x::DiagonalMatrix,y::AbstractVector) = x.x .* y
function (*)(x::DiagonalMatrix,y::AbstractMatrix)
    l = size(x)
    result = zeros(Number,l,size(y)[2])
    for i=1:l
        result[i,:] = x.x[i]*y[i,:]
    end
    result
end

#Get pullback for diagonal matrix
pullback_diagonal = function(f, 𝐱; 𝐩 = nothing)
    l = length(𝐱)
    result = Vector{Number}(zeros(eltype(𝐱),l))
    for i=1:l
        𝐲 = Dual.(𝐱,0)
        𝐲[i] = Dual(𝐲[i].a,1)
        result[i] = ( 𝐩 === nothing ? f(𝐲)[i].b : f(𝐲,𝐩)[i].b)
    end
    x = 𝐩 === nothing ? f(𝐱) : f(𝐱,𝐩)
    p(t) = DiagonalMatrix(result)*t
    return x, p
end

println("Successfully included")

end
