using HomotopyContinuation
import MultivariatePolynomials
const MP = MultivariatePolynomials

struct Intersection{S<:Systems.AbstractSystem} <: Systems.AbstractSystem
    F::S
    degrees::Vector{Int}
    A::Matrix{Complex{Float64}}
    b::Vector{Complex{Float64}}
end

function Intersection(F::Vector{<:MP.AbstractPolynomial};
    system=Systems.FPSystem)

    @assert length(F) > MP.nvariables(F)

    A=rand(Complex{Float64}, MP.nvariables(F), length(F) - MP.nvariables(F))
    b=rand(Complex{Float64}, length(F))

    homvar = Utilities.uniquevar(F)
    F_hom = Utilities.homogenize(F, homvar)
    # Reorder system such that the polynomial with highest degree are the first ones.
    sort!(F_hom, by=MP.maxdegree, rev=true)
    degrees = MP.maxdegree.(F_hom)

    Intersection(system(F_hom), degrees, A, b)
end

Intersection(F1)

struct MapIntersectionCache{SC<:Systems.AbstractSystemCache, T} <: Systems.AbstractSystemCache
    cache::SC
    eval_buffer::Vector{T}
    jac_buffer::Matrix{T}
end

function Systems.cache(MI::MapIntersection, x)
    c = Systems.cache(MI.F, x)
    eval_buffer = Systems.evaluate(MI.F, x, c)
    jac_buffer = Systems.jacobian(MI.F, x, c)
    MapIntersectionCache(c, eval_buffer, jac_buffer)
end

Base.size(MI::MapIntersection) = (size(MI.A, 1), size(MI.F)[2])

function Systems.evaluate!(u, MI::MapIntersection, x, c::MapIntersectionCache)
    Systems.evaluate!(c.eval_buffer, MI.F, x, c.cache)
    A_mul_B!(u, MI.A, c.eval_buffer)
    u
end
function Systems.evaluate(MI::MapIntersection, x, c::MapIntersectionCache)
    u = similar(c.eval_buffer, size(MI)[1])
    Systems.evaluate!(u, MI, x, c)
    u
end

function Systems.jacobian!(u, MI::MapIntersection, x, c::MapIntersectionCache)
    Systems.jacobian!(c.jac_buffer, MI.F, x, c.cache)
    A_mul_B!(u, MI.A, c.jac_buffer)
    u
end

function Systems.jacobian(MI::MapIntersection, x, c::MapIntersectionCache)
    u = similar(c.jac_buffer, size(MI))
    Systems.jacobian!(u, MI, x, c)
    u
end

function Systems.evaluate_and_jacobian!(u, U, MI::MapIntersection, x, c::MapIntersectionCache)
    Systems.evaluate_and_jacobian!(c.eval_buffer, c.jac_buffer, MI.F, x, c.cache)
    A_mul_B!(u, MI.A, c.eval_buffer)
    A_mul_B!(U, MI.A, c.jac_buffer)
    nothing
end
function Systems.evaluate_and_jacobian(MI::MapIntersection, x, c::MapIntersectionCache)
    u = similar(c.eval_buffer, size(MI)[1])
    U = similar(c.jac_buffer, size(MI))
    Systems.evaluate_and_jacobian!(u, U, MI, x, c)
    u, U
end
