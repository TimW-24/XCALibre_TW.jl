export RANS
export Laminar
export AbstractMomentumModel, AbstractTurbulenceModel

abstract type AbstractMomentumModel end
abstract type AbstractTurbulenceModel end

# Models 
struct Laminar <: AbstractTurbulenceModel end 
Adapt.@adapt_structure Laminar

struct RANS{T} end
Adapt.Adapt.@adapt_structure RANS 

RANS{Laminar}(mesh) = Laminar()

function initialise(
    turbulence::Laminar, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}
    return model
end

function turbulence!(model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, config
    ) where {T,F,M,Tu<:Laminar,E,D,BI}
    nothing
end

# struct RANS{M,F1,F2,V,T,E,D,BI}
#     model::M
#     U::F1 
#     p::F2
#     nu::V
#     turbulence::T
#     energy::E
#     mesh::D
#     boundary_info::BI
# end 
# Adapt.@adapt_structure RANS
# RANS{Laminar}(; mesh, viscosity) = begin
#     U = VectorField(mesh); F1 = typeof(U)
#     p = ScalarField(mesh); F2 = typeof(p)
#     V = typeof(viscosity)
#     flag = false; F = typeof(flag)
#     D = typeof(mesh)
#     boundary_info = @time begin boundary_map(mesh) end; BI = typeof(boundary_info)
#     RANS{Laminar,F1,F2,V,F,F,D,BI}(
#         Laminar(), U, p, viscosity, flag, flag, mesh, boundary_info
#     )
# end

# isturbulent(model) = begin
#     typeof(model).parameters[1] <: AbstractTurbulenceModel
# end