export RANS
export Laminar
export Laminar_rho
export AbstractMomentumModel, AbstractTurbulenceModel
export isturbulent

abstract type AbstractMomentumModel end
abstract type AbstractTurbulenceModel end

# Models 
struct Laminar <: AbstractMomentumModel end 
struct Laminar_rho <: AbstractMomentumModel end 

struct boundary_info{I<:Integer, S<:Symbol}
    ID::I
    Name::S
end

struct RANS{M,F1,F2,V,T,E,D,BI}
    model::M
    U::F1 
    p::F2
    nu::V
    turbulence::T
    energy::E
    mesh::D
    boundary_info::BI
end 

RANS{Laminar}(; mesh, viscosity) = begin
    U = VectorField(mesh); F1 = typeof(U)
    p = ScalarField(mesh); F2 = typeof(p)
    V = typeof(viscosity)
    flag = false; F = typeof(flag)
    D = typeof(mesh)
    boundary_info = @time begin boundary_map(mesh) end; BI = typeof(boundary_info)
    RANS{Laminar,F1,F2,V,F,F,D,BI}(
        Laminar(), U, p, viscosity, flag, flag, mesh, boundary_info
    )
end

RANS{Laminar_rho}(; mesh, viscosity) = begin
    U = VectorField(mesh); F1 = typeof(U)
    p = ScalarField(mesh); F2 = typeof(p)
    V = typeof(viscosity)
    flag = false; F = typeof(flag)
    energy = ScalarField(mesh); E = typeof(energy)
    D = typeof(mesh)
    boundary_info = @time begin boundary_map(mesh) end; BI = typeof(boundary_info)
    RANS{Laminar_rho,F1,F2,V,F,E,D,BI}(
        Laminar_rho(), U, p, viscosity, flag, energy, mesh, boundary_info
    )
end

isturbulent(model) = begin
    typeof(model).parameters[1] <: AbstractTurbulenceModel
end