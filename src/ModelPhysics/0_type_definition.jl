export Physics
export Transient, Steady


struct Physics{T,F,M,Tu,E,D,BI}
    time::T
    fluid::F
    momentum::M 
    turbulence::Tu 
    energy::E
    domain::D
    boundary_info::BI
end 
Adapt.@adapt_structure Physics

struct Transient end
Adapt.@adapt_structure Transient

struct Steady end
Adapt.@adapt_structure Steady

struct Momentum{V,S,SS}
    U::V 
    p::S 
    sources::SS
end 
Adapt.@adapt_structure Momentum 

Momentum(mesh::AbstractMesh) = begin
    U = VectorField(mesh)
    p = ScalarField(mesh)
    Momentum(U, p, nothing)
end

Physics(; time, fluid, turbulence, energy, domain) = begin
    momentum = Momentum(domain)
    fluid = fluid(domain)
    # turbulence = typeof(turbulence)(domain)
    turbulence = turbulence(domain)
    energy = energy(domain, fluid)
    boundary_info = boundary_map(domain)
    Physics(
        time,
        fluid,
        momentum, 
        turbulence, 
        energy,
        domain, 
        boundary_info
    )
end

# TO DO: RELOCATE TWO FUNCS BELOW TO COMMON LOCATION/MODULE AND EXPORT
struct boundary_info{I<:Integer, S<:Symbol}
    ID::I
    Name::S
end
Adapt.@adapt_structure boundary_info

# Create LUT to map boudnary names to indices
function boundary_map(mesh)
    I = Integer; S = Symbol
    boundary_map = boundary_info{I,S}[]

    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 

    for (i, boundary) in enumerate(mesh_temp.boundaries)
        push!(boundary_map, boundary_info{I,S}(i, boundary.name))
    end

    return boundary_map
end