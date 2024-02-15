export AbstractOperator, AbstractSource   
export Operator, Source, Src
export Time, Laplacian, Divergence, Si
export Model, Equation, ModelEquation

# ABSTRACT TYPES 

abstract type AbstractSource end
abstract type AbstractOperator end

# OPERATORS

# Base Operator

struct Operator{F,P,S,T} <: AbstractOperator
    flux::F
    phi::P 
    sign::S
    type::T
end
Adapt.@adapt_structure Operator
# operators

struct Time{T} end
function Adapt.adapt_structure(to, itp::Time{T}) where {T}
    Time{T}()
end

struct Laplacian{T} end
function Adapt.adapt_structure(to, itp::Laplacian{T}) where {T}
    Laplacian{T}()
end

struct Divergence{T} end
function Adapt.adapt_structure(to, itp::Divergence{T}) where {T}
    Divergence{T}()
end

struct Si end
function Adapt.adapt_structure(to, itp::Si)
    Si()
end

# constructors

Time{T}(flux, phi) where T = Operator(
    flux, phi, 1, Time{T}()
    )

Time{T}(phi) where T = Operator(
    ConstantScalar(one(_get_int(phi.mesh))), phi, 1, Time{T}()
    )

Laplacian{T}(flux, phi) where T = Operator(
    flux, phi, 1, Laplacian{T}()
    )

Divergence{T}(flux, phi) where T = Operator(
    flux, phi, 1, Divergence{T}()
    )

Si(flux, phi) = Operator(
    flux, phi, 1, Si()
)

# SOURCES

# Base Source
struct Src{F,S,T} <: AbstractSource
    field::F 
    sign::S 
    type::T
end
Adapt.@adapt_structure Src
# Source types

struct Source end
Adapt.@adapt_structure Source
Source(f::T) where T = Src(f, 1, typeof(f))
# Source(f::ScalarField) = Src(f.values, 1, typeof(f))
# Source(f::Number) = Src(f.values, 1, typeof(f)) # To implement!!

# MODEL TYPE
struct Model{T,S,TN,SN}
    # equation::E
    terms::T
    sources::S
end
function Adapt.adapt_structure(to, itp::Model{TN,SN}) where {TN,SN}
    terms = Adapt.adapt_structure(to, itp.terms); T = typeof(terms)
    sources = Adapt.adapt_structure(to, itp.sources); S = typeof(sources)
    Model{T,S,TN,SN}(terms,sources)
end
Model{TN,SN}(terms::T, sources::S) where {T,S,TN,SN} = begin
    Model{T,S,TN,SN}(terms, sources)
end
# Model(eqn::E, terms::T, sources::S, TN, SN) where {E,T,S} = begin
#     Model{E,T,S,TN,SN}(eqn, terms, sources)
# end

# Linear system matrix equation

struct Equation{SMCSC,VTf}
    A::SMCSC
    b::VTf
    R::VTf
    Fx::VTf
    # mesh::Mesh2{Ti,Tf}
end
Adapt.@adapt_structure Equation
Equation(mesh::Mesh2) = begin
    nCells = length(mesh.cells)
    Tf = _get_float(mesh)
    i, j, v = sparse_matrix_connectivity(mesh)
    A = sparse(i, j, v); SMCSC = typeof(A)
    b = zeros(Tf, nCells); VTf = typeof(b)
    R = zeros(Tf, nCells)  
    Fx = zeros(Tf, nCells)
    Equation{SMCSC,VTf}(
        A,
        b,
        R,
        Fx
        # mesh
        )
end

function sparse_matrix_connectivity(mesh::Mesh2)
    (; cells, cell_neighbours) = mesh
    nCells = length(cells)
    TI = _get_int(mesh) # would this result in regression (type identified inside func?)
    TF = _get_float(mesh) # would this result in regression (type identified inside func?)
    i = TI[]
    j = TI[]
    for cID = 1:nCells   
        cell = cells[cID]
        push!(i, cID) # diagonal row index
        push!(j, cID) # diagonal column index
        # for fi ∈ eachindex(cell.facesID)
        for fi ∈ cell.faces_range
            # neighbour = cell.neighbours[fi]
            neighbour = cell_neighbours[fi]
            push!(i, cID) # cell index (row)
            push!(j, neighbour) # neighbour index (column)
        end
    end
    v = zeros(TF, length(i))
    return i, j, v
end

# Model equation type 

struct ModelEquation{M,E,S,P}
    model::M 
    equation::E 
    solver::S
    preconditioner::P
end
Adapt.@adapt_structure ModelEquation