export AbstractOperator, AbstractSource, AbstractEquation   
export Operator, Source, Src
export Time, Laplacian, Divergence, Si
export Model, ScalarEquation, VectorEquation, ModelEquation, ScalarModel, VectorModel
export nzval_index
export spindex

# ABSTRACT TYPES 

abstract type AbstractSource end
abstract type AbstractOperator end
abstract type AbstractEquation end

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
struct Src{F,S} <: AbstractSource
    field::F 
    sign::S 
    # type::T
end
Adapt.@adapt_structure Src

# Source types

struct Source end
Adapt.@adapt_structure Source
Source(f::T) where T = Src(f, 1)

# MODEL TYPE
struct Model{TN,SN,T,S}
    terms::T
    sources::S
end
# Adapt.@adapt_structure Model
function Adapt.adapt_structure(to, itp::Model{TN,SN,TT,SS}) where {TN,SN,TT,SS}
    terms = Adapt.adapt(to, itp.terms); T = typeof(terms)
    sources = Adapt.adapt(to, itp.sources); S = typeof(sources)
    Model{TN,SN,T,S}(terms, sources)
end
Model{TN,SN}(terms::T, sources::S) where {TN,SN,T,S} = begin
    Model{TN,SN,T,S}(terms, sources)
end

# Linear system matrix equation

## ORIGINAL STRUCTURE PARAMETERISED FOR GPU
struct ScalarEquation{VTf<:AbstractVector, ASA<:AbstractSparseArray} <: AbstractEquation
    A::ASA
    b::VTf
    R::VTf
    Fx::VTf
end
Adapt.@adapt_structure ScalarEquation
ScalarEquation(mesh::AbstractMesh, periodicConnectivity) = begin
    nCells = length(mesh.cells)
    Tf = _get_float(mesh)
    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 
    i, j, v = sparse_matrix_connectivity(mesh_temp) # This needs to be a kernel
    i = [i; periodicConnectivity.i]
    j = [j; periodicConnectivity.j]
    v = zeros(Tf, length(j))
    # i, j, v = sparse_matrix_connectivity(mesh_temp) # This needs to be a kernel
    # # i = periodicConnectivity.i
    # # j = periodicConnectivity.j
    # v = zeros(Tf, length(j))
    backend = _get_backend(mesh)
    ScalarEquation(
        _convert_array!(sparse(i, j, v), backend),
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend)
        )
end

struct VectorEquation{VTf<:AbstractVector, ASA<:AbstractSparseArray} <: AbstractEquation
    A0::ASA
    A::ASA
    bx::VTf
    by::VTf
    bz::VTf
    R::VTf
    Fx::VTf
end
Adapt.@adapt_structure VectorEquation
VectorEquation(mesh::AbstractMesh, periodicConnectivity) = begin
    nCells = length(mesh.cells)
    Tf = _get_float(mesh)
    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 
    i, j, v = sparse_matrix_connectivity(mesh_temp) # This needs to be a kernel
    i = [i; periodicConnectivity.i]
    j = [j; periodicConnectivity.j]
    v = zeros(Tf, length(j))
    backend = _get_backend(mesh)
    VectorEquation(
        _convert_array!(sparse(i, j, v), backend) ,
        _convert_array!(sparse(i, j, v), backend) ,
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend)
        )
end

# Sparse matrix connectivity function definition
function sparse_matrix_connectivity(mesh::AbstractMesh)
    (; cells, cell_neighbours) = mesh
    nCells = length(cells)
    TI = _get_int(mesh)
    TF = _get_float(mesh)
    nindex  = index_array_size(mesh)
    i = zeros(TI, nindex)
    j = zeros(TI, nindex)
    index = zero(TI)
    for cID = 1:nCells
        index  += 1
        cell = cells[cID]
        i[index] = cID # diagonal row index
        j[index] = cID # diagonal column index
        for fi ∈ cell.faces_range
            index  += 1
            neighbour = cell_neighbours[fi]
            i[index] = cID # cell index (row)
            j[index] = neighbour # neighbour index (column)
        end
    end
    v = zeros(TF, index)
    return i, j, v
end

function index_array_size(mesh)
    (; cells, cell_neighbours) = mesh
    nCells = length(cells)
    nindex = 0
    for cID = 1:nCells   
        cell = cells[cID]
        nindex += 1
        for fi ∈ cell.faces_range
            nindex += 1
        end
    end
    nindex
end

# Nzval index function definition for sparse array
function nzval_index(colptr, rowval, start_index, required_index, ione)
    # Set start value and offset to 0
    start = colptr[start_index]
    offset = 0
    
    # Loop over rowval array and increment offset until required value
    for j in start:length(rowval)
        offset += 1
        if rowval[j] == required_index
            break
        end
    end

    # Calculate index to output
    return start + offset - ione
end

function spindex(colptr::AbstractArray{T}, rowval, i, j) where T

    # rows = rowvals(A)
    start_ind = colptr[j]
    end_ind = colptr[j+1]

    # ind = zero(eltype(colptr))
    ind = zero(T)
    for nzi in start_ind:end_ind
    # for nzi in nzrange(A, j)
        if rowval[nzi] == i
            ind = nzi
            break
            # return ind
        end
    end
    return ind
end

# Model equation type 
struct ScalarModel end
Adapt.@adapt_structure ScalarModel

struct VectorModel end
Adapt.@adapt_structure VectorModel

struct ModelEquation{T,M,E,S,P}
    type::T
    model::M 
    equation::E 
    solver::S
    preconditioner::P
end
Adapt.@adapt_structure ModelEquation