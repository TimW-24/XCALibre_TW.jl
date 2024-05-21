export get_phi, get_flux
export get_source, get_source_sign
export _A, _A0, _b
export _nzval, _rowval, _colptr
export get_sparse_fields
export XDir, YDir, ZDir, get_values

# Components
struct XDir{T} 
    value::T
end
Adapt.@adapt_structure XDir
XDir() = XDir(1)

struct YDir{T}  
    value::T
end
Adapt.@adapt_structure YDir
YDir() = YDir(2)

struct ZDir{T}
    value::T
end
Adapt.@adapt_structure ZDir
ZDir() = ZDir(3)

## MODEL ACCESSORS

@inline get_phi(eqn::ModelEquation)  = begin 
    eqn.model.terms[1].phi
end

@inline get_flux(eqn::ModelEquation, ti::Integer) = begin 
    eqn.model.terms[ti].flux
end

@inline get_source(eqn::ModelEquation, ti::Integer) = begin 
    eqn.model.sources[ti].field
end

@inline get_source_sign(eqn::ModelEquation, ti::Integer) = begin 
    eqn.model.sources[ti].sign
end

## SPARSE MATRIX ACCESSORS

# Access Scalar Model Equation
@inline _A(eqn::ModelEquation{T,M,E,S,P}) where {T<:ScalarModel,M,E,S,P} = begin
    eqn.equation.A
end
@inline _b(eqn::ModelEquation{T,M,E,S,P}) where {T<:ScalarModel,M,E,S,P} = begin
    eqn.equation.b
end
@inline _b(eqn::ModelEquation{T,M,E,S,P},c::Nothing) where {T<:ScalarModel,M,E,S,P} = begin
    eqn.equation.b
end

# Access Vector Model Equation
@inline _A0(eqn::ModelEquation{T,M,E,S,P}) where {T<:VectorModel,M,E,S,P} = begin
    eqn.equation.A0
end
@inline _A(eqn::ModelEquation{T,M,E,S,P}) where {T<:VectorModel,M,E,S,P} = begin
    eqn.equation.A
end
@inline _b(eqn::ModelEquation{T,M,E,S,P}, c::XDir) where {T<:VectorModel,M,E,S,P} = begin
    eqn.equation.bx
end
@inline _b(eqn::ModelEquation{T,M,E,S,P}, c::YDir) where {T<:VectorModel,M,E,S,P} = begin
    eqn.equation.by
end
@inline _b(eqn::ModelEquation{T,M,E,S,P}, c::ZDir) where {T<:VectorModel,M,E,S,P} = begin
    eqn.equation.bz
end

@inline _nzval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.nzVal
@inline _nzval(A::SparseArrays.SparseMatrixCSC) = A.nzval

@inline _colptr(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.colPtr
@inline _colptr(A::SparseArrays.SparseMatrixCSC) = A.colptr

@inline _rowval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.rowVal
@inline _rowval(A::SparseArrays.SparseMatrixCSC) = A.rowval

@inline get_sparse_fields(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = begin
    A.nzVal, A.rowVal, A.colPtr
end

@inline get_sparse_fields(A::SparseArrays.SparseMatrixCSC) = begin
    A.nzval, A.rowval, A.colptr
end

@inline get_values(phi::ScalarField, component::Nothing) = phi.values
@inline get_values(psi::VectorField, component::XDir) = psi.x.values
@inline get_values(psi::VectorField, component::YDir) = psi.y.values
@inline get_values(psi::VectorField, component::ZDir) = psi.z.values