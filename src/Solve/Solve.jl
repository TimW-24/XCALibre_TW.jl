module Solve

const WORKGROUP = 32

using Krylov
using LoopVectorization
using LinearAlgebra
using SparseArrays
using LinearOperators
using KernelAbstractions
using Atomix
# using ILUZero

using FVM_1D.Mesh
using FVM_1D.Fields
using FVM_1D.ModelFramework
using FVM_1D.Discretise

using Reexport

@reexport using ILUZero


include("Preconditioners/Preconditioners.jl")
include("Solve_1_api.jl")

end