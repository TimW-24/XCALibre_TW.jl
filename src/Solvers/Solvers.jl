module Solvers

const WORKGROUP = 32

using Accessors
using LoopVectorization
using LinearAlgebra
using Statistics
using Krylov
using LinearOperators
using ProgressMeter
using Printf
using CUDA
using KernelAbstractions
using Atomix
using Adapt

using FVM_1D.Mesh
using FVM_1D.Fields
using FVM_1D.ModelFramework
using FVM_1D.Discretise
using FVM_1D.Solve
using FVM_1D.Calculate
using FVM_1D.RANSModels
using FVM_1D.VTK

include("Solvers_0_functions.jl")
include("Solvers_1_SIMPLE.jl")
include("Solvers_2_PISO.jl")

end