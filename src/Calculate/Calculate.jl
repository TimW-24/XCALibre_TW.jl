module Calculate

const WORKGROUP = 32

using LinearAlgebra
using StaticArrays
using LoopVectorization
using SparseArrays
using Adapt
using Atomix
using KernelAbstractions
using CUDA
using FVM_1D.Mesh
using FVM_1D.Fields
using FVM_1D.ModelFramework
using FVM_1D.Discretise

# include("Calculate_0_types.jl")
include("Calculate_0_gradient.jl")
include("Calculate_0_divergence.jl")
include("Calculate_1_green_gauss.jl")
include("Calculate_2_interpolation.jl")
include("Calculate_3_orthogonality_correction.jl")
include("Calculate_5_surface_normal_gradient.jl")

end