module Calculate

using LinearAlgebra
using StaticArrays
using LoopVectorization
using SparseArrays
using Statistics

using FVM_1D.Mesh
using FVM_1D.Fields
using FVM_1D.ModelFramework
using FVM_1D.Discretise
using FVM_1D.Solve

# include("Calculate_0_types.jl")
include("Calculate_0_gradient.jl")
include("Calculate_0_divergence.jl")
include("Calculate_1_green_gauss.jl")
include("Calculate_2_interpolation.jl")
include("Calculate_3_orthogonality_correction.jl")
include("Calculate_4_wall_distance.jl")
include("Calculate_5_surface_normal_gradient.jl")

end