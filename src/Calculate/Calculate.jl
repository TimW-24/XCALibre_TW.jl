module Calculate

using LinearAlgebra
using StaticArrays
using SparseArrays
using Accessors
using Adapt
using Atomix
using KernelAbstractions
using GPUArrays
# using CUDA
using XCALibre.Mesh
using XCALibre.Fields
using XCALibre.ModelFramework
using XCALibre.Discretise
using XCALibre.Solve

include("Calculate_0_gradient.jl")
include("Calculate_0_divergence.jl")
include("Calculate_1_green_gauss.jl")
include("Calculate_2_interpolation.jl")
include("Calculate_3_orthogonality_correction.jl")
include("Calculate_4_wall_distance.jl")
include("Calculate_5_surface_normal_gradient.jl")

end