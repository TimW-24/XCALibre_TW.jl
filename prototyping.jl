using FVM_1D
using FVM_1D.FoamMesh
using StaticArrays

mesh_file = "unv_sample_meshes/OF_cavity_hex/constant/polyMesh"
@time fowners = build_foamMesh(mesh_file; integer=Int64, float=Float64)

test = "4(14 140 136 15)"

@time s = split(test, ['(',' ', ')'], keepempty=false)
@time s = split(test, ['('], keepempty=false)

@time @inbounds SVector{3}(parse.(Float64, s))

test1 = "{"
test2 = "walls { nFaces          125;"

@time s = split(test1, [' ', ';', '{', '}'], keepempty=false)
@time s = split(test2, [' ', ';', '{', '}'], keepempty=false)

length(s)

parse(Int64, s[2])