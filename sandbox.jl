using FVM_1D.UNV_3D

using StaticArrays

using Statistics

using LinearAlgebra

unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/700_cell_case.unv"

points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
faces
volumes
boundaryElements

mesh=build_mesh3D(unv_mesh)

mesh.cells[1].faces_range

mesh.cells[5].neighbours_range
mesh.cell_neighbours
mesh.cells[1].neighbours_range

SVector(0.0)