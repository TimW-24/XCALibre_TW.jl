using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")


unv_mesh="src/UNV_3D/TET_PRISM.unv"
unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"

@time mesh = build_mesh3D(unv_mesh)

points, edges, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
efaces[165]
volumes
boundaryElements

a=length(boundaryElements[1].elements)
b=length(boundaryElements[2].elements)
c=length(boundaryElements[3].elements)
d=length(boundaryElements[4].elements)
e=length(boundaryElements[5].elements)
f=length(boundaryElements[6].elements)

x=[57 58 59]

dict=Dict()

for (n,f) in enumerate(x)
    dict[f] = n
end
dict

dict[57]

# Index for volumes error found. Fixing later.

volumes[1].volumes
points[7]
points[52]
points[40]
points[175]
points[169]
points[241]


findall(x->x==[7,52,40],efaces)
findall(x->x==[52,7,40],efaces)
findall(x->x==[40,7,52],efaces)
findall(x->x==[52,40,7],efaces)

for i=1:length(efaces)
    sort!(efaces[i].faces)
end
efaces

face_nodes=[]
for i=1:length(efaces)
    push!(face_nodes,efaces[i].faces)
end

findall(x->x==[7,40,52],face_nodes)

efaces[119]

findall(x->x==[169,175,241],face_nodes)


@time mesh = build_mesh3D(unv_mesh)

cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)
nodes = FVM_1D.UNV_3D.build_nodes(points, node_cells_range)
boundaries = FVM_1D.UNV_3D.build_boundaries(boundaryElements)

nbfaces = sum(length.(getproperty.(boundaries, :IDs_range)))

bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = FVM_1D.UNV_3D.generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes)

iface_nodes, iface_nodes_range, iface_owners_cells = FVM_1D.UNV_3D.generate_internal_faces(volumes, nbfaces, nodes, node_cells)


get_data(array, range, index) = @view array[range[index]]
get_data(array, range) =  array[range] #@view array[range] # 
nodeIDs = get_data
faceIDs = get_data
cellIDs = get_data

iface_nodes_range .= [
iface_nodes_range[i] .+ length(bface_nodes) for i ∈ eachindex(iface_nodes_range)
]

# Concatenate boundary and internal faces
face_nodes = vcat(bface_nodes, iface_nodes)
face_nodes_range = vcat(bface_nodes_range, iface_nodes_range)
face_owner_cells = vcat(bface_owners_cells, iface_owners_cells)


cell_faces, cell_nsign, cell_faces_range, cell_neighbours=FVM_1D.UNV_3D.generate_cell_face_connectivity(volumes, nbfaces, face_owner_cells)

cells = FVM_1D.UNV_3D.build_cells(cell_nodes_range, cell_faces_range)

faces = FVM_1D.UNV_3D.build_faces(face_nodes_range, face_owner_cells)

