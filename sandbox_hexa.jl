using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")


unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"

unv_mesh="src/UNV_3D/HEXA_HM.unv"


points, edges, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
efaces
volumes
boundaryElements

cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)  # Should be Hybrid compatible, tested for hexa.
nodes = FVM_1D.UNV_3D.build_nodes(points, node_cells_range) # Hyrbid compatible, works for Tet and Hexa
boundaries = FVM_1D.UNV_3D.build_boundaries(boundaryElements) # Hybrid compatible

nbfaces = sum(length.(getproperty.(boundaries, :IDs_range))) # total boundary faces

bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = 
begin
    FVM_1D.UNV_3D.generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes) # Hybrid compatible, tested with hexa
end

iface_nodes, iface_nodes_range, iface_owners_cells = 
begin 
    FVM_1D.UNV_3D.generate_internal_faces(volumes, nbfaces, nodes, node_cells) # Hybrid compatible, tested with hexa.
end

# NOTE: A function will be needed here to reorder the nodes IDs of "faces" to be geometrically sound! (not needed for tet cells though)
bface_nodes,iface_nodes=FVM_1D.UNV_3D.order_face_nodes(bface_nodes_range,iface_nodes_range,bface_nodes,iface_nodes,nodes)
#2 methods, using old as new function produced negative volumes?

# Shift range of nodes_range for internal faces (since it will be appended)
iface_nodes_range .= [
    iface_nodes_range[i] .+ length(bface_nodes) for i ∈ eachindex(iface_nodes_range)
    ]

# Concatenate boundary and internal faces
face_nodes = vcat(bface_nodes, iface_nodes)
face_nodes_range = vcat(bface_nodes_range, iface_nodes_range)
face_owner_cells = vcat(bface_owners_cells, iface_owners_cells)

# Sort out cell to face connectivity
cell_faces, cell_nsign, cell_faces_range, cell_neighbours = begin
    FVM_1D.UNV_3D.generate_cell_face_connectivity(volumes, nbfaces, face_owner_cells) # Hybrid compatible. Hexa and tet tested.
end

# Build mesh (without calculation of geometry/properties)
cells = FVM_1D.UNV_3D.build_cells(cell_nodes_range, cell_faces_range) # Hybrid compatible. Hexa tested.
faces = FVM_1D.UNV_3D.build_faces(face_nodes_range, face_owner_cells) # Hybrid compatible. Hexa tested.

mesh = Mesh3(
    cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, 
    faces, face_nodes, boundaries, 
    nodes, node_cells,
    SVector{3, Float64}(0.0, 0.0, 0.0), UnitRange{Int64}(0, 0), boundary_cellsID
) # Hexa tested.

# Update mesh to include all geometry calculations required
FVM_1D.UNV_3D.calculate_centres!(mesh) # Uses centroid instead of geometric. Will need changing, should work fine for regular cells and faces
FVM_1D.UNV_3D.calculate_face_properties!(mesh) # Touched up face properties, double check values.
FVM_1D.UNV_3D.calculate_area_and_volume!(mesh)
mesh.cells

