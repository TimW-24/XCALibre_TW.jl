using FVM_1D
using FVM_1D.FoamMesh
using StaticArrays

mesh_file = "unv_sample_meshes/OF_cavity_hex/constant/polyMesh"

foamdata = read_foamMesh(mesh_file, Int64, Float64)

connectivity = connect_mesh(foamdata, Int64, Float64)

mesh = generate_mesh(foamdata, connectivity, Int64, Float64)

@time mesh = compute_geometry!(mesh)

field = ScalarField(mesh)
field.values .= 1:length(field.values)
field.values
@time write_vtk("foamMeshTest", mesh, ("F", field))