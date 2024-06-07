export write_vtk, model2vtk
export copy_to_cpu

function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, name) where {T,F,M,Tu<:Laminar,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p)
    )
    write_vtk(name, model.domain, args...)
end

# function model2vtk(model::RANS{KOmega,F1,F2,V,T,E,D}, name) where {F1,F2,V,T,E,D}
#     args = (
#         ("U", model.U), 
#         ("p", model.p),
#         ("k", model.turbulence.k),
#         ("omega", model.turbulence.omega),
#         ("nut", model.turbulence.nut)
#     )
#     write_vtk(name, model.mesh, args...)
# end

function write_vtk(name, mesh::Mesh2, args...) #, Ux, Uy, Uz, p)
    # UxNodes = FVM.NodeScalarField(Ux)
    # UyNodes = FVM.NodeScalarField(Uy)
    # UzNodes = FVM.NodeScalarField(Uz)
    # pNodes = FVM.NodeScalarField(p)
    # FVM.interpolate2nodes!(UxNodes, Ux)
    # FVM.interpolate2nodes!(UyNodes, Uy)
    # FVM.interpolate2nodes!(UzNodes, Uz)
    # FVM.interpolate2nodes!(pNodes, p)
    (; cell_nodes) = mesh
    filename = name*".vtk"
    open(filename, "w") do io
        write(io, "# vtk DataFile Version 3.0\n")
        write(io, "jCFD simulation data\n")
        write(io, "ASCII\n")
        write(io, "DATASET UNSTRUCTURED_GRID\n")
        nPoints = length(mesh.nodes)
        nCells = length(mesh.cells)
        write(io, "POINTS $(nPoints) double\n")

        backend = _get_backend(mesh)
        nodes_cpu, cells_cpu, cell_nodes_cpu = copy_to_cpu(mesh.nodes, mesh.cells, mesh.cell_nodes, backend)

        for node ∈ nodes_cpu
            (; coords) = node
            println(io, coords[1]," ", coords[2]," ", coords[3])
        end
        sumIndeces = 0
        for cell ∈ cells_cpu
            # sumIndeces += length(cell.nodesID)
            sumIndeces += length(cell.nodes_range)
        end
        cellListSize = sumIndeces + nCells
        write(io, "CELLS $(nCells) $(cellListSize)\n")
        for cell ∈ cells_cpu
            # nNodes = length(cell.nodesID)
            nNodes = length(cell.nodes_range)
            nodes = ""
            # for nodeID ∈ cell.nodesID 
            for ni ∈ cell.nodes_range 
                nodeID = cell_nodes_cpu[ni]
                node = "$(nodeID-1)"
                nodes = nodes*" "*node
            end 
            println(io, nNodes," ", nodes)
        end

        write(io, "CELL_TYPES $(nCells)\n")

        for cell ∈ cells_cpu
            # nCellIDs = length(cell.nodesID)
            nCellIDs = length(cell.nodes_range)
            if nCellIDs == 3
                type = "5"
            elseif nCellIDs == 4
                type = "9"
            elseif nCellIDs > 4
                type = "7"
            end
            println(io, type)
        end

        write(io, "CELL_DATA $(nCells)\n")

        for arg ∈ args
            label = arg[1]
            field = arg[2]
            field_type = typeof(field)
            if field_type <: ScalarField
                write(io, "SCALARS $(label) double 1\n")
                write(io, "LOOKUP_TABLE CellColors\n")
                values_cpu = copy_scalarfield_to_cpu(field.values, backend)
                for value ∈ values_cpu
                    println(io, value)
                end
            elseif field_type <: VectorField
                write(io, "VECTORS $(label) double\n")
                x_cpu, y_cpu, z_cpu = copy_to_cpu(field.x.values, field.y.values, field.z.values, backend)
                for i ∈ eachindex(x_cpu)
                    println(io, x_cpu[i]," ",y_cpu[i] ," ",z_cpu[i] )
                end
            else
                throw("""
                Input data should be a ScalarField or VectorField e.g. ("U", U)
                """)
            end
        end
        
        # write(io, "POINT_DATA $(nPoints)\n")
        # write(io, "SCALARS p double 1\n")
        # write(io, "LOOKUP_TABLE default\n")
        # for p ∈ pNodes.values
        #     println(io, p)
        # end
        # write(io, "VECTORS U double\n")
        # for i ∈ 1:length(UxNodes.values)
        #     println(io, UxNodes.values[i]," ",UyNodes.values[i] ," ",UzNodes.values[i] )
        # end
        # # Boundary information
        # # to be implemented
    end
end

function copy_scalarfield_to_cpu(a, backend::CUDABackend)
    a_cpu = Array{eltype(a)}(undef, length(a))
    
    copyto!(a_cpu, a)
    return a_cpu
end

function copy_scalarfield_to_cpu(a, backend::CPU)
    a_cpu = a
    return a_cpu
end

function copy_to_cpu(a, b, c, backend::CUDABackend)
    a_cpu = Array{eltype(a)}(undef, length(a))
    b_cpu = Array{eltype(b)}(undef, length(b))
    c_cpu = Array{eltype(c)}(undef, length(c))
    
    copyto!(a_cpu, a)
    copyto!(b_cpu, b)
    copyto!(c_cpu, c)
    return a_cpu, b_cpu, c_cpu
end

function copy_to_cpu(a, b, c, backend::CPU)
    a_cpu = a
    b_cpu = b
    c_cpu = c
    return a_cpu, b_cpu, c_cpu
end