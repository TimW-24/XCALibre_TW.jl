export build_mesh3D
#push! is BANNED! THOSE FOUND USING PUSH WILL BE EXILED!
# [] is also BANNED! No more kneecaps if found using []!
#Exceptions apply to where I have no idea what the size will be

function build_mesh3D(unv_mesh; integer=Int64, float=Float64)
    stats = @timed begin
        println("Loading UNV File...")
        points, edges, bfaces, volumes, boundaryElements = load_3D(
            unv_mesh; integer=integer, float=float)
        println("File Read Successfully")
        println("Generating Mesh...")

        node_cells, node_cells_range = generate_node_cells(points, volumes) #Rewritten, optimized
        nodes = generate_nodes(points, node_cells_range) #Rewritten, optimzied

        faces_nodesIDs, owners_cellIDs = generate_internal_faces(volumes, bfaces, nodes, node_cells) 

        boundary_faces, boundary_face_range = generate_boundary_faces(boundaryElements,bfaces) #Rewritten
        
        boundary_cells = generate_boundary_cells(bfaces, all_cell_faces, all_cell_faces_range) #Rewritten, error found, using face index of boundary_faces instead of bfaces

        face_nodes = generate_face_nodes(faces) #Removed push
        cell_nodes = generate_cell_nodes(volumes) #Removed push

        all_cell_faces = generate_all_cell_faces(faces, cell_face_nodes) # New method needed

        cell_nodes_range = generate_cell_nodes_range(volumes) #Removed push
        face_nodes_range = generate_face_nodes_range(faces) #Removed Push
        all_cell_faces_range = generate_all_cell_faces_range(volumes) #Removed push

        cells_centre = calculate_centre_cell(volumes, nodes) #Removed push

        

        cell_faces, cell_faces_range = generate_cell_faces(bfaces, volumes, all_cell_faces) # Removed push

        boundaries = generate_boundaries(boundaryElements, boundary_face_range) #Removed push

        face_ownerCells = generate_face_ownerCells(faces, all_cell_faces, all_cell_faces_range) #New method approach needed

        faces_area = calculate_face_area(nodes, faces) #Rewrite needed, removed push
        faces_centre = calculate_face_centre(faces, nodes) # Removed push
        faces_normal = calculate_face_normal(nodes, faces, face_ownerCells, cells_centre, faces_centre) # Rewrite needed
        faces_e, faces_delta, faces_weight = calculate_face_properties(faces, face_ownerCells, cells_centre, faces_centre, faces_normal) #Removed push

        cells_volume = calculate_cell_volume(volumes, all_cell_faces_range, all_cell_faces, faces_normal, cells_centre, faces_centre, face_ownerCells, faces_area) #Removed push

        cells = generate_cells(volumes, cells_centre, cells_volume, cell_nodes_range, cell_faces_range) #Removed push
        cell_neighbours = generate_cell_neighbours(cells, cell_faces) # Removed push, new method needed
        faces = generate_faces(faces, face_nodes_range, faces_centre, faces_normal, faces_area, face_ownerCells, faces_e, faces_delta, faces_weight) #Removed push

        cell_nsign = calculate_cell_nsign(cells, faces, cell_faces) #removed push

        get_float = SVector(0.0, 0.0, 0.0)
        get_int = UnitRange(0, 0)

        mesh = Mesh3(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes, boundaries, nodes, node_cells, get_float, get_int, boundary_cells)

    end
    println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    println("Mesh ready!")
    return mesh
    #For unit testing
    #return mesh,cell_face_nodes, node_cells, all_cell_faces,boundary_cells,boundary_faces,all_cell_faces_range
end

# Node connectivity

function generate_node_cells(points, volumes)
    temp_node_cells = [Int64[] for _ ∈ eachindex(points)] # array of vectors to hold cellIDs

    # Add cellID to each point that defines a "volume"
    for (cellID, volume) ∈ enumerate(volumes)
        for nodeID ∈ volume.volumes
            push!(temp_node_cells[nodeID], cellID)
        end
    end

    node_cells_size = sum(length.(temp_node_cells)) # number of cells in node_cells

    index = 0 # change to node cells index
    node_cells = zeros(Int64, node_cells_size)
    node_cells_range = [UnitRange{Int64}(1, 1) for _ ∈ eachindex(points)]
    for (nodeID, cellsID) ∈ enumerate(temp_node_cells)
        for cellID ∈ cellsID
            index += 1
            node_cells[index] = cellID
        end
        node_cells_range[nodeID] = UnitRange{Int64}(index - length(cellsID) + 1, index)
    end
    return node_cells, node_cells_range
end

function generate_nodes(points, cells_range)
    nodes = [Node(SVector{3, Float64}(0.0,0.0,0.0), 1:1) for _ ∈ eachindex(points)]
    @inbounds for i ∈ eachindex(points)
        nodes[i] =  Node(points[i].xyz, cells_range[i])
    end
    return nodes
end

# DEFINE FUNCTIONS
function calculate_face_properties(faces, face_ownerCells, cells_centre, faces_centre, face_normal)
    faces_e = Vector{SVector{3,Float64}}(undef,length(faces))
    faces_delta = Vector{Float64}(undef,length(faces))
    faces_weight = Vector{Float64}(undef,length(faces))
    for i = eachindex(faces) #Boundary Face
        if face_ownerCells[i, 2] == face_ownerCells[i, 1]
            cc = cells_centre[face_ownerCells[i, 1]]
            cf = faces_centre[i]

            d_cf = cf - cc

            delta = norm(d_cf)
            faces_delta[i] = delta
            e = d_cf / delta
            faces_e[i] = e
            weight = one(Float64)
            faces_weight[i] = weight

        else #Internal Face
            c1 = cells_centre[face_ownerCells[i, 1]]
            c2 = cells_centre[face_ownerCells[i, 2]]
            cf = faces_centre[i]
            d_1f = cf - c1
            d_f2 = c2 - cf
            d_12 = c2 - c1

            delta = norm(d_12)
            faces_delta[i] = delta
            e = d_12 / delta
            faces_e[i] = e
            weight = abs((d_1f ⋅ face_normal[i]) / (d_1f ⋅ face_normal[i] + d_f2 ⋅ face_normal[i]))
            faces_weight[i] = weight

        end
    end
    return faces_e, faces_delta, faces_weight
end

function calculate_face_normal(nodes, faces, face_ownerCells, cells_centre, faces_centre) #Rewrite needed
    face_normal = Vector{SVector{3,Float64}}(undef,length(faces))
    for i = eachindex(faces)
        n1 = nodes[faces[i].faces[1]].coords
        n2 = nodes[faces[i].faces[2]].coords
        n3 = nodes[faces[i].faces[3]].coords

        t1x = n2[1] - n1[1]
        t1y = n2[2] - n1[2]
        t1z = n2[3] - n1[3]

        t2x = n3[1] - n1[1]
        t2y = n3[2] - n1[2]
        t2z = n3[3] - n1[3]

        nx = t1y * t2z - t1z * t2y
        ny = -(t1x * t2z - t1z * t2x)
        nz = t1x * t2y - t1y * t2x

        magn2 = (nx)^2 + (ny)^2 + (nz)^2

        snx = nx / sqrt(magn2)
        sny = ny / sqrt(magn2)
        snz = nz / sqrt(magn2)

        normal = SVector(snx, sny, snz)
        face_normal[i] = normal

        if face_ownerCells[i, 2] == face_ownerCells[i, 1]
            cc = cells_centre[face_ownerCells[i, 1]]
            cf = faces_centre[i]

            d_cf = cf - cc

            if d_cf ⋅ face_normal[i] < 0
                face_normal[i] = -1.0 * face_normal[i]
            end
        else
            c1 = cells_centre[face_ownerCells[i, 1]]
            c2 = cells_centre[face_ownerCells[i, 2]]
            cf = faces_centre[i]
            d_12 = c2 - c1

            if d_12 ⋅ face_normal[i] < 0
                face_normal[i] = -1.0 * face_normal[i]
            end
        end
    end
    return face_normal
end

function calculate_face_centre(faces, nodes)
    face_centres = Vector{SVector{3,Float64}}(undef,length(faces))
    for i = eachindex(faces)
        temp_coords = Vector{SVector{3,Float64}}(undef,length(faces[i].faces))
        for ic = 1:length(faces[i].faces)
            temp_coords[ic] = nodes[faces[i].faces[ic]].coords
        end
        face_centres[i] = sum(temp_coords) / length(faces[i].faces)
    end
    return face_centres
end

function calculate_face_area(nodes, faces) # Need to shorten
    face_area= Vector{Float64}(undef,length(faces))
    for i = eachindex(faces)
        if faces[i].faceCount == 3
            n1 = nodes[faces[i].faces[1]].coords
            n2 = nodes[faces[i].faces[2]].coords
            n3 = nodes[faces[i].faces[3]].coords

            t1x = n2[1] - n1[1]
            t1y = n2[2] - n1[2]
            t1z = n2[3] - n1[3]

            t2x = n3[1] - n1[1]
            t2y = n3[2] - n1[2]
            t2z = n3[3] - n1[3]

            area2 = (t1y * t2z - t1z * t2y)^2 + (t1x * t2z - t1z * t2x)^2 + (t1y * t2x - t1x * t2y)^2
            area = sqrt(area2) / 2
            face_area[i]= area
        end

        if faces[i].faceCount > 3
            n1 = nodes[faces[i].faces[1]].coords
            n2 = nodes[faces[i].faces[2]].coords
            n3 = nodes[faces[i].faces[3]].coords

            t1x = n2[1] - n1[1]
            t1y = n2[2] - n1[2]
            t1z = n2[3] - n1[3]

            t2x = n3[1] - n1[1]
            t2y = n3[2] - n1[2]
            t2z = n3[3] - n1[3]

            area2 = (t1y * t2z - t1z * t2y)^2 + (t1x * t2z - t1z * t2x)^2 + (t1y * t2x - t1x * t2y)^2
            area = sqrt(area2) / 2

            for ic = 4:faces[i].faceCount
                n1 = nodes[faces[i].faces[ic]].coords
                n2 = nodes[faces[i].faces[2]].coords
                n3 = nodes[faces[i].faces[3]].coords

                t1x = n2[1] - n1[1]
                t1y = n2[2] - n1[2]
                t1z = n2[3] - n1[3]

                t2x = n3[1] - n1[1]
                t2y = n3[2] - n1[2]
                t2z = n3[3] - n1[3]

                area2 = (t1y * t2z - t1z * t2y)^2 + (t1x * t2z - t1z * t2x)^2 + (t1y * t2x - t1x * t2y)^2
                area = area + sqrt(area2) / 2

            end

            face_area[i]=area

        end
    end
    return face_area
end

function generate_cell_faces(bfaces, volumes, all_cell_faces)
    cell_faces = Vector{Int64}[] # May be a way to preallocate this. For now will leave as []
    cell_face_range = Vector{UnitRange{Int64}}(undef,length(volumes))
    counter_start = 0
    x = 0
    max = length(bfaces)

    for i = eachindex(volumes)
        push!(cell_faces, all_cell_faces[counter_start+1:counter_start+length(volumes[i].volumes)])
        counter_start = counter_start + length(volumes[i].volumes)
        cell_faces[i] = filter(x -> x > max, cell_faces[i])

        if length(cell_faces[i]) == 1
            cell_face_range[i] = UnitRange(x + 1, x + 1)
            x = x + 1
        else
            cell_face_range[i] = UnitRange(x + 1, x + length(cell_faces[i]))
            x = x + length(cell_faces[i])
        end
    end
    cell_faces = reduce(vcat, cell_faces)

    return cell_faces, cell_face_range
end

function calculate_cell_nsign(cells, faces, cell_faces)
    cell_nsign = Vector{Int}(undef,length(cell_faces))
    counter=0
    for i = 1:length(cells)
        centre = cells[i].centre
        for ic = 1:length(cells[i].faces_range)
            fcentre = faces[cell_faces[cells[i].faces_range][ic]].centre
            fnormal = faces[cell_faces[cells[i].faces_range][ic]].normal
            d_cf = fcentre - centre
            fnsign = zero(Int)

            if d_cf ⋅ fnormal > zero(Float64)
                fnsign = one(Int)
            else
                fnsign = -one(Int)
            end
            counter=counter+1
            cell_nsign[counter] = fnsign
        end

    end
    return cell_nsign
end

function calculate_cell_volume(volumes, all_cell_faces_range, all_cell_faces, face_normal, cells_centre, faces_centre, face_ownerCells, faces_area)
    cells_volume = Vector{Float64}(undef,length(volumes))
    for i = eachindex(volumes)
        volume = zero(Float64) # to avoid type instability
        for f = all_cell_faces_range[i]
            findex = all_cell_faces[f]

            normal = face_normal[findex]
            cc = cells_centre[i]
            fc = faces_centre[findex]
            d_fc = fc - cc

            if face_ownerCells[findex, 1] ≠ face_ownerCells[findex, 2]
                if dot(d_fc, normal) < 0.0
                    normal = -1.0 * normal
                end
            end

            volume = volume + (normal[1] * faces_centre[findex][1] * faces_area[findex])

        end
        cells_volume[i] = volume
    end
    return cells_volume
end

function calculate_centre_cell(volumes,nodes)
    cell_centres = Vector{SVector{3,Float64}}(undef,length(volumes))
    for i = eachindex(volumes)
        temp_coords = Vector{SVector{3,Float64}}(undef,length(volumes[i].volumes))
        for ic = eachindex(volumes[i].volumes)
            temp_coords[ic] = nodes[volumes[i].volumes[ic]].coords
        end
        cell_centres[i] = sum(temp_coords) / length(volumes[i].volumes)
    end
    return cell_centres
end

# function calculate_centre_cell(volumes, nodes)
#     centre_store = SVector{3,Float64}[]
#     for i = 1:length(volumes)
#         cell_store = typeof(nodes[volumes[1].volumes[1]].coords)[]
#         for ic = 1:length(volumes[i].volumes)
#             push!(cell_store, nodes[volumes[i].volumes[ic]].coords)
#         end
#         centre = (sum(cell_store) / length(cell_store))
#         push!(centre_store, centre)
#     end
#     return centre_store
# end

function generate_cell_neighbours(cells, cell_faces)
    cell_neighbours = Vector{Int64}(undef,length(cell_faces))
    counter=0
    for ID = 1:length(cells)
        for i = cells[ID].faces_range
            faces = cell_faces[i]
            for ic = 1:length(i)
                face = faces[ic]
                index = findall(x -> x == face, cell_faces)
                if length(index) == 2
                    if i[1] <= index[1] <= i[end]
                        for ip = 1:length(cells)
                            if cells[ip].faces_range[1] <= index[2] <= cells[ip].faces_range[end]
                                counter=counter+1
                                cell_neighbours[counter] = ip
                            end
                        end
                    end
                    if i[1] <= index[2] <= i[end]
                        for ip = 1:length(cells)
                            if cells[ip].faces_range[1] <= index[1] <= cells[ip].faces_range[end]
                                counter=counter+1
                                cell_neighbours[counter] = ip
                            end
                        end
                    end
                end
            end
        end
    end
    return cell_neighbours
end

# NOTE: the function has been written to be extendable to multiple element types
function generate_internal_faces(volumes, bfaces, nodes, node_cells)

    # determine total number of faces based on cell type (including duplicates)
    total_faces = 0
    for volume ∈ volumes
        # add faces for tets
        if volume.volumeCount == 4
            total_faces += 4
        end
        # add conditions to add faces for other cell types
    end

    # Face nodeIDs for each cell is a vector of vectors of vectors :-)
    cells_faces_nodeIDs = Vector{Vector{Int64}}[Vector{Int64}[] for _ ∈ 1:length(volumes)] 

    # Generate all faces for each cell/element/volume
    for (cellID, volume) ∈ enumerate(volumes)
        # Generate faces for tet elements
        if volume.volumeCount == 4
            nodesID = volume.volumes
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[3], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[4]])
        end
        # add conditions for other cell types
    end

    # Sort nodesIDs for each face based on ID (need to correct order later to be physical)
    # this allows to find duplicates later more easily (query on ordered ids is faster)
    for face_nodesID ∈ cells_faces_nodeIDs
        for nodesID ∈ face_nodesID
            sort!(nodesID)
        end
    end

    # Find owner cells for each face
    owners_cellIDs = Vector{Int64}[zeros(Int64, 2) for _ ∈ 1:total_faces]
    facei = 0 # faceID counter (will be reduced to internal faces later)
    for (cellID, faces_nodeIDs) ∈ enumerate(cells_faces_nodeIDs) # loop over cells
        for facei_nodeIDs ∈ faces_nodeIDs # loop over vectors of nodeIDs for each face
            facei += 1 # face counter
            owners_cellIDs[facei][1] = cellID # ID of first cell containing the face
            for nodeID ∈ facei_nodeIDs # loop over ID of each node in the face
                cells_range = nodes[nodeID].cells_range
                node_cellIDs = @view node_cells[cells_range] # find cells that use this node
                for nodei_cellID ∈ node_cellIDs # loop over cells that share the face node
                    if nodei_cellID !== cellID # ensure cellID is not same as current cell 
                        for face ∈ cells_faces_nodeIDs[nodei_cellID]
                            if face == facei_nodeIDs
                                owners_cellIDs[facei][2] = nodei_cellID # set owner cell ID 
                                break
                            end
                        end
                    end
                end
            end
        end
    end

    # Sort all face owner vectors
    sort!.(owners_cellIDs) # in-place sorting

    # Extract nodesIDs for each face from all cells into a vector of vectors
    faces_nodesIDs = Vector{Int64}[Int64[] for _ ∈ 1:total_faces] # nodesID for all faces
    fID = 0 # counter to keep track of faceID
    for celli_faces_nodeIDs ∈ cells_faces_nodeIDs
        for nodesID ∈ celli_faces_nodeIDs
            fID += 1
            faces_nodesIDs[fID] = nodesID
        end
    end

    # Remove duplicates
    unique_indices = unique(i -> faces_nodesIDs[i], eachindex(faces_nodesIDs))
    unique!(faces_nodesIDs)
    keepat!(owners_cellIDs, unique_indices)

    # Remove boundary faces

    total_bfaces = 0 # count boundary faces
    for owners ∈ owners_cellIDs
        if owners[1] == 0
            total_bfaces += 1
        end
    end

    bfaces_indices = zeros(Int64, total_bfaces) # preallocate memory
    counter = 0
    for (i, owners) ∈ enumerate(owners_cellIDs)
        if owners[1] == 0
            counter += 1
            bfaces_indices[counter] = i # contains indices of faces to remove
        end
    end

    deleteat!(owners_cellIDs, bfaces_indices)
    deleteat!(faces_nodesIDs, bfaces_indices)

    println("Removing ", total_bfaces, " (from ", length(bfaces), ") boundary faces")

    return faces_nodesIDs, owners_cellIDs

end

function quad_internal_faces(volumes, faces)
    store_cell_faces1 = Int64[]

    for i = 1:length(volumes)
        cell_faces = zeros(Int, 6, 4)

        cell_faces[1, 1:4] = volumes[i].volumes[1:4]
        cell_faces[2, 1:4] = volumes[i].volumes[5:8]
        cell_faces[3, 1:2] = volumes[i].volumes[1:2]
        cell_faces[3, 3:4] = volumes[i].volumes[5:6]
        cell_faces[4, 1:2] = volumes[i].volumes[3:4]
        cell_faces[4, 3:4] = volumes[i].volumes[7:8]
        cell_faces[5, 1:2] = volumes[i].volumes[2:3]
        cell_faces[5, 3:4] = volumes[i].volumes[6:7]
        cell_faces[6, 1] = volumes[i].volumes[1]
        cell_faces[6, 2] = volumes[i].volumes[4]
        cell_faces[6, 3] = volumes[i].volumes[5]
        cell_faces[6, 4] = volumes[i].volumes[8]

        for ic = 1:6
            push!(store_cell_faces1, cell_faces[ic, :])
        end
    end

    sorted_cell_faces = Int64[]
    for i = 1:length(store_cell_faces1)

        push!(sorted_cell_faces, sort(store_cell_faces1[i]))
    end

    sorted_faces = Int64[]
    for i = 1:length(faces)
        push!(sorted_faces, sort(faces[i].faces))
    end

    internal_faces = setdiff(sorted_cell_faces, sorted_faces)

    for i = 1:length(internal_faces)
        push!(faces, UNV_3D.Face(faces[end].faceindex + 1, faces[end].faceCount, internal_faces[i]))
    end
    return faces
end

function generate_boundaries(boundaryElements, boundary_face_range)
    boundaries = Vector{Boundary{Symbol,UnitRange{Int64}}}(undef,length(boundaryElements))
    for i = eachindex(boundaryElements)
        boundaries[i] = Boundary(Symbol(boundaryElements[i].name), boundary_face_range[i])
    end
    return boundaries
end

function generate_boundary_cells(bfaces, all_cell_faces, all_cell_faces_range)
    boundary_cells = Vector{Int64}(undef,length(bfaces))
    index_all_cell_faces = Vector{Int64}(undef,length(bfaces))
    for ic = eachindex(bfaces) 
        for i in eachindex(all_cell_faces) 
                if all_cell_faces[i] == bfaces[ic].faceindex
                    index_all_cell_faces[ic]=i
                end
        end
        for i = eachindex(all_cell_faces_range) 
            if all_cell_faces_range[i][1] <= index_all_cell_faces[ic] <= all_cell_faces_range[i][end]
                boundary_cells[ic]=i
            end
        end
    end
    return boundary_cells
end

function generate_boundary_faces(boundaryElements,bfaces) #Only works if all bc have more than 1 face, which is very unlikely
    boundary_faces = Vector{Int64}(undef,length(bfaces)) #Same length as bfaces
    counter = 0
    boundary_face_range = Vector{UnitRange{Int64}}(undef,length(boundaryElements))
    for i = eachindex(boundaryElements)
        for n = eachindex(boundaryElements[i].elements)
            counter=counter+1
            boundary_faces[counter] = boundaryElements[i].elements[n]
        end
        boundary_face_range[i] = UnitRange(boundaryElements[i].elements[1], boundaryElements[i].elements[end])
    end
    return boundary_faces, boundary_face_range
end

function generate_face_ownerCells(faces, all_cell_faces, all_cell_faces_range)
    cell_face_index = Vector{Vector{Int64}}(undef, length(faces))
    for i = 1:length(cell_face_index)
        cell_face_index[i] = findall(x -> x == i, all_cell_faces)
    end

    face_owners = zeros(Int, length(cell_face_index), 2)
    for ic = 1:length(all_cell_faces_range)
        for i = 1:length(cell_face_index)
            if all_cell_faces_range[ic][1] <= cell_face_index[i][1] <= all_cell_faces_range[ic][end]
                face_owners[i, 1] = ic
                face_owners[i, 2] = ic
            end

            if length(cell_face_index[i]) == 2
                if all_cell_faces_range[ic][1] <= cell_face_index[i][2] <= all_cell_faces_range[ic][end]
                    face_owners[i, 2] = ic
                end
            end

        end
    end
    return face_owners
end

#Generate Faces

function generate_face_nodes(faces)
    face_nodes = Vector{Int64}(undef, length(faces) * 3) # number of bc faces times number of nodes per face. Tet Only for now.
    counter = 0
    for n = eachindex(faces)
        for i = 1:faces[n].faceCount
            counter = counter + 1
            face_nodes[counter] = faces[n].faces[i]
        end
    end
    return face_nodes
end

#Generate cells
function generate_cell_nodes(volumes)
    cell_nodes = Vector{Int64}(undef, length(volumes) * 4) #length of cells times number of nodes per cell
    counter = 0
    for n = eachindex(volumes)
        for i = 1:volumes[n].volumeCount
            counter = counter + 1
            cell_nodes[counter] = volumes[n].volumes[i]
        end
    end
    return cell_nodes
end

function generate_all_cell_faces(faces, cell_face_nodes)
    sorted_faces = Vector{Vector{Int64}}(undef, length(faces))
    for i = 1:length(faces)
        sorted_faces[i] = sort(faces[i].faces)
    end

    all_cell_faces = zeros(Int, length(cell_face_nodes)) #May only work for Tet
    for i = 1:length(cell_face_nodes)
        all_cell_faces[i] = findfirst(x -> x == cell_face_nodes[i], sorted_faces)
    end
    return all_cell_faces
end

#Nodes Range
function generate_cell_nodes_range(volumes)
    cell_nodes_range = Vector{UnitRange{Int64}}(undef, length(volumes))
    x = 0
    for i = eachindex(volumes)
        cell_nodes_range[i] = UnitRange(x + 1, x + length(volumes[i].volumes))
        x = x + length(volumes[i].volumes)
    end
    return cell_nodes_range
end


function generate_face_nodes_range(faces)
    face_nodes_range = Vector{UnitRange{Int64}}(undef, length(faces))
    x = 0
    for i = eachindex(faces)
        face_nodes_range[i] = UnitRange(x + 1, x + faces[i].faceCount)
        x = x + faces[i].faceCount
    end
    return face_nodes_range
end

function generate_all_cell_faces_range(volumes)
    cell_faces_range = Vector{UnitRange{Int64}}(undef, length(volumes))
    x = 0
    @inbounds for i = eachindex(volumes)
        #Tetra
        if length(volumes[i].volumes) == 4
            cell_faces_range[i] = UnitRange(x + 1, x + 4)
            x = x + 4
        end

        #Hexa
        if length(volumes[i].volumes) == 8
            cell_faces_range[i] = UnitRange(x + 1, x + 6)
            x = x + 6
        end
    end
    return cell_faces_range
end

#Generate cells
function generate_cells(volumes, cells_centre, cells_volume, cell_nodes_range, cell_faces_range)
    cells = Vector{Cell{Float64,SVector{3,Float64},UnitRange{Int64}}}(undef,length(volumes))
    for i = eachindex(volumes)
        cells[i] = Cell(cells_centre[i], cells_volume[i], cell_nodes_range[i], cell_faces_range[i])
    end
    return cells
end

function generate_faces(faces, face_nodes_range, faces_centre, faces_normal, faces_area, face_ownerCells, faces_e, faces_delta, faces_weight)
    faces3D = Vector{Face3D{Float64,SVector{2,Int64},SVector{3,Float64},UnitRange{Int64}}}(undef,length(faces))
    for i = eachindex(faces)
        faces3D[i] = Face3D(face_nodes_range[i], SVector(face_ownerCells[i, 1], face_ownerCells[i, 2]), faces_centre[i], faces_normal[i], faces_e[i], faces_area[i], faces_delta[i], faces_weight[i])
    end
    return faces3D
end