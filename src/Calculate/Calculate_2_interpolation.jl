export correct_boundaries!
export interpolate!

# Function to correct interpolation at boundaries (expands loop to reduce allocations)

@generated function correct_boundaries!(phif, phi, BCs)
    unpacked_BCs = []
    for i ∈ 1:length(BCs.parameters)
        unpack = quote
            
            # BC = BCs[$i]

            # name = BC.name
            # index = boundary_index(boundaries, name)

            # boundary = boundaries[BC.ID]
            # adjust_boundary!(BC, phif, phi, boundary, faces)
            for i in eachindex(BCs)
                (; ID) = BCs[i]
                (; facesID, cellsID) = boundaries_cpu[ID]
                facesID = _convert_array(facesID, backend)
                cellsID = _convert_array(cellsID, backend)
                #KERNEL LAUNCH
                adjust_boundary!(backend, BCs[i], phif, phi, facesID, cellsID)
            end

            # adjust_boundary!(BC, phif, phi, BC.ID, faces)
        end
        push!(unpacked_BCs, unpack)
    end
    quote
    (; mesh) = phif
    (; boundaries) = mesh 

    ## REMOVE MEMCOPY!!!!!!!!!!!!!!!!
    boundaries_cpu = Array{eltype(mesh.boundaries)}(undef, length(boundaries))
    copyto!(boundaries_cpu, boundaries)

    backend = _get_backend(mesh)
    $(unpacked_BCs...) 
    end
end

# function adjust_boundary!(
#     BC::Dirichlet, phif::FaceScalarField, phi, boundary, faces)
#     (; facesID, cellsID) = boundary
#     @inbounds for fID ∈ facesID
#         phif.values[fID] = BC.value 
#     end
# end

# function adjust_boundary!(
#     BC::Neumann, phif::FaceScalarField, phi, boundary, faces)
#     (;facesID, cellsID) = boundary
#     @inbounds for fi ∈ eachindex(facesID)
#         fID = facesID[fi]
#         cID = cellsID[fi]
#         # (; normal, e, delta) = faces[fID]
#         phif.values[fID] = phi.values[cID] #+ BC.value*delta*(normal⋅e)
#     end
# end

function adjust_boundary!(backend, BC::Dirichlet, phif::FaceScalarField, phi, facesID, cellsID)
    (; values) = phif
    phif_values = values
    (; values) = phi
    phi_values = values
    kernel! = adjust_boundary_dirichlet_scalar!(backend)
    kernel!(BC, phif, phi, facesID, cellsID, phif_values, phi_values, ndrange = length(facesID))
end

function adjust_boundary!(backend, BC::Neumann, phif::FaceScalarField, phi, facesID, cellsID)
    (; values) = phif
    phif_values = values
    (; values) = phi
    phi_values = values
    kernel! = adjust_boundary_neumann_scalar!(backend)
    kernel!(BC, phif, phi, facesID, cellsID, phif_values, phi_values, ndrange = length(facesID))
end

@kernel function adjust_boundary_dirichlet_scalar!(BC, phif, phi, facesID, cellsID, phif_values, phi_values)
    i = @index(Global)

    @inbounds begin
        fID = facesID[i]
        phif_values[fID] = BC.value
    end
end

@kernel function adjust_boundary_neumann_scalar!(BC, phif, phi, facesID, cellsID, phif_values, phi_values)
    i = @index(Global)

    @inbounds begin
        fID = facesID[i]
        cID = cellsID[i]
        phif_values[fID] = phi_values[cID] 
    end
end

function adjust_boundary!(
    BC::KWallFunction, phif::FaceScalarField, phi, boundary, faces)
    (;facesID, cellsID) = boundary
    @inbounds for fi ∈ eachindex(facesID)
        fID = facesID[fi]
        cID = cellsID[fi]
        phif.values[fID] = phi.values[cID] # Using Neumann condition
    end
end

function adjust_boundary!(
    BC::NutWallFunction, phif::FaceScalarField, phi, boundary, faces)
    (;facesID, cellsID) = boundary
    @inbounds for fi ∈ eachindex(facesID)
        fID = facesID[fi]
        cID = cellsID[fi]
        phif.values[fID] = phi.values[cID] # Using Neumann condition
    end
end

function adjust_boundary!(
    BC::OmegaWallFunction, phif::FaceScalarField, phi, boundary, faces)
    (;facesID, cellsID) = boundary
    @inbounds for fi ∈ eachindex(facesID)
        fID = facesID[fi]
        cID = cellsID[fi]
        phif.values[fID] = phi.values[cID] # Using Neumann condition
    end
end

# function adjust_boundary!( 
#     BC::Dirichlet, psif::FaceVectorField, psi::VectorField, boundary, faces
#     )

#     (; x, y, z) = psif
#     (; facesID) = boundary

#     @inbounds for fID ∈ facesID
#         x[fID] = BC.value[1]
#         y[fID] = BC.value[2]
#         z[fID] = BC.value[3]
#     end
# end

# function adjust_boundary!( 
#     BC::Neumann, psif::FaceVectorField, psi::VectorField, boundary, faces
#     ) 

#     (; x, y, z) = psif
#     (; facesID, cellsID) = boundary

#     @inbounds for fi ∈ eachindex(facesID)
#         fID = facesID[fi]
#         cID = cellsID[fi]
#         psi_cell = psi[cID]
#         # normal = faces[fID].normal
#         # Line below needs sorting out for general user-defined gradients
#         # now only works for zero gradient
#         # psi_boundary =   psi_cell - (psi_cell⋅normal)*normal
#         x[fID] = psi_cell[1]
#         y[fID] = psi_cell[2]
#         z[fID] = psi_cell[3]
#     end
# end

## GPU VECTOR ADJUST BOUNDARY FUNCTIONS AND KERNELS

function adjust_boundary!(backend, BC::Dirichlet, psif::FaceVectorField, psi::VectorField, facesID, cellsID)
    (; x, y, z) = psif
    kernel! = adjust_boundary_dirichlet_vector!(backend)
    kernel!(BC, psif, psi, facesID, cellsID, x, y, z, ndrange = length(facesID))
end

function adjust_boundary!(backend, BC::Neumann, psif::FaceVectorField, psi::VectorField, facesID, cellsID)
    (; x, y, z) = psif
    kernel! = adjust_boundary_neumann_vector!(backend)
    kernel!(BC, psif, psi, facesID, cellsID, x, y, z, ndrange = length(facesID))
end

@kernel function adjust_boundary_dirichlet_vector!(BC, psif, psi, facesID, cellsID, x, y, z)
    i = @index(Global)
    @inbounds begin
        fID = facesID[i]
        x[fID] = BC.value[1]
        y[fID] = BC.value[2]
        z[fID] = BC.value[3]
    end
end

@kernel function adjust_boundary_neumann_vector!(BC, psif, psi, facesID, cellsID, x, y, z)
    i = @index(Global)
    @inbounds begin
        fID = facesID[i]
        cID = cellsID[i]
        psi_cell = psi[cID]
        # normal = faces[fID].normal
        # Line below needs sorting out for general user-defined gradients
        # now only works for zero gradient
        # psi_boundary =   psi_cell - (psi_cell⋅normal)*normal
        x[fID] = psi_cell[1]
        y[fID] = psi_cell[2]
        z[fID] = psi_cell[3]
    end
end


# SCALAR INTERPOLATION

## CPU code
# function interpolate!(phif::FaceScalarField, phi::ScalarField) 
#     vals = phi.values 
#     fvals = phif.values
#     mesh = phi.mesh 
#     faces = mesh.faces
#     @inbounds for fID ∈ eachindex(faces)
#         # (; weight, ownerCells) = faces[fi]
#         face = faces[fID]
#         weight = face.weight
#         ownerCells = face.ownerCells
#         phi1 = vals[ownerCells[1]]
#         phi2 = vals[ownerCells[2]]
#         one_minus_weight = 1 - weight
#         fvals[fID] = weight*phi1 + one_minus_weight*phi2 # check weight is used correctly!
#     end
# end

## Kernel code
function interpolate!(phif::FaceScalarField, phi::ScalarField)
    # Extract values arrays from scalar fields 
    vals = phi.values
    fvals = phif.values

    # Extract faces from mesh
    mesh = phif.mesh
    faces = mesh.faces

    # Launch interpolate kernel
    backend = _get_backend(mesh)
    kernel! = interpolate_Scalar!(backend)
    kernel!(fvals, vals, faces, ndrange = length(faces))
end

@kernel function interpolate_Scalar!(fvals, vals, faces)
    # Define index for thread
    i = @index(Global)

    @inbounds begin
        # Deconstruct faces to use weight and ownerCells in calculations
        (; weight, ownerCells) = faces[i]

        # Calculate initial values based on index queried from ownerCells
        phi1 = vals[ownerCells[1]]
        phi2 = vals[ownerCells[2]]

        # Calculate one minus weight
        one_minus_weight = 1 - weight

        # Update phif values array for interpolation
        fvals[i] = weight*phi1 + one_minus_weight*phi2 # check weight is used correctly!
    end
end

# VECTOR INTERPOLATION

## CPU code
# function interpolate!(psif::FaceVectorField, psi::VectorField)
#     (; x, y, z) = psif # must extend to 3D
#     mesh = psi.mesh
#     faces = mesh.faces
#     @inbounds for fID ∈ eachindex(faces)
#         # (; weight, ownerCells) = faces[fID]
#         face = faces[fID]
#         weight = face.weight
#         ownerCells = face.ownerCells
#         # w, df = weight(Linear, cells, faces, fi)
#         cID1 = ownerCells[1]; cID2 = ownerCells[2]
#         x1 = psi.x[cID1]; x2 = psi.x[cID2]
#         y1 = psi.y[cID1]; y2 = psi.y[cID2]
#         one_minus_weight = 1 - weight
#         x[fID] = weight*x1 + one_minus_weight*x2 # check weight is used correctly!
#         y[fID] = weight*y1 + one_minus_weight*y2 # check weight is used correctly!
#     end
# end

## Kernel code
function interpolate!(psif::FaceVectorField, psi::VectorField)
    # Extract x, y, z, values from FaceVectorField
    (; mesh, x, y, z) = psif
    xf = x; yf = y; zf = z; #Redefine x, y, z values to be used in kernel

    # Extract x, y, z, values from VectorField
    (; x, y, z) = psi

    #Extract faces array from mesh
    faces = mesh.faces

    # Launch interpolate kernel
    backend = _get_backend(mesh)
    kernel! = interpolate_Vector!(backend)
    kernel!(x, y, xf, yf, faces, ndrange = length(faces))
end

@kernel function interpolate_Vector!(x, y, xf, yf, faces)
    # Define index for thread
    i = @index(Global)

    @inbounds begin
        # Deconstruct faces to use weight and ownerCells in calculations
        (; weight, ownerCells) = faces[i]

        # Define indices for initial x and y values from psi struct
        cID1 = ownerCells[1]; cID2 = ownerCells[2]
        x1 = x[cID1]; x2 = x[cID2]
        y1 = y[cID1]; y2 = y[cID2]

        # Calculate one minus weight
        one_minus_weight = 1 - weight

        # Update psif x and y arrays for interpolation (IMPLEMENT 3D)
        xf[i] = weight*x1 + one_minus_weight*x2 # check weight is used correctly!
        yf[i] = weight*y1 + one_minus_weight*y2 # check weight is used correctly!
    end
end

# GRADIENT INTERPOLATION

function interpolate!(
    gradf::FaceVectorField, grad::Grad, phi
    )
    (; mesh, x, y, z) = gradf
    (; cells, faces) = mesh
    (; values) = phi
    nbfaces = total_boundary_faces(mesh)
    start = nbfaces + 1
    @inbounds for fID ∈ start:length(faces)
        face = faces[fID]
        (; delta, ownerCells, e) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        grad1 = grad(cID1)
        grad2 = grad(cID2)
        # get weight for current scheme
        w, df = weight(get_scheme(grad), cells, faces, fID)
        one_minus_weight = 1 - w
        # calculate interpolated value
        grad_ave = w*grad1 + one_minus_weight*grad2
        # correct interpolation
        grad_corr = grad_ave + ((values[cID2] - values[cID1])/delta - (grad_ave⋅e))*e
        x[fID] = grad_corr[1]
        y[fID] = grad_corr[2]
        z[fID] = grad_corr[3]
    end
end

# function weight(::Type{Midpoint}, cells, faces, fID)
#     w = 0.5
#     return w
# end

# function correct_gradient_interpolation!(::Type{Linear}, gradf, phi)
#     values = phi.phi.values
#     mesh = phi.mesh
#     (; cells, faces) = mesh
#     start = total_boundary_faces(mesh) + 1
#     finish = length(faces)
#     @inbounds for fID ∈ start:finish
#     # for fi ∈ 1:length(faces)
#         # (; ownerCells, delta, e) = faces[fi]
#         face = faces[fID]
#         ownerCells = face.ownerCells
#         delta = face.ownerCells
#         e = face.e
#         w, df = weight(Linear, cells, faces, fi)
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         grad_ave = gradf(fID)
#         grad_corr = grad_ave + ((values[cID2] - values[cID1])/delta - (grad_ave⋅e))*e
#         gradf.x[fID] = grad_corr[1]
#         gradf.y[fID] = grad_corr[2]
#         gradf.z[fID] = grad_corr[3]
#     end
# end

############ OLD LINEAR GRADIENT INTERPOLATION IMPLEMENTATION #############


# function interpolate!(::Type{Linear}, gradf::FaceVectorField, grad, BCs)
#     (; mesh, x, y, z) = gradf
#     (; cells, faces) = mesh
#     nbfaces = total_boundary_faces(mesh)
#     start = nbfaces + 1
#     @inbounds for fID ∈ start:length(faces)
#         # (; ownerCells) = faces[fi]
#         face = faces[fID]
#         ownerCells = face.ownerCells
#         w, df = weight(Linear, cells, faces, fID)
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         # grad1 = grad(cID1)
#         # grad2 = grad(cID2)
#         grad1 = grad(cID1)
#         grad2 = grad(cID2)
#         one_minus_weight = 1.0 - w
#         gradi = w*grad1 + one_minus_weight*grad2
#         x[fID] = gradi[1]
#         y[fID] = gradi[2]
#         z[fID] = gradi[3]
#     end
#     correct_gradient_interpolation!(Linear, gradf, grad)
#     # boundary faces
#     for BC ∈ BCs
#         bi = boundary_index(boundaries, BC.name)
#         boundary = boundaries[bi]
#         correct_boundary!(BC, gradf, grad, boundary, faces)
#     end
# end

# function correct_interpolation!(
#     ::Type{Linear}, phif::FaceScalarField{I,F}, grad, phif0) where {I,F}
#     mesh = phif.mesh
#     (; cells, faces) = mesh
#     start = total_boundary_faces(mesh) + 1
#     finish = length(faces)
#     @inbounds for fID ∈ start:finish
#         # (; ownerCells) = faces[fi]
#         face = faces[fID]
#         ownerCells = face.ownerCells
#         w, df = weight(Linear, cells, faces, fi)
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         grad1 = grad(cID1)
#         grad2 = grad(cID2)
#         one_minus_weight = 1.0 - w
#         grad_ave = w*grad1 + one_minus_weight*grad2
#         phif.values[fID] = phif0[fID] + grad_ave⋅df
#     end
# end

# function correct_gradient_interpolation!(::Type{Linear}, gradf, phi)
#     values = phi.phi.values
#     mesh = phi.mesh
#     (; cells, faces) = mesh
#     start = total_boundary_faces(mesh) + 1
#     finish = length(faces)
#     @inbounds for fID ∈ start:finish
#     # for fi ∈ 1:length(faces)
#         # (; ownerCells, delta, e) = faces[fi]
#         face = faces[fID]
#         ownerCells = face.ownerCells
#         delta = face.ownerCells
#         e = face.ownerCells
#         w, df = weight(Linear, cells, faces, fi)
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         # c1 = cells[cID1].centre
#         # c2 = cells[cID2].centre
#         # distance = c2 - c1
#         # d = distance/delta
#         d = e
#         # grad1 = grad(cID1)
#         # grad2 = grad(cID2)
#         # grad_ave = w*grad1 + (1.0 - w)*grad2
#         grad_ave = gradf(fID)
#         grad_corr = grad_ave + ((values[cID2] - values[cID1])/delta - (grad_ave⋅d))*d
#         gradf.x[fID] = grad_corr[1]
#         gradf.y[fID] = grad_corr[2]
#         gradf.z[fID] = grad_corr[3]
#     end
# end

# function correct_boundary!( # Another way is to use the boundary value and geometry to calc
#     BC::Dirichlet, gradf::FaceVectorField, grad, boundary, faces)
#     (; mesh, x, y, z) = gradf
#     (; facesID) = boundary
#     @inbounds for fID ∈ facesID
#         face = faces[fID]
#         normal = faces[fID].normal
#         cID = face.ownerCells[1]
#         grad_cell = grad(cID)
#         # grad_cell = grad[cID]
#         grad_boundary = grad_cell #.*normal .+ grad_cell
#         # grad_boundary = ((BC.value - grad.phi.values[fID])/face.delta)*normal
#         x[fID] = grad_boundary[1]
#         y[fID] = grad_boundary[2]
#         z[fID] = grad_boundary[3]
#     end
# end

# function correct_boundary!(
#     BC::Neumann, gradf::FaceVectorField, grad, boundary, faces)
#     (; mesh, x, y, z) = gradf
#     (; facesID) = boundary
#     @inbounds for fID ∈ facesID
#         face = faces[fID]
#         normal = faces[fID].normal
#         cID = face.ownerCells[1]
#         grad_cell = grad(cID)
#         # grad_cell = grad[cID]
#         grad_boundary =   grad_cell - (grad_cell⋅normal)*normal # needs sorting out!
#         x[fID] = grad_boundary[1]
#         y[fID] = grad_boundary[2]
#         z[fID] = grad_boundary[3]
#     end
# end