export Grad
export grad!, source!
export get_scheme

# Define Gradient type and functionality

struct Grad{S<:AbstractScheme,F,R,I,M} <: AbstractField
    field::F
    result::R
    correctors::I
    correct::Bool
    mesh::M
end
function Adapt.adapt_structure(to, itp::Grad{S}) where {S}
    field = Adapt.adapt_structure(to, itp.field); F = typeof(field)
    result = Adapt.adapt_structure(to, itp.result); R = typeof(result)
    correctors = Adapt.adapt_structure(to, itp.correctors); I = typeof(correctors)
    correct = Adapt.adapt_structure(to, itp.correct)
    mesh = Adapt.adapt_structure(to, itp.mesh); M = typeof(mesh)
    Grad{S,F,R,I,M}(field, result, correctors, correct, mesh)
end
Grad{S}(phi::ScalarField) where S= begin
    mesh = phi.mesh
    grad = VectorField(mesh)
    F = typeof(phi)
    R = typeof(grad)
    I = _get_int(mesh)
    M = typeof(mesh)
    Grad{S,F,R,I,M}(phi, grad, one(I), false, mesh)
end

Grad{S}(psi::VectorField) where S = begin
    mesh = psi.mesh
    tgrad = TensorField(mesh)
    F = typeof(psi)
    R = typeof(tgrad)
    # I = eltype(mesh.nodes[1].neighbourCells)
    I = _get_int(mesh)
    M = typeof(mesh)
    Grad{S,F,R,I,M}(psi, tgrad, one(I), false, mesh)
end

# Grad{S}(phi::ScalarField, correctors::Integer) where S = begin 
#     mesh = phi.mesh
#     (; cells) = mesh
#     ncells = length(cells)
#     F = eltype(mesh.nodes[1].coords)
#     I = eltype(mesh.nodes[1].neighbourCells)
#     SF = typeof(phi)
#     M = typeof(mesh)
#     gradx = zeros(F, ncells)
#     grady = zeros(F, ncells)
#     gradz = zeros(F, ncells)
#     Grad{S,I,F,SF,M}(phi, gradx, grady, gradz, correctors, true, mesh)
# end
# get_scheme(term::Grad{S,I,F}) where {S,I,F} = S

Base.getindex(grad::Grad{S,F,R,I,M}, i::Integer) where {S,F,R<:VectorField,I,M} = begin
    Tf = eltype(grad.result.x.values)
    SVector{3,Tf}(
        grad.result.x[i], 
        grad.result.y[i], 
        grad.result.z[i]
        )
end

Base.getindex(grad::Grad{S,F,R,I,M}, i::Integer) where {S,F,R<:AbstractTensorField,I,M} = begin
    Tf = eltype(grad.result.xx.values)
    tensor = grad.result
    SMatrix{3,3,Tf,9}(
        tensor.xx[i],
        tensor.yx[i],
        tensor.zx[i],
        tensor.xy[i],
        tensor.yy[i],
        tensor.zy[i],
        tensor.xz[i],
        tensor.yz[i],
        tensor.zz[i],
        )
end

Base.getindex(t::T{Grad{S,F,R,I,M}}, i::Integer) where {S,F,R<:AbstractTensorField,I,M} = begin
    tensor = t.parent.result
    Tf = eltype(tensor.xx.values)
    SMatrix{3,3,Tf,9}(
        tensor.xx[i],
        tensor.xy[i],
        tensor.xz[i],
        tensor.yx[i],
        tensor.yy[i],
        tensor.yz[i],
        tensor.zx[i],
        tensor.zy[i],
        tensor.zz[i],
        )
end

# GRADIENT CALCULATION FUNCTIONS

## Orthogonal (uncorrected) gradient calculation

function grad!(grad::Grad{Orthogonal,F,R,I,M}, phif, phi, BCs) where {F,R<:VectorField,I,M}
    interpolate!(phif, phi)
    correct_boundaries!(phif, phi, BCs)
    green_gauss!(grad.result.x, grad.result.y, grad.result.z, phif)
    # for i ∈ 1:2
    #     correct_interpolation!(grad, phif, phi)
    #     green_gauss!(grad, phif; source)
    # end
end

function grad!(grad::Grad{Orthogonal,F,R,I,M}, psif, psi, BCs) where {F,R<:TensorField,I,M}
    interpolate!(psif, psi)
    correct_boundaries!(psif, psi, BCs)
    green_gauss!(grad.result.xx, grad.result.yx, grad.result.zx, psif.x)
    green_gauss!(grad.result.xy, grad.result.yy, grad.result.zy, psif.y)
    green_gauss!(grad.result.xz, grad.result.yz, grad.result.zz, psif.z)
end

## Mid-point gradient calculation

# interpolate_midpoint!(phif::FaceScalarField, phi::ScalarField) = begin
#     mesh = phi.mesh
#     (; faces) = mesh
#     for i ∈ eachindex(faces)
#         owners = faces[i].ownerCells 
#         c1 = owners[1]
#         c2 = owners[2]
#         phif[i] = 0.5*(phi[c1] + phi[c2])
#     end
# end

function interpolate_midpoint!(phif::FaceScalarField, phi::ScalarField)
    (; mesh) = phi
    (; faces) = mesh
    backend = _get_backend(mesh)
    kernel! = interpolate_midpoint_scalar!(backend, 2)
    kernel!(faces, phif, phi, ndrange = length(faces))
    KernelAbstractions.synchronize(backend)
end

@kernel function interpolate_midpoint_scalar!(faces, phif, phi)
    i = @index(Global)

    @inbounds begin
        (; ownerCells) = faces[i]
        c1 = ownerCells[1]
        c2 = ownerCells[2]
        phif[i] = 0.5*(phi[c1] + phi[c2])
    end
end

interpolate_midpoint!(psif::FaceVectorField, psi::VectorField) = begin
    (; x, y, z) = psif
    mesh = psi.mesh
    faces = mesh.faces
    weight = 0.5
    @inbounds for fID ∈ eachindex(faces)
        face = faces[fID]
        weight = face.weight
        ownerCells = face.ownerCells
        c1 = ownerCells[1]; c2 = ownerCells[2]
        x1 = psi.x[c1]; x2 = psi.x[c2]
        y1 = psi.y[c1]; y2 = psi.y[c2]
        z1 = psi.z[c1]; z2 = psi.z[c2]
        x[fID] = weight*(x1 + x2)
        y[fID] = weight*(y1 + y2)
        z[fID] = weight*(z1 + z2)
    end
end

# correct_interpolation!(dx,dy,dz, phif, phi) = begin
#     (; mesh, values) = phif
#     (; faces, cells, nbfaces) = mesh
#     F = _get_float(mesh)
#     phic = phi.values
#     # nbfaces = total_boundary_faces(mesh)
#     start = nbfaces + 1
#     weight = 0.5
#     @inbounds @simd for fID ∈ start:length(faces)
#         face = faces[fID]
#         ownerCells = face.ownerCells
#         owner1 = ownerCells[1]
#         owner2 = ownerCells[2]
#         cell1 = cells[owner1]
#         cell2 = cells[owner2]
#         phi1 = phic[owner1]
#         phi2 = phic[owner2]
#         ∇phi1 = SVector{3, F}(dx[owner1], dy[owner1], dz[owner1])
#         ∇phi2 = SVector{3, F}(dx[owner2], dy[owner2], dz[owner2])
#         rf = face.centre 
#         rP = cell1.centre 
#         rN = cell2.centre
#         phifᵖ = weight*(phi1 + phi2)
#         ∇phi = weight*(∇phi1 + ∇phi2)
#         Ri = rf - weight*(rP + rN)
#         values[fID] = phifᵖ + ∇phi⋅Ri
#     end
# end

function correct_interpolation!(dx, dy, dz, phif, phi)
    (; mesh, values) = phif
    (; faces, cells, boundaries) = mesh
    nbfaces = length(mesh.boundary_cellsID)
    F = _get_float(mesh)
    phic = phi.values
    # nbfaces = total_boundary_faces(mesh)
    # start = nbfaces+1
    weight = 0.5
    backend = _get_backend(mesh)
    kernel! = correct_interpolation_kernel!(backend, 2)
    kernel!(faces, cells, nbfaces, phic, F, weight, dx, dy, dz, values, ndrange = length(faces)-nbfaces)
    KernelAbstractions.synchronize(backend)
end

@kernel function correct_interpolation_kernel!(
    faces, cells::AbstractArray{Cell{TF, SV, R}}, nbfaces, phic, F, weight, dx, dy, dz, values) where {TF,SV,R}
    i = @index(Global)
    i += nbfaces

    # (; ownerCells, centre) = faces[i]
    # centre_faces = centre
    face = faces[i]
    rf = face.centre
    ownerCells = face.ownerCells

    owner1 = ownerCells[1]
    owner2 = ownerCells[2]

    # (; centre) = cells[owner1]
    # centre_cell1 = centre
    rP = cells[owner1].centre

    # (; centre) = cells[owner2]
    # centre_cell2 = centre
    rN = cells[owner2].centre

    phi1 = phic[owner1]
    phi2 = phic[owner2]

    # ∇phi1 = @inbounds SVector{3, F}(dx[owner1], dy[owner1], dz[owner1])
    # ∇phi2 = @inbounds SVector{3, F}(dx[owner2], dy[owner2], dz[owner2])

    # ∇phi1 = @inbounds SVector{3}(dx[owner1], dy[owner1], dz[owner1])
    # ∇phi2 = @inbounds SVector{3}(dx[owner2], dy[owner2], dz[owner2])

    ∇phi1 = @SVector TF[dx[owner1], dy[owner1], dz[owner1]]
    ∇phi2 = @SVector TF[dx[owner2], dy[owner2], dz[owner2]]


    phifᵖ = weight*(phi1 + phi2)
    ∇phi = weight*(∇phi1 + ∇phi2)
    Ri = rf - weight*(rP + rN)

    values[i] = phifᵖ + ∇phi⋅Ri
end

function grad!(grad::Grad{Midpoint,F,R,I,M}, phif, phi, BCs) where {F,R<:VectorField,I,M}
    interpolate_midpoint!(phif, phi)
    correct_boundaries!(phif, phi, BCs)
    green_gauss!(grad.result.x, grad.result.y, grad.result.z, phif)
    for i ∈ 1:2
        correct_interpolation!(grad.result.x, grad.result.y, grad.result.z, phif, phi)
        green_gauss!(grad.result.x, grad.result.y, grad.result.z, phif)
    end
end

function grad!(grad::Grad{Midpoint,F,R,I,M}, psif, psi, BCs) where {F,R<:TensorField,I,M}
    interpolate_midpoint!(psif, psi)
    correct_boundaries!(psif, psi, BCs)
    for i ∈ 1:2
    correct_interpolation!(grad.result.xx, grad.result.yx, grad.result.zx, psif.x, psi.x)
    green_gauss!(grad.result.xx, grad.result.yx, grad.result.zx, psif.x)

    correct_interpolation!(grad.result.xy, grad.result.yy, grad.result.zy, psif.y, psi.y)
    green_gauss!(grad.result.xy, grad.result.yy, grad.result.zy, psif.y)

    correct_interpolation!(grad.result.xz, grad.result.yz, grad.result.zz, psif.z, psi.z)
    green_gauss!(grad.result.xz, grad.result.yz, grad.result.zz, psif.z)
    end
end