export pressure_force, viscous_force
export stress_tensor, wall_shear_stress
export lift_to_drag, aero_coeffs

lift_to_drag(patch::Symbol, model, ρ, nu) = begin
    oldstd = stdout
    redirect_stdout(devnull)
    Fp = pressure_force(patch, model.momentum.p, ρ)
    Fv = viscous_force(patch, model.momentum.U, ρ, nu, model.turbulence.nut)
    redirect_stdout(oldstd)
    Ft = Fp + Fv
    aero_eff = Ft[2]/Ft[1]
    print("Aerofoil L/D: ",round(aero_eff,sigdigits = 4))
    return aero_eff
end

aero_coeffs(patch::Symbol, chord::R where R <: Real, velocity::Vector{Float64}, model, ρ, nu) = begin
    oldstd = stdout
    redirect_stdout(devnull)
    Fp = pressure_force(patch, model.momentum.p, ρ)
    Fv = viscous_force(patch, model.momentum.U, ρ, nu, model.turbulence.nut)
    redirect_stdout(oldstd)
    Ft = Fp + Fv
    C_l = 2Ft[2]/(ρ*(velocity[1]^2)*chord*0.001)
    C_d = 2Ft[1]/(ρ*(velocity[1]^2)*chord*0.001)
    print("Lift Coefficient: ",round(C_l,sigdigits = 4))
    print("\nDrag Coefficient: ",round(C_d,sigdigits = 4))
    return C_l,C_d
end

pressure_force(patch::Symbol, p::ScalarField, rho) = begin
    mesh = p.mesh
    ID = boundary_index(mesh.boundaries, patch)
    @info "calculating pressure forces on patch: $patch at index $ID"
    boundary = mesh.boundaries[ID]
    # (; facesID, cellsID) = boundary
    (; IDs_range) = boundary
    x = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    y = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    z = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    snflux = FaceVectorField(x,y,z, mesh)
    surface_flux!(snflux, p, IDs_range)
    sumx, sumy, sumz = 0.0, 0.0, 0.0, 0.0
    for i ∈ eachindex(snflux.x)
        # fID = facesID[i]
        # fID = IDs_range[i]
        sumx += snflux.x[i]
        sumy += snflux.y[i]
        sumz += snflux.z[i]
    end
    Fp = rho.*[sumx, sumy, sumz]
    print("\n Pressure force: (", Fp[1], " ", Fp[2], " ", Fp[3], ")\n")
    return Fp
end

viscous_force(patch::Symbol, U::VectorField, rho, ν, νt) = begin
    mesh = U.mesh
    (; faces, boundaries, boundary_cellsID) = mesh
    nboundaries = length(U.BCs)
    ID = boundary_index(boundaries, patch)
    @info "calculating viscous forces on patch: $patch at index $ID"
    boundary = mesh.boundaries[ID]
    # (; facesID, cellsID) = boundary
    (; IDs_range) = boundary
    x = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    y = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    z = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    snGrad = FaceVectorField(x,y,z, mesh)
    for i ∈ 1:nboundaries
        if ID == U.BCs[i].ID
        surface_normal_gradient!(snGrad, U, U.BCs[i].value, IDs_range)
        end
    end
    sumx, sumy, sumz = 0.0, 0.0, 0.0, 0.0
    for i ∈ eachindex(snGrad)
        # fID = facesID[i]
        # cID = cellsID[i]
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        face = faces[fID]
        area = face.area
        sumx += snGrad.x[i]*area*(ν + νt[cID]) # this may need using νtf? (wall funcs)
        sumy += snGrad.y[i]*area*(ν + νt[cID])
        sumz += snGrad.z[i]*area*(ν + νt[cID])
    end
    Fv = rho.*[sumx, sumy, sumz]
    print("\n Viscous force: (", Fv[1], " ", Fv[2], " ", Fv[3], ")\n")
    return Fv
end

########### Must update
# wall_shear_stress(patch::Symbol, model::RANS{M,F1,F2,V,T,E,D}) where {M,F1,F2,V,T,E,D} = begin
#     # Line below needs to change to do selection based on nut BC
#     M == Laminar ? nut = ConstantScalar(0.0) : nut = model.turbulence.nut
#     (; mesh, U, nu) = model
#     (; boundaries, faces) = mesh
#     ID = boundary_index(boundaries, patch)
#     boundary = boundaries[ID]
#     (; facesID, cellsID) = boundary
#     @info "calculating viscous forces on patch: $patch at index $ID"
#     x = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
#     y = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
#     z = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
#     tauw = FaceVectorField(x,y,z, mesh)
#     Uw = zero(_get_float(mesh))
#     for i ∈ 1:length(U.BCs)
#         if ID == U.BCs[i].ID
#             Uw = U.BCs[i].value
#         end
#     end
#     surface_normal_gradient(tauw, facesID, cellsID, U, Uw)
#     pos = fill(SVector{3,Float64}(0,0,0), length(facesID))
#     for i ∈ eachindex(tauw)
#         fID = facesID[i]
#         cID = cellsID[i]
#         face = faces[fID]
#         nueff = nu[cID]  + nut[cID]
#         tauw.x[i] *= nueff # this may need using νtf? (wall funcs)
#         tauw.y[i] *= nueff
#         tauw.z[i] *= nueff
#         pos[i] = face.centre
#     end
    
#     return tauw, pos
# end

stress_tensor(U, ν, νt, config) = begin
    mesh = U.mesh
    TF = _get_float(mesh)
    gradU = Grad{Midpoint}(U)
    gradUT = T(gradU)
    Uf = FaceVectorField(U.mesh)
    grad!(gradU, Uf, U, U.BCs, zero(TF), config) # assuming time=0
    grad!(gradU, Uf, U, U.BCs, time, config)
    nueff = ScalarField(U.mesh) # temp variable
    nueff.values .= ν .+ νt.values
    Reff = TensorField(U.mesh)
    for i ∈ eachindex(Reff)
        Reff[i] = -nueff[i].*(gradU[i] .+ gradUT[i])
    end
    return Reff
end

# viscous_forces(patch::Symbol, Reff::TensorField, U::VectorField, rho, ν, νt) = begin
#     mesh = U.mesh
#     faces = mesh.faces
#     ID = boundary_index(mesh.boundaries, patch)
#     @info "calculating viscous forces on patch: $patch at index $ID"
#     boundary = mesh.boundaries[ID]
#     (; facesID, cellsID) = boundary
#     x = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
#     y = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
#     z = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
#     snGrad = FaceVectorField(x,y,z, mesh)
#     surface_flux(snGrad, facesID, cellsID, Reff)
#     # surface_normal_gradient(snGrad, facesID, cellsID, U, U.BCs[ID].value)
#     sumx, sumy, sumz = 0.0, 0.0, 0.0, 0.0
#     for i ∈ eachindex(snGrad)
#         fID = facesID[i]
#         cID = cellsID[i]
#         face = faces[fID]
#         area = face.area
#         sumx += snGrad.x[i] #*area*(ν + νt[cID]) # this may need to be using νtf?
#         sumy += snGrad.y[i] #*area*(ν + νt[cID])
#         sumz += snGrad.z[i] #*area*(ν + νt[cID])
#     end
#     rho.*[sumx, sumy, sumz]
# end