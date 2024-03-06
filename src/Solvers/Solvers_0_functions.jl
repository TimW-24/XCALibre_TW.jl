export flux!, update_nueff!, residual!, inverse_diagonal!, remove_pressure_source!, H!, correct_velocity!

# update_nueff!(nueff, nu, turb_model) = begin
#     if turb_model === nothing
#         for i ∈ eachindex(nueff)
#             nueff[i] = nu[i]
#         end
#     else
#         for i ∈ eachindex(nueff)
#             nueff[i] = nu[i] + turb_model.νtf[i]
#         end
#     end
# end

function update_nueff!(nueff, nu, turb_model)
    (; mesh) = nueff
    backend = _get_backend(mesh)
    if turb_model === nothing
        kernel! = update_nueff_laminar!(backend)
        kernel!(nu, nueff, ndrange = length(nueff))
    else
        (; νtf) = turb_model
        kernel! = update_nueff_turbulent!(backend)
        kernel!(nu, νtf, nueff, ndrange = length(nueff))
    end
    
end

@kernel function update_nueff_laminar!(nu, nueff)
    i = @index(Global)

    @inbounds begin
        nueff[i] = nu[i]
    end
end

@kernel function update_nueff_turbulent!(nu, νtf, nueff)
    i = @index(Global)
    
    @inbounds begin
        nueff[i] = nu[i] + νtf[i]
    end
end

function residual!(Residual, equation, phi, iteration)
    (; A, b, R, Fx) = equation
    values = phi.values
    # Option 1

    backend = _get_backend(phi.mesh)
    
    # mul!(Fx, A, values)

    sparse_matmul!(A, values, Fx, backend)
    KernelAbstractions.synchronize(backend)

    @inbounds @. R = abs(Fx - b)^2
    KernelAbstractions.synchronize(backend)

    res = sqrt(mean(R))/norm(b)


    # res = max(norm(R), eps())/abs(mean(values))

    # sum_mean = zero(TF)
    # sum_norm = zero(TF)
    # @inbounds for i ∈ eachindex(R)
    #     sum_mean += values[i]
    #     sum_norm += abs(Fx[i] - b[i])^2
    # end
    # N = length(values)
    # res = sqrt(sum_norm)/abs(sum_mean/N)

    # Option 2
    # mul!(Fx, opA, values)
    # @inbounds @. R = b - Fx
    # sum = zero(TF)
    # @inbounds for i ∈ eachindex(R)
    #     sum += (R[i])^2 
    # end
    # res = sqrt(sum/length(R))

    # Option 3 (OpenFOAM definition)
    # solMean = mean(values)
    # mul!(R, opA, values .- solMean)
    # term1 = abs.(R)
    # mul!(R, opA, values)
    # Fx .= b .- R.*solMean./values
    # term2 = abs.(Fx)
    # N = sum(term1 .+ term2)
    # res = (1/N)*sum(abs.(b .- R))

    # term1 = abs.(opA*(values .- solMean))
    # term2 = abs.(b .- R.*solMean./values)
    # N = sum(term1 .+ term2)
    # res = (1/N)*sum(abs.(b - opA*values))

    # print("Residual: ", res, " (", niterations(solver), " iterations)\n") 
    # @printf "\tResidual: %.4e (%i iterations)\n" res niterations(solver)
    # return res
    Residual[iteration] = res
    nothing
end

function sparse_matmul!(a, b, c, backend)
    if size(a)[2] != length(b)
        error("Matrix size mismatch!")
        return nothing
    end

    nzval_array = nzval(a)
    colptr_array = colptr(a)
    rowval_array = rowval(a)
    fzero = zero(eltype(c))

    kernel! = matmul_copy_zeros_kernel!(backend)
    kernel!(c, fzero, ndrange = length(c))
    KernelAbstractions.synchronize(backend)

    kernel! = sparse_matmul_kernel!(backend)
    kernel!(nzval_array, rowval_array, colptr_array, b, c, ndrange=length(c))
    KernelAbstractions.synchronize(backend)
end

@kernel function sparse_matmul_kernel!(nzval, rowval, colptr, mulvec, res)
    i = @index(Global)

    # res[i] = zero(eltype(res))

    # @inbounds begin
        @synchronize
        start = colptr[i]
        fin = colptr[i+1]

        for j in start:fin-1
            val = nzval[j] #A[j,i]
            row = rowval[j] #Row index of non-zero element in A
            Atomix.@atomic res[row] += mulvec[i] * val
        end
    # end
end

@kernel function matmul_copy_zeros_kernel!(c, fzero)
    i = @index(Global)

    @inbounds begin
        c[i] = fzero
    end
end

# function flux!(phif::FS, psif::FV) where {FS<:FaceScalarField,FV<:FaceVectorField}
#     (; mesh, values) = phif
#     (; faces) = mesh 
#     @inbounds for fID ∈ eachindex(faces)
#         (; area, normal) = faces[fID]
#         Sf = area*normal
#         values[fID] = psif[fID]⋅Sf
#     end
# end

function flux!(phif::FS, psif::FV) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; mesh, values) = phif
    (; faces) = mesh

    backend = _get_backend(mesh)
    kernel! = flux_kernel!(backend)
    kernel!(faces, values, psif, ndrange = length(faces))
end

@kernel function flux_kernel!(faces, values, psif)
    i = @index(Global)

    @inbounds begin
        (; area, normal) = faces[i]
        Sf = area*normal
        values[i] = psif[i]⋅Sf
    end
end

volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

# function inverse_diagonal!(rD::S, eqn) where S<:ScalarField
#     (; mesh, values) = rD
#     cells = mesh.cells
#     A = eqn.A
#     @inbounds for i ∈ eachindex(values)
#         # D = view(A, i, i)[1]
#         D = A[i,i]
#         volume = cells[i].volume
#         values[i] = volume/D
#     end
# end

function inverse_diagonal!(rD::S, eqn) where S<:ScalarField
    (; mesh, values) = rD
    cells = mesh.cells
    A = eqn.A
    nzval_array = nzval(A)
    colptr_array = colptr(A)
    rowval_array = rowval(A)
    backend = _get_backend(mesh)

    ione = one(_get_int(mesh))

    kernel! = inverse_diagonal_kernel!(backend)
    kernel!(ione, colptr_array, rowval_array, nzval_array, cells, values, ndrange = length(values))
end

@kernel function inverse_diagonal_kernel!(ione, colptr, rowval, nzval, cells, values)
    # i from 1 to number of variables in values
    i = @index(Global)

    @inbounds begin
        nIndex = nzval_index(colptr, rowval, i, i, ione)
        D = nzval[nIndex]
        (; volume) = cells[i]
        values[i] = volume/D
    end
end

# function correct_velocity!(U, Hv, ∇p, rD)
#     Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
#     dpdx = ∇p.result.x; dpdy = ∇p.result.y; rDvalues = rD.values
#     @inbounds @simd for i ∈ eachindex(Ux)
#         rDvalues_i = rDvalues[i]
#         Ux[i] = Hvx[i] - dpdx[i]*rDvalues_i
#         Uy[i] = Hvy[i] - dpdy[i]*rDvalues_i
#     end
# end

function correct_velocity!(U, Hv, ∇p, rD)
    Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.result.x; dpdy = ∇p.result.y; rDvalues = rD.values
    backend = _get_backend(U.mesh)

    kernel! = correct_velocity_kernel!(backend)
    kernel!(rDvalues, Ux, Hvx, dpdx, Uy, Hvy, dpdy, ndrange = length(Ux))
end

@kernel function correct_velocity_kernel!(rDvalues,
                                          Ux, Hvx, dpdx,
                                          Uy, Hvy, dpdy)
    i = @index(Global)
    
    @inbounds begin
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] - dpdx[i]*rDvalues_i
        Uy[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

# remove_pressure_source!(ux_eqn::M1, uy_eqn::M2, ∇p) where {M1,M2} = begin # Extend to 3D
#     cells = get_phi(ux_eqn).mesh.cells
#     source_sign = get_source_sign(ux_eqn, 1)
#     dpdx, dpdy = ∇p.result.x, ∇p.result.y
#     bx, by = ux_eqn.equation.b, uy_eqn.equation.b
#     @inbounds for i ∈ eachindex(bx)
#         volume = cells[i].volume
#         bx[i] -= source_sign*dpdx[i]*volume
#         by[i] -= source_sign*dpdy[i]*volume
#     end
# end

remove_pressure_source!(ux_eqn::M1, uy_eqn::M2, ∇p) where {M1,M2} = begin # Extend to 3D
    backend = _get_backend(get_phi(ux_eqn).mesh)
    cells = get_phi(ux_eqn).mesh.cells
    source_sign = get_source_sign(ux_eqn, 1)
    dpdx, dpdy = ∇p.result.x, ∇p.result.y
    bx, by = ux_eqn.equation.b, uy_eqn.equation.b

    kernel! = remove_pressure_source_kernel!(backend)
    kernel!(cells, source_sign, dpdx, dpdy, bx, by, ndrange = length(bx))
end

@kernel function remove_pressure_source_kernel!(cells, source_sign, dpdx, dpdy, bx, by) #Extend to 3D
    # i ranges from 1 to number of elements in bx
    i = @index(Global)

    @inbounds begin
        (; volume) = cells[i]
        bx[i] -= source_sign*dpdx[i]*volume
        by[i] -= source_sign*dpdy[i]*volume
    end
end

# H!(Hv, v::VF, ux_eqn, uy_eqn) where VF<:VectorField = 
# begin # Extend to 3D!
#     (; x, y, z, mesh) = Hv 
#     (; cells, faces) = mesh
#     (; cells, cell_neighbours, faces) = mesh
#     Ax = ux_eqn.equation.A; Ay = uy_eqn.equation.A
#     bx = ux_eqn.equation.b; by = uy_eqn.equation.b
#     vx, vy = v.x, v.y
#     F = eltype(v.x.values)
#     @inbounds for cID ∈ eachindex(cells)
#         cell = cells[cID]
#         # (; neighbours, volume) = cell
#         (; volume) = cell
#         sumx = zero(F)
#         sumy = zero(F)
#         # @inbounds for nID ∈ neighbours
#         @inbounds for ni ∈ cell.faces_range 
#             nID = cell_neighbours[ni]
#             sumx += Ax[cID,nID]*vx[nID]
#             sumy += Ay[cID,nID]*vy[nID]
#         end

#         D = view(Ax, cID, cID)[1] # add check to use max of Ax or Ay)
#         rD = 1/D
#         # rD = volume/D
#         x[cID] = (bx[cID] - sumx)*rD
#         y[cID] = (by[cID] - sumy)*rD
#         z[cID] = zero(F)
#     end
# end


function H!(Hv, v::VF, ux_eqn, uy_eqn) where VF<:VectorField # Extend to 3D!
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    (; cells, cell_neighbours, faces) = mesh
    backend = _get_backend(mesh)

    Ax = ux_eqn.equation.A
    bx = ux_eqn.equation.b
    nzval_x = nzval(Ax)
    colptr_x = colptr(Ax)
    rowval_x = rowval(Ax)
    
    Ay = uy_eqn.equation.A
    by = uy_eqn.equation.b
    nzval_y = nzval(Ay)
    colptr_y = colptr(Ay)
    rowval_y = rowval(Ay)
    
    vx, vy = v.x, v.y
    F = _get_float(mesh)
    ione = one(_get_int(mesh))
    
    kernel! = H_kernel!(backend)
    kernel!(ione, cells, F, cell_neighbours,
            nzval_x, colptr_x, rowval_x, bx, vx,
            nzval_y, colptr_y, rowval_y, by, vy,
            x, y, z, ndrange = length(cells))
end

@kernel function H_kernel!(ione, cells, F, cell_neighbours,
                           nzval_x, colptr_x, rowval_x, bx, vx,
                           nzval_y, colptr_y, rowval_y, by, vy,
                            x, y, z) #Extend to 3D!
    i = @index(Global)

    @inbounds begin
        sumx = zero(F)
        sumy = zero(F)
        (; faces_range) = cells[i]

        for ni ∈ faces_range
            nID = cell_neighbours[ni]
            xIndex = nzval_index(colptr_x, rowval_x, nID, i, ione)
            yIndex = nzval_index(colptr_y, rowval_y, nID, i, ione)
            sumx += nzval_x[xIndex]*vx[nID]
            sumy += nzval_y[yIndex]*vy[nID]
        end

        # D = view(Ax, i, i)[1] # add check to use max of Ax or Ay)
        DIndex = nzval_index(colptr_x, rowval_x, i, i, ione)
        D = nzval_x[DIndex]
        rD = 1/D
        # rD = volume/D
        x[i] = (bx[i] - sumx)*rD
        y[i] = (by[i] - sumy)*rD
        z[i] = zero(F)
    end
end

courant_number(U, mesh::Mesh2, runtime) = begin
    dt = runtime.dt 
    co = zero(_get_float(mesh))
    # courant_max = zero(_get_float(mesh))
    cells = mesh.cells
    for i ∈ eachindex(U)
        umag = norm(U[i])
        volume = cells[i].volume
        dx = sqrt(volume)
        co = max(co, umag*dt/dx)
    end
    return co
end