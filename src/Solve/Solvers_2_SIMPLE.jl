export isimple!, flux!

function isimple!(
    mesh::Mesh2{TI,TF}, velocity, nu, ux, uy, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations
    ; resume=true, pref=nothing) where {TI,TF}

    n_cells = m = n = length(mesh.cells)

    # Pre-allocate fields
    U = VectorField(mesh)
    Uf = FaceVectorField(mesh)
    # mdot = ScalarField(mesh)
    mdotf = FaceScalarField(mesh)
    pf = FaceScalarField(mesh)
    ∇p = Grad{Linear}(p)
    
    Hv = VectorField(mesh)
    Hvf = FaceVectorField(mesh)
    Hv_flux = FaceScalarField(mesh)
    divHv_new = ScalarField(mesh)
    divHv = Div(Hv, FaceVectorField(mesh), zeros(TF, n_cells), mesh)
    rD = ScalarField(mesh)
    rDf = FaceScalarField(mesh)

    # Pre-allocated auxiliary variables
    ux0 = zeros(TF, n_cells)
    uy0 = zeros(TF, n_cells)
    p0 = zeros(TF, n_cells)

    ux0 .= velocity[1]
    uy0 .= velocity[2]
    p0 .= zero(TF)
    rDf.values .= 1.0

    # Define models 
    x_momentum_model    = create_model(ConvectionDiffusion, mdotf, nu, ux, ∇p.x)
    y_momentum_model    = create_model(ConvectionDiffusion, mdotf, nu, uy, ∇p.y)
    # x_momentum_model    = create_model(ConvectionDiffusion, Uf, nu, ux, ∇p.x)
    # y_momentum_model    = create_model(ConvectionDiffusion, Uf, nu, uy, ∇p.y)

    # pressure_correction = create_model(Diffusion, rDf, p, divHv.values)
    pressure_correction = create_model(Diffusion, rDf, p, divHv_new.values)

    # Define equations
    x_momentum_eqn  = Equation(mesh)
    y_momentum_eqn  = Equation(mesh)
    pressure_eqn    = Equation(mesh)

    # Define preconditioners and linear operators
    opAx = LinearOperator(x_momentum_eqn.A)
    Px = ilu0(x_momentum_eqn.A)
    opPUx = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, Px, v))
    
    opAy = LinearOperator(y_momentum_eqn.A)
    Py = ilu0(y_momentum_eqn.A)
    opPUy = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, Py, v))
    
    discretise!(pressure_eqn, pressure_correction)
    apply_boundary_conditions!(pressure_eqn, pressure_correction, pBCs)
    opAp = LinearOperator(pressure_eqn.A)
    opPP = opLDL(pressure_eqn.A)

    solver_p = setup_p.solver(pressure_eqn.A, pressure_eqn.b)
    solver_U = setup_U.solver(x_momentum_eqn.A, x_momentum_eqn.b)

    #### NEED TO IMPLEMENT A SENSIBLE INITIALISATION TO INCLUDE WARM START!!!!
    # Update initial (guessed) fields

    @turbo ux0 .= ux.values
    @turbo uy0 .= uy.values 
    @turbo p0 .= p.values
    @turbo U.x .= ux.values #velocity[1]
    @turbo U.y .= uy.values# velocity[2]
    # @inbounds ux.values .= velocity[1]
    # @inbounds uy.values .= velocity[2]
    # end
    volume  = volumes(mesh)
    rvolume  = 1.0./volume
    
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, UBCs)
    flux!(mdotf, Uf)

    source!(∇p, pf, p, pBCs)
    
    # Perform SIMPLE loops 
    R_ux = TF[]
    @time for iteration ∈ 1:iterations

        print("\n\nIteration ", iteration, "\n") # 91 allocations
        
        print("Solving x-momentum. ")
        
        # source!(∇p, pf, p, pBCs)
        neg!(∇p)

        discretise!(x_momentum_eqn, x_momentum_model)
        @turbo @. y_momentum_eqn.A.nzval = x_momentum_eqn.A.nzval
        apply_boundary_conditions!(x_momentum_eqn, x_momentum_model, uxBCs)
        implicit_relaxation!(x_momentum_eqn, ux0, setup_U.relax)
        ilu0!(Px, x_momentum_eqn.A)
        run!(
            x_momentum_eqn, x_momentum_model, uxBCs, 
            setup_U, opA=opAx, opP=opPUx, solver=solver_U
        )

        res = residual(x_momentum_eqn, ux, opAx, solver_U)
        push!(R_ux, res)
        if res <= 1e-6
            print("\nSimulation converged... Stop!\n")
            break
        end

        print("Solving y-momentum. ")

        @turbo @. y_momentum_eqn.b = 0.0
        apply_boundary_conditions!(y_momentum_eqn, y_momentum_model, uyBCs)
        implicit_relaxation!(y_momentum_eqn, uy0, setup_U.relax)
        ilu0!(Py, y_momentum_eqn.A)
        run!(
            y_momentum_eqn, y_momentum_model, uyBCs, 
            setup_U, opA=opAy, opP=opPUy, solver=solver_U
        )

        @turbo for i ∈ eachindex(ux0)
            ux0[i] = U.x[i]
            uy0[i] = U.y[i]
            # U.x[i] = ux.values[i]
            # U.y[i] = uy.values[i]
        end
        
        inverse_diagonal!(rD, x_momentum_eqn)
        interpolate!(rDf, rD)
        remove_pressure_source!(x_momentum_eqn, y_momentum_eqn, ∇p, rD)
        # H!(Hv, U, x_momentum_eqn, y_momentum_eqn)
        H!(Hv, ux, uy, x_momentum_eqn, y_momentum_eqn)
        
        # @turbo for i ∈ eachindex(ux0)
        #     U.x[i] = ux0[i]
        #     U.y[i] = uy0[i]
        # end

        # div!(divHv, UBCs) # 7 allocations
        # @turbo @. divHv.values *= rvolume
        
        interpolate!(Hvf, Hv)
        correct_boundaries!(Hvf, Hv, UBCs)
        flux!(Hv_flux, Hvf)
        div!(divHv_new, Hv_flux)
        # @turbo @. divHv_new.values *= rvolume

        @inbounds @. rD.values *= volume
        interpolate!(rDf, rD)
        @inbounds @. rD.values *= rvolume

        print("Solving pressure correction. ")

        discretise!(pressure_eqn, pressure_correction)
        apply_boundary_conditions!(pressure_eqn, pressure_correction, pBCs)
        setReference!(pressure_eqn, pref, 1)
        run!(
            pressure_eqn, pressure_correction, pBCs, 
            setup_p, opA=opAp, opP=opPP, solver=solver_p
        )
        
        source!(∇p, pf, p, pBCs)
        # grad!(∇p, pf, p, pBCs) 
        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U)
        correct_boundaries!(Uf, U, UBCs)
        flux!(mdotf, Uf)

        
        explicit_relaxation!(p, p0, setup_p.relax)
        source!(∇p, pf, p, pBCs)
        # grad!(∇p, pf, p, pBCs) 
        correct_velocity!(ux, uy, Hv, ∇p, rD)
    end # end for loop
    return R_ux, U, Uf        
end # end function


function residual(equation::Equation{TI,TF}, phi, opA, solver) where {TI,TF}
    (; A, b, R, Fx) = equation
    values = phi.values
    # Option 1
    # mul!(Fx, opA, values)
    # @inbounds @. R = abs(Fx - b)
    # res = norm(R)/mean(values)

    # Option 2
    # mul!(Fx, opA, values)
    # @inbounds @. R = b - Fx
    # sum = zero(TF)
    # @inbounds for i ∈ eachindex(R)
    #     sum += (R[i])^2 
    # end
    # res = sqrt(sum/length(R))

    # Option 3 (OpenFOAM definition)
    solMean = mean(values)
    term1 = abs.(opA*(values .- solMean))
    term2 = abs.(b .- opA*solMean*values./values)
    N = sum(term1 + term2)
    res = (1/N)*sum(abs.(b - opA*values))

    # print("Residual: ", res, " (", niterations(solver), " iterations)\n") 
    @printf "Residual: %.4e (%i iterations)\n" res niterations(solver)
    return res
end

function calculate_residuals(
    divU::ScalarField, UxEqn::Equation, UyEqn::Equation, Ux::ScalarField, Uy::ScalarField)
    
    continuityError = abs(mean(-divU.values))
    # UxResidual = norm(UxEqn.A*Ux.values - UxEqn.b)/Rx₀
    # UyResidual = norm(UyEqn.A*Uy.values - UyEqn.b)/Ry₀

    solMean = mean(Ux.values)
    N = sum(
        abs.(UxEqn.A*(Ux.values .- solMean)) + 
        abs.(UxEqn.b .- UxEqn.A*ones(length(UxEqn.b))*solMean)
        )
    UxResidual = (1/N)*sum(abs.(UxEqn.b - UxEqn.A*Ux.values))

    solMean = mean(Uy.values)
    N = sum(
        abs.(UyEqn.A*(Uy.values .- solMean)) + 
        abs.(UyEqn.b .- UyEqn.A*ones(length(UyEqn.b))*solMean)
        )
    UyResidual = (1/N)*sum(abs.(UyEqn.b - UyEqn.A*Uy.values))

    # UxResidual = sum(sqrt.((UxEqn.b - UxEqn.A*Ux.values).^2))/length(UxEqn.b)/Rx₀
    # UyResidual = sum(sqrt.((UyEqn.b - UyEqn.A*Uy.values).^2))/length(UyEqn.b)/Ry₀

    return continuityError, UxResidual, UyResidual
end


function flux!(phif::FaceScalarField{TI,TF}, psif::FaceVectorField{TI,TF}) where {TI,TF}
    (; mesh, values) = phif
    (; faces) = mesh 
    @inbounds for fID ∈ eachindex(faces)
        (; area, normal) = faces[fID]
        Sf = area*normal
        values[fID] = psif(fID)⋅Sf
    end
end

function implicit_relaxation!(eqn::Equation{I,F}, field, alpha) where {I,F}
    (; A, b) = eqn
    @inbounds @simd for i ∈ eachindex(b)
        A[i,i] /= alpha
        b[i] += (1.0 - alpha)*A[i,i]*field[i]
    end
end

# function correct_face_velocity!(Uf, p, )
#     mesh = Uf.mesh
#     (; cells, faces) = mesh
#     nbfaces = total_boundary_faces(mesh)
#     for fID ∈ (nbfaces + 1):length(faces)
#         face = faces[fID]
#         gradp = 0.0
#         Uf.x = nothing
#         ################
#         # CONTINUE 
#         ################
#     end
# end

volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

# function correct_boundary_Hvf!(Hvf, ux, uy, ∇pf, UBCs)
#     mesh = ux.mesh
#     for BC ∈ UBCs
#         if typeof(BC) <: Neumann
#             bi = boundary_index(mesh, BC.name)
#             boundary = mesh.boundaries[bi]
#             correct_flux_boundary!(BC, phif, phi, boundary, faces)
#         end
#     end
# end

# function correct_flux_boundary!(
#     BC::Neumann, phif::FaceScalarField{I,F}, phi, boundary, faces) where {I,F}
#     (; facesID, cellsID) = boundary
#     for fID ∈ facesID
#         phif.values[fID] = BC.value 
#     end
# end

function inverse_diagonal!(rD::ScalarField{I,F}, eqn) where {I,F}
    # D = @view eqn.A[diagind(eqn.A)]
    # @. rD.values = 1.0./D
    A = eqn.A
    rD_values = rD.values
    @inbounds for i ∈ eachindex(rD_values)
        # ap = @view A[i,i]
        # rD_values[i] = 1.0/ap
        rD_values[i] = 1.0/view(A, i, i)[1]
    end
end

function explicit_relaxation!(phi, phi0, alpha)
    # @. phi.values = alpha*phi.values + (1.0 - alpha)*phi0
    # @. phi0 = phi.values
    values = phi.values
    @inbounds @simd for i ∈ eachindex(values)
        # values[i] = alpha*values[i] + (1.0 - alpha)*phi0[i]
        values[i] = phi0[i] + alpha*(values[i] - phi0[i])
        phi0[i] = values[i]
    end
end

function correct_velocity!(U, Hv, ∇p, rD)
    # @. U.x = Hv.x - ∇p.x*rD.values
    # @. U.y = Hv.y - ∇p.y*rD.values
    Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.x; dpdy = ∇p.y; rDvalues = rD.values
    @inbounds @simd for i ∈ eachindex(Ux)
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] - dpdx[i]*rDvalues_i
        Uy[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

function correct_velocity!(ux, uy, Hv, ∇p, rD)
    # @. ux.values = Hv.x - ∇p.x*rD.values
    # @. uy.values = Hv.y - ∇p.y*rD.values
    u = ux.values; v = uy.values; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.x; dpdy = ∇p.y; rDvalues = rD.values
    @inbounds @simd for i ∈ eachindex(u)
        rDvalues_i = rDvalues[i]
        u[i] = Hvx[i] - dpdx[i]*rDvalues_i
        v[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

function neg!(∇p)
    # ∇p.x .*= -1.0
    # ∇p.y .*= -1.0
    dpdx = ∇p.x; dpdy = ∇p.y
    @inbounds for i ∈ eachindex(dpdx)
        dpdx[i] *= -1.0
        dpdy[i] *= -1.0
    end
end

function remove_pressure_source!(x_momentum_eqn, y_momentum_eqn, ∇p, rD)
    # @. x_momentum_eqn.b -= ∇p.x/rD.values
    # @. y_momentum_eqn.b -= ∇p.y/rD.values
    dpdx, dpdy, rD = ∇p.x, ∇p.y, rD.values
    bx, by = x_momentum_eqn.b, y_momentum_eqn.b
    @inbounds for i ∈ eachindex(bx)
        # rDi = rD[i]
        # bx[i] -= dpdx[i]/rDi
        # by[i] -= dpdy[i]/rDi
        bx[i] -= dpdx[i]
        by[i] -= dpdy[i]
    end
end

function setReference!(pEqn::Equation{TI,TF}, pRef, cellID::TI) where {TI,TF}
    if pRef === nothing
        return nothing
    else
        pEqn.b[cellID] += pEqn.A[cellID,cellID]*pRef
        pEqn.A[cellID,cellID] += pEqn.A[cellID,cellID]
    end
end

function H!(Hv::VectorField, v::VectorField{I,F}, xeqn, yeqn) where {I,F}
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = xeqn.A;  Ay = yeqn.A
    bx = xeqn.b; by = yeqn.b; # bz = zeros(length(bx))
    
    vx, vy = v.x, v.y
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        (; neighbours) = cell
        sumx = zero(F)
        sumy = zero(F)
        @inbounds for nID ∈ neighbours
            sumx += Ax[cID,nID]*vx[nID]
            sumy += Ay[cID,nID]*vy[nID]
        end
        rD = 1.0/Ax[cID, cID]
        x[cID] = (bx[cID] - sumx)*rD
        y[cID] = (by[cID] - sumy)*rD
        z[cID] = zero(F)
    end
end

function H!(
    Hv::VectorField, ux::ScalarField{I,F}, uy::ScalarField{I,F}, xeqn, yeqn
    ) where {I,F}
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = xeqn.A;  Ay = yeqn.A
    bx = xeqn.b; by = yeqn.b; # bz = zeros(length(bx))
    ux_vals = ux.values
    uy_vals = uy.values
    
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        (; neighbours) = cell
        sumx = zero(F)
        sumy = zero(F)
        @inbounds for nID ∈ neighbours
            sumx += Ax[cID,nID]*ux_vals[nID]
            sumy += Ay[cID,nID]*uy_vals[nID]
        end
        rD = 1.0/Ax[cID, cID]
        x[cID] = (bx[cID] - sumx)*rD
        y[cID] = (by[cID] - sumy)*rD
        z[cID] = zero(F)
    end
end