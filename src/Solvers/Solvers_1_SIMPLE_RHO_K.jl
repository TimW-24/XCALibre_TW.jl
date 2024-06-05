export simple_rho_K!

function simple_rho_K!(model, thermodel, config; resume=true, pref=nothing) 

    @info "Extracting configuration and input fields..."
    (; U, p, energy, nu, mesh) = model
    (; solvers, schemes, runtime) = config

    (; rho) = thermodel

    transonic = false

    @info "Preallocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    # nuf = ConstantScalar(nu) # Implement constant field!
    rhorDf = FaceScalarField(mesh)
    mueff = FaceScalarField(mesh)
    keff_by_cp = FaceScalarField(mesh)
    initialise!(rhorDf, 1.0)
    divHv = ScalarField(mesh)
    # rho = ScalarField(mesh)
    initialise!(rho, 1.0)
    divK = ScalarField(mesh)
    divU = ScalarField(mesh)
    gradDivU = Grad{schemes.U.gradient}(divU)

    @info "Defining models..."

    ux_eqn = (
        Time{schemes.U.time}(rho, U.x)
        + Divergence{schemes.U.divergence}(mdotf, U.x) 
        - Laplacian{schemes.U.laplacian}(mueff, U.x) 
        == 
        -Source(∇p.result.x)
    ) → Equation(mesh)
    
    uy_eqn = (
        Time{schemes.U.time}(rho, U.y)
        + Divergence{schemes.U.divergence}(mdotf, U.y) 
        - Laplacian{schemes.U.laplacian}(mueff, U.y) 
        == 
        -Source(∇p.result.y)
    ) → Equation(mesh)

    uz_eqn = (
        Time{schemes.U.time}(rho, U.z)
        + Divergence{schemes.U.divergence}(mdotf, U.z) 
        - Laplacian{schemes.U.laplacian}(mueff, U.z) 
        == 
        -Source(∇p.result.z)
    ) → Equation(mesh)

    p_eqn = (
        Laplacian{schemes.p.laplacian}(rhorDf, p) == Source(divHv)
    ) → Equation(mesh)

    # Actually using enthalpy -> energy = cp * T
    energy_eqn = (
        Time{schemes.energy.time}(rho, energy)
        + Divergence{schemes.energy.divergence}(mdotf, energy) 
        - Laplacian{schemes.energy.laplacian}(keff_by_cp, energy) 
        == 
        -Source(divK)
    ) → Equation(mesh)

    @info "Initialising preconditioners..."
    
    @reset ux_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
    @reset uy_eqn.preconditioner = ux_eqn.preconditioner
    @reset uz_eqn.preconditioner = ux_eqn.preconditioner
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, runtime)
    @reset energy_eqn.preconditioner = set_preconditioner(
                    solvers.energy.preconditioner, energy_eqn, energy.BCs, runtime)

    @info "Pre-allocating solvers..."
     
    @reset ux_eqn.solver = solvers.U.solver(_A(ux_eqn), _b(ux_eqn))
    @reset uy_eqn.solver = solvers.U.solver(_A(uy_eqn), _b(uy_eqn))
    @reset uz_eqn.solver = solvers.U.solver(_A(uz_eqn), _b(uz_eqn))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))
    @reset energy_eqn.solver = solvers.energy.solver(_A(energy_eqn), _b(energy_eqn))

    if isturbulent(model)
        @info "Initialising turbulence model..."
        turbulence = initialise_RANS(mdotf, p_eqn, config, model)
        config = turbulence.config
    else
        turbulence = nothing
    end

    R_ux, R_uy, R_uz, R_p, R_e = SIMPLE_RHO_K_loop(
    model, thermodel, ∇p, gradDivU, ux_eqn, uy_eqn, uz_eqn, p_eqn, energy_eqn, turbulence, transonic, config ; resume=resume, pref=pref)

    return R_ux, R_uy, R_uz, R_p, R_e     
end # end function

function SIMPLE_RHO_K_loop(
    model, thermodel, ∇p, gradDivU, ux_eqn, uy_eqn, uz_eqn, p_eqn, energy_eqn, turbulence, transonic, config ; resume, pref)
    
    # Extract model variables and configuration
    (;mesh, U, p, energy, nu) = model
    (; rho, rhof, Psi, Psif) = thermodel
    # ux_model, uy_model = ux_eqn.model, uy_eqn.model
    p_model = p_eqn.model
    (; solvers, schemes, runtime) = config
    (; iterations, write_interval) = runtime

    # Need to replace this with ThermoModel
    R = 287.0
    Cp = 1005.0
    Pr = 0.7
    
    rho = get_flux(ux_eqn, 1)
    mdotf = get_flux(ux_eqn, 2)
    mueff = get_flux(ux_eqn, 3)
    rhorDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)
    keff_by_cp = get_flux(energy_eqn, 3)
    divK = get_source(energy_eqn, 1)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT)
    S2 = ScalarField(mesh)
    # ∇U = Grad{schemes.U.gradient}(U)
    divU = gradDivU.field
    divUf = FaceScalarField(mesh)
    # gradDivU = Grad{schemes.U.gradient}(divU)

    # Temp sources to test GradUT explicit source
    # divUTx = zeros(Float64, length(mesh.cells))
    # divUTy = zeros(Float64, length(mesh.cells))

    n_cells = length(mesh.cells)
    Uf = FaceVectorField(mesh)
    pf = FaceScalarField(mesh)
    energyf = FaceScalarField(mesh)
    # rhof = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    rhorD = ScalarField(mesh)
    # Kf = FaceVectorField(mesh)
    Kf = FaceScalarField(mesh)
    K = ScalarField(mesh)
    Psi = ScalarField(mesh)
    Psif = FaceScalarField(mesh)

    # Pre-allocate auxiliary variables

    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)  

    # Pre-allocate vectors to hold residuals 

    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    R_e = ones(TF, iterations)

    
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, U.BCs)
    interpolate!(energyf, energy)   
    correct_boundaries!(energyf, energy, energy.BCs)

    thermo_Psi!(thermodel, energy, energyf, Psi, Psif)

    interpolate!(pf, p)   
    correct_boundaries!(pf, p, p.BCs)

    thermo_rho!(thermodel, p, pf, Psi, Psif, rho, rhof)

    flux!(mdotf, Uf, rhof)
    grad!(∇p, pf, p, p.BCs)

    update_nueff!(mueff, nu, rhof, turbulence)
    
    for i ∈ eachindex(Uf)
        Kf.values[i] = 0.5*norm(Uf[i])^2*mdotf.values[i]
    end
    for i ∈ eachindex(K)
        K.values[i] = 0.5*norm(U[i])^2
    end
    correct_face_interpolation!(Kf, K, Uf) 
    div!(divK, Kf)


    # Calculate keff_by_cp
    @. keff_by_cp.values = mueff.values/Pr
    volumes = getproperty.(mesh.cells, :volume)

    @info "Staring SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations
        
        # Set up and solve momentum equations

        
        
        # wallBC!(ux_eqn, uy_eqn, U, mesh, mueff)
        
        @. prev = U.x.values
        # ux_eqn.b .-= divUTx
        discretise!(ux_eqn, prev, runtime)
        apply_boundary_conditions!(ux_eqn, U.x.BCs)
        implicit_relaxation_improved!(ux_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(ux_eqn.preconditioner)
        run!(ux_eqn, solvers.U) #opP=Pu.P, solver=solver_U)
        residual!(R_ux, ux_eqn.equation, U.x, iteration)

        @. prev = U.y.values        
        # uy_eqn.b .-= divUTy
        discretise!(uy_eqn, prev, runtime)
        apply_boundary_conditions!(uy_eqn, U.y.BCs)
        implicit_relaxation_improved!(uy_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(uy_eqn.preconditioner)
        run!(uy_eqn, solvers.U)
        residual!(R_uy, uy_eqn.equation, U.y, iteration)

        if typeof(mesh) <: Mesh3
            @. prev = U.z.values
            discretise!(uz_eqn, prev, runtime)
            apply_boundary_conditions!(uz_eqn, U.z.BCs)
            # uy_eqn.b .-= divUTy
            implicit_relaxation!(uz_eqn.equation, prev, solvers.U.relax)
            update_preconditioner!(uz_eqn.preconditioner)
            run!(uz_eqn, solvers.U)
            residual!(R_uz, uz_eqn.equation, U.z, iteration)
        end

        # Set up and solve energy equation
        @. prev = energy.values
        discretise!(energy_eqn, prev, runtime)
        apply_boundary_conditions!(energy_eqn, energy.BCs)
        implicit_relaxation_improved!(energy_eqn.equation, prev, solvers.energy.relax)
        update_preconditioner!(energy_eqn.preconditioner)
        run!(energy_eqn, solvers.energy)
        residual!(R_e, energy_eqn.equation, energy, iteration)

        # γ = solvers.energy.relax
        # γ = 1
        # @. Psi.values = (1-γ)*Psi.values + γ*Cp/(R*energy.values)
        interpolate!(energyf, energy)
        # correct_face_interpolation(energyf, energy, Uf) 
        correct_boundaries!(energyf, energy, energy.BCs)
        # @. Psif.values = (1-γ)*Psif.values + γ*Cp/(R*energyf.values)

        thermo_Psi!(thermodel, energy, energyf, Psi, Psif)

        inverse_diagonal!(rD, ux_eqn.equation, uy_eqn.equation, uz_eqn.equation)
        @. rD.values *= rho.values
        interpolate!(rDf, rD)
        @. rhorDf.values = rDf.values

        remove_pressure_source!(ux_eqn, uy_eqn, uz_eqn, ∇p)
        H!(Hv, U, ux_eqn, uy_eqn, uz_eqn)

        interpolate!(Uf, Hv) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs)
        @. Uf.x.values *= rhof.values
        @. Uf.y.values *= rhof.values
        @. Uf.z.values *= rhof.values
        div!(divHv, Uf)

        # Set up and solve pressure equation
        @. prev = p.values
        discretise!(p_eqn, prev, runtime)
        apply_boundary_conditions!(p_eqn, p.BCs)
        setReference!(p_eqn.equation, pref, 1)
        update_preconditioner!(p_eqn.preconditioner)
        run!(p_eqn, solvers.p)
        explicit_relaxation!(p, prev, solvers.p.relax)
        residual!(R_p, p_eqn.equation, p, iteration)

        grad!(∇p, pf, p, p.BCs) 

        # correct = false
        # if correct
        #     ncorrectors = 1
        #     for i ∈ 1:ncorrectors
        #         discretise!(p_eqn)
        #         apply_boundary_conditions!(p_eqn, p.BCs)
        #         setReference!(p_eqn.equation, pref, 1)
        #         # grad!(∇p, pf, p, pBCs) 
        #         interpolate!(gradpf, ∇p, p)
        #         nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
        #         correct!(p_eqn.equation, p_model.terms.term1, pf)
        #         run!(p_model, solvers.p)
        #         grad!(∇p, pf, p, pBCs) 
        #     end
        # end

        # γ = solvers.p.relax
        # γ = 1
        # @. rho.values = (1- γ)*rho.values + γ*p.values*Psi.values
        interpolate!(pf, p)
        correct_face_interpolation!(pf, p, Uf)
        correct_boundaries!(pf, p, p.BCs)
        # @. rhof.values = (1- γ)*rhof.values + γ*pf.values*Psif.values

        thermo_rho!(thermodel, p, pf, Psi, Psif, rho, rhof)

        flux!(mdotf, Uf) # Uf here is Hvf (reused memory above)
        interpolate!(pf, p)   
        correct_boundaries!(pf, p, p.BCs)
        # interpolate!(gradpf, ∇p, p)
        pgrad = face_normal_gradient(p, pf, ∇p, gradpf)
        # @. mdotf.values -= pgrad.values*rhorDf.values # Not sure if this or one below
        @. mdotf.values -= pgrad.values*rhof.values*rDf.values


        # Correct velocity and mass flux
        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U)
        correct_boundaries!(Uf, U, U.BCs)
           
        if isturbulent(model)
            grad!(gradU, Uf, U, U.BCs)
            turbulence!(turbulence, model, S, S2, prev) 
            # update_nueff!(muff, nu, turbulence)
            update_nueff!(mueff, nu, rhof, turbulence)
        end
        
        # Update stuff
        update_nueff!(mueff, nu, rhof, turbulence)
        for i ∈ eachindex(Uf)
            Kf.values[i] = 0.5*norm(Uf[i])^2*mdotf.values[i]
        end
        div!(divK, Kf)


        @. keff_by_cp.values = mueff.values./Pr

        
        convergence = 1e-12

        if (R_uz[iteration] == one(TF) &&
            R_ux[iteration] <= convergence && 
            R_uy[iteration] <= convergence || 
            R_uz[iteration] <= convergence && 
            R_p[iteration] <= convergence && 
            R_e[iteration] <= convergence)

            print(
                """
                \n\n\n\n\n
                Simulation converged! $iteration iterations in
                """)
                if !signbit(write_interval)
                    model2vtk(model, @sprintf "iteration_%.6d" iteration)
                end
            break
        end

        ProgressMeter.next!(
            progress, showvalues = [
                (:iter,iteration),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:p, R_p[iteration]),
                (:energy, R_e[iteration]),
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0
            model2vtk(model, @sprintf "iteration_%.6d" iteration)
            write_vtk((@sprintf "rho_%.6d" iteration), mesh, ("rho", rho))
            write_vtk((@sprintf "gradDivU_%.6d" iteration), mesh, ("gradDivU", gradDivU.result))
            
        end

    end # end for loop
    return R_ux, R_uy, R_uz, R_p, R_e
end

# function face_normal_gradient(phi::ScalarField, phif::FaceScalarField)
#     mesh = phi.mesh
#     sngrad = FaceScalarField(mesh)
#     (; faces, cells) = mesh
#     nbfaces = length(mesh.boundary_cellsID) #boundary_faces(mesh)
#     start_faceID = nbfaces + 1
#     last_faceID = length(faces)
#     for fID ∈ start_faceID:last_faceID
#     # for fID ∈ eachindex(faces)
#         face = faces[fID]
#         (; area, normal, ownerCells, delta) = face 
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         cell1 = cells[cID1]
#         cell2 = cells[cID2]
#         phi1 = phi[cID1]
#         phi2 = phi[cID2]
#         face_grad = area*(phi2 - phi1)/delta
#         sngrad.values[fID] = face_grad
#     end
#     # Now deal with boundary faces
#     for fID ∈ 1:nbfaces
#         face = faces[fID]
#         (; area, normal, ownerCells, delta) = face 
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         cell1 = cells[cID1]
#         cell2 = cells[cID2]
#         phi1 = phi[cID1]
#         phi2 = phi[cID2]
#         face_grad = area*(phi2 - phi1)/delta
#         sngrad.values[fID] = face_grad
#     end
#     sngrad
# end

function face_normal_gradient(phi::ScalarField, phif::FaceScalarField, grad, gradf)
    mesh = phi.mesh
    sngrad = FaceScalarField(mesh)
    (; faces, cells) = mesh
    nbfaces = length(mesh.boundary_cellsID) #boundary_faces(mesh)
    start_faceID = nbfaces + 1
    last_faceID = length(faces)
    for fID ∈ start_faceID:last_faceID
    # for fID ∈ eachindex(faces)
        face = faces[fID]
        (; area, e, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        cell1 = cells[cID1]
        cell2 = cells[cID2]
        phi1 = phi[cID1]
        phi2 = phi[cID2]
        gradf_face = gradf[fID]
        # Minimum Correction approach
        costheta = (e[1]*normal[1] + e[2]*normal[2] + e[3]*normal[3]) 
        Ef = costheta*area
        face_grad = Ef*(phi2 - phi1)/delta
        Tf1 = (normal[1] - costheta*e[1])*area
        Tf2 = (normal[2] - costheta*e[2])*area
        Tf3 = (normal[3] - costheta*e[3])*area
        gradf_face = 0.5*(grad[cID1] + grad[cID2])
        face_grad_int = Tf1*gradf_face[1] + Tf2*gradf_face[2] + Tf3*gradf_face[3]
        sngrad.values[fID] = face_grad + face_grad_int
    end
    # Now deal with boundary faces
    for fID ∈ 1:nbfaces
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        cell1 = cells[cID1]
        cell2 = cells[cID2]
        phi1 = phi[cID1]
        phi2 = phi[cID2]
        face_grad = area*(phi2 - phi1)/delta
        sngrad.values[fID] = face_grad
    end
    sngrad
end

function correct_face_interpolation!(phif::FaceScalarField, phi, Uf)
    mesh = phif.mesh
    (; faces, cells) = mesh
    for fID ∈ eachindex(faces)
        face = faces[fID]
        (; ownerCells, area, normal) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        phi1 = phi[cID1]
        phi2 = phi[cID2]
        flux = area*normal⋅Uf[fID]
        if flux > 0.0
            phif.values[fID] = phi1
        else
            phif.values[fID] = phi2
        end
    end
end

function correct_face_interpolation!(phif::FaceVectorField, phi, Uf)
    mesh = phif.mesh
    (; faces, cells) = mesh
    for fID ∈ eachindex(faces)
        face = faces[fID]
        (; ownerCells, area, normal) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        phi1 = phi[cID1]
        phi2 = phi[cID2]
        flux = area*normal⋅Uf[fID]
        if flux > 0.0
            phif.x.values[fID] = phi1[1]
            phif.y.values[fID] = phi1[2]
            phif.z.values[fID] = phi1[3]
        else
            phif.x.values[fID] = phi2[1]
            phif.y.values[fID] = phi2[2]
            phif.z.values[fID] = phi2[3]
        end
    end
end

function wallBC!(ux_eqn, uy_eqn, U, mesh, nueff)
    (; boundaries, boundary_cellsID, faces, cells) = mesh
    for bci ∈ 1:length(U.x.BCs)
        if U.x.BCs[bci] isa Wall{}
            (; IDs_range) = boundaries[U.x.BCs[bci].ID]
            
            @inbounds for i ∈ eachindex(IDs_range)
                faceID = IDs_range[i]
                cellID = boundary_cellsID[faceID]
                face = faces[faceID]
                cell = cells[cellID]
                (; area, normal, delta) = face 

                Uc = U.x.values[cellID]
                Vc = U.y.values[cellID]
                nueff_face = nueff.values[faceID]

                ux_eqn.equation.A[cellID, cellID] += nueff_face*area*(1)/delta
                uy_eqn.equation.A[cellID, cellID] += nueff_face*area*(1)/delta

                ux_eqn.equation.b[cellID] += nueff_face*area*((0)*(1-normal[1]*normal[1]) + (Vc-0)*(normal[2]*normal[1]))/delta - Uc*nueff_face*area*(normal[1]*normal[1])/delta
                uy_eqn.equation.b[cellID] += nueff_face*area*((Uc-0)*(normal[1]*normal[2]) + (0)*(1-normal[2]*normal[2]))/delta - Vc*nueff_face*area*(normal[2]*normal[2])/delta
            end
        end
        if U.x.BCs[bci] isa Symmetry{}
            (; IDs_range) = boundaries[U.x.BCs[bci].ID]
            
            @inbounds for i ∈ eachindex(IDs_range)
                faceID = IDs_range[i]
                cellID = boundary_cellsID[faceID]
                face = faces[faceID]
                cell = cells[cellID]
                (; area, normal, delta) = face 

                Uc = U.x.values[cellID]
                Vc = U.y.values[cellID]
                nueff_face = nueff.values[faceID]

                # ux_eqn.equation.A[cellID, cellID] += nueff_face*area*(1)/delta
                # uy_eqn.equation.A[cellID, cellID] += nueff_face*area*(1)/delta

                ux_eqn.equation.b[cellID] -= 2.0*nueff_face*area*(Uc*normal[1] + Vc*normal[2])*normal[1]/delta
                uy_eqn.equation.b[cellID] -= 2.0*nueff_face*area*(Uc*normal[1] + Vc*normal[2])*normal[2]/delta
            end
        end
    end
    return ux_eqn, uy_eqn
end