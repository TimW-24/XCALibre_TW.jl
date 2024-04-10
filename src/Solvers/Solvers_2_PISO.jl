export piso!

piso!(model_in, config, backend; resume=true, pref=nothing) = begin

    R_ux, R_uy, R_uz, R_p, model = setup_incompressible_solvers(
        PISO, model_in, config, backend;
        resume=true, pref=nothing
        )
        
    return R_ux, R_uy, R_uz, R_p, model
end

function PISO(
    model, ∇p, ux_eqn, uy_eqn, uz_eqn, p_eqn, turbulence, config, backend ; resume, pref)
    
    # Extract model variables and configuration
    (;mesh, U, p, nu) = model
    # ux_model, uy_model = ux_eqn.model, uy_eqn.model
    p_model = p_eqn.model
    (; solvers, schemes, runtime) = config
    (; iterations, write_interval) = runtime
    
    mdotf = get_flux(ux_eqn, 2)
    nueff = get_flux(ux_eqn, 3)
    rDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT)
    S2 = ScalarField(mesh)

    # Temp sources to test GradUT explicit source
    # divUTx = zeros(Float64, length(mesh.cells))
    # divUTy = zeros(Float64, length(mesh.cells))

    n_cells = length(mesh.cells)
    Uf = FaceVectorField(mesh)
    pf = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)

    # Pre-allocate auxiliary variables

    # Consider using allocate from KernelAbstractions 
    # e.g. allocate(backend, Float32, res, res)
    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)
    prev = _convert_array!(prev, backend)  

    # Pre-allocate vectors to hold residuals 

    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, U.BCs)
    flux!(mdotf, Uf)
    grad!(∇p, pf, p, p.BCs)

    update_nueff!(nueff, nu, turbulence)

    CUDA.@allowscalar nbfaces = mesh.boundaries[end].IDs_range[end]
    nfaces = length(mesh.faces)

    @info "Staring PISO loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations

        @. prev = U.x.values
        discretise!(ux_eqn, prev, runtime, nfaces, nbfaces)
        apply_boundary_conditions!(ux_eqn, U.x.BCs)
        # ux_eqn.b .-= divUTx
        # implicit_relaxation!(ux_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(ux_eqn.preconditioner, mesh)
        run!(ux_eqn, solvers.U, U.x) #opP=Pu.P, solver=solver_U)
        residual!(R_ux, ux_eqn.equation, U.x, iteration)

        @. prev = U.y.values
        discretise!(uy_eqn, prev, runtime, nfaces, nbfaces)
        apply_boundary_conditions!(uy_eqn, U.y.BCs)
        # uy_eqn.b .-= divUTy
        # implicit_relaxation!(uy_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(uy_eqn.preconditioner, mesh)
        run!(uy_eqn, solvers.U, U.y)
        residual!(R_uy, uy_eqn.equation, U.y, iteration)

        if typeof(mesh) <: Mesh3
            @. prev = U.z.values
            discretise!(uz_eqn, prev, runtime, nfaces, nbfaces)
            apply_boundary_conditions!(uz_eqn, U.z.BCs)
            # uy_eqn.b .-= divUTy
            # implicit_relaxation!(uz_eqn, prev, solvers.U.relax, mesh)
            update_preconditioner!(uz_eqn.preconditioner, mesh)
            run!(uz_eqn, solvers.U, U.z)
            residual!(R_uz, uz_eqn.equation, U.z, iteration)
        end
        
        inverse_diagonal!(rD, ux_eqn)
        interpolate!(rDf, rD)
        remove_pressure_source!(ux_eqn, uy_eqn, uz_eqn, ∇p)

        for i ∈ 1:2
        H!(Hv, U, ux_eqn, uy_eqn, uz_eqn)

        interpolate!(Uf, Hv) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs)
        div!(divHv, Uf)
   
        @. prev = p.values
        discretise!(p_eqn, prev, runtime, nfaces, nbfaces)
        apply_boundary_conditions!(p_eqn, p.BCs)
        setReference!(p_eqn, pref, 1)
        update_preconditioner!(p_eqn.preconditioner, mesh)
        run!(p_eqn, solvers.p, p)

        explicit_relaxation!(p, prev, solvers.p.relax)
        residual!(R_p, p_eqn.equation, p, iteration)

        grad!(∇p, pf, p, p.BCs) 

        correct = false
        if correct
            ncorrectors = 1
            for i ∈ 1:ncorrectors
                discretise!(p_eqn, nfaces, nbfaces)
                apply_boundary_conditions!(p_eqn, p.BCs)
                setReference!(p_eqn.equation, pref, 1)
                # grad!(∇p, pf, p, pBCs) 
                interpolate!(gradpf, ∇p, p)
                nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
                correct!(p_eqn.equation, p_model.terms.term1, pf)
                run!(p_model, solvers.p)
                grad!(∇p, pf, p, pBCs) 
            end
        end

        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U)
        correct_boundaries!(Uf, U, U.BCs)
        flux!(mdotf, Uf)

        
        if isturbulent(model)
            grad!(gradU, Uf, U, U.BCs)
            turbulence!(turbulence, model, S, S2, prev) 
            update_nueff!(nueff, nu, turbulence)
        end

    end # corrector loop end
        
        # for i ∈ eachindex(divUTx)
        #     vol = mesh.cells[i].volume
        #     divUTx = -sqrt(2)*(nuf[i] + νt[i])*(gradUT[i][1,1]+ gradUT[i][1,2] + gradUT[i][1,3])*vol
        #     divUTy = -sqrt(2)*(nuf[i] + νt[i])*(gradUT[i][2,1]+ gradUT[i][2,2] + gradUT[i][2,3])*vol
        # end
        
        # convergence = 1e-7

        # if (R_ux[iteration] <= convergence && 
        #     R_uy[iteration] <= convergence && 
        #     R_p[iteration] <= convergence)

        #     print(
        #         """
        #         \n\n\n\n\n
        #         Simulation converged! $iteration iterations in
        #         """)
        #         if !signbit(write_interval)
        #             model2vtk(model, @sprintf "timestep_%.6d" iteration)
        #         end
        #     break
        # end

        # co = courant_number(U, mesh, runtime) # MUST IMPLEMENT!!!!!!

        ProgressMeter.next!(
            progress, showvalues = [
                (:time,iteration*runtime.dt),
                # (:Courant,co),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:p, R_p[iteration]),
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0
            model2vtk(model, @sprintf "timestep_%.6d" iteration)
        end

    end # end for loop
    return R_ux, R_uy, R_uz, R_p, model
end