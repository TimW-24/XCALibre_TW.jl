export piso_comp!

piso_comp!(model_in, config; resume=true, pref=nothing) = begin

    R_ux, R_uy, R_uz, R_p, R_e, model = setup_unsteady_compressible_solvers(
        CPISO, model_in, config;
        resume=true, pref=nothing
        )
        
    return R_ux, R_uy, R_uz, R_p, model
end

# Setup for all compressible algorithms
function setup_unsteady_compressible_solvers(
    solver_variant, 
    model_in, config; resume=true, pref=nothing
    ) 

    (; solvers, schemes, runtime, hardware) = config

    @info "Extracting configuration and input fields..."

    model = adapt(hardware.backend, model_in)
    (; U, p) = model.momentum
    (; rho) = model.fluid
    mesh = model.domain

    @info "Preallocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rhorDf = FaceScalarField(mesh)
    mueff = FaceScalarField(mesh)
    mueffgradUt = VectorField(mesh)
    # initialise!(rDf, 1.0)
    rhorDf.values .= 1.0
    divHv = ScalarField(mesh)
    ddtrho = ScalarField(mesh)
    psidpdt = ScalarField(mesh)
    divmdotf = ScalarField(mesh)
    psi = ScalarField(mesh)


    @info "Defining models..."

    # rho eqn doesn't work at the moment.
    # rho_eqn = (
    #     Time{schemes.rho.time}(rho) 
    #     == 
    #     -Source(divmdotf)
    # ) → ScalarEquation(mesh)

    U_eqn = (
        Time{schemes.U.time}(rho, U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(mueff, U) 
        == 
        -Source(∇p.result)
        +Source(mueffgradUt)
    ) → VectorEquation(mesh)

    if typeof(model.fluid) <: WeaklyCompressible
        # p_eqn = (
        #     Time{schemes.p.time}(psi, p)  # correction(fvm::ddt(p)) means d(p)/d(t) - d(pold)/d(t)
        #     - Laplacian{schemes.p.laplacian}(rhorDf, p)
        #     ==
        #     -Source(divHv)
        #     -Source(ddtrho)
        #     +Source(psidpdt)
        # ) → ScalarEquation(mesh)
        p_eqn = (
            Time{schemes.p.time}(psi, p)  # correction(fvm::ddt(p)) means d(p)/d(t) - d(pold)/d(t)
            - Laplacian{schemes.p.laplacian}(rhorDf, p)
            ==
            -Source(divHv)
        ) → ScalarEquation(mesh)
    elseif typeof(model.fluid) <: Compressible
        pconv = FaceScalarField(mesh)
        p_eqn = (
            Time{schemes.p.time}(psi, p)
            - Laplacian{schemes.p.laplacian}(rhorDf, p) 
            + Divergence{schemes.p.divergence}(pconv, p)
            ==
            -Source(divHv)
            -Source(ddtrho) # Needs to capture the correction part of dPdT and the explicit drhodt
        ) → ScalarEquation(mesh)
    end

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, U_eqn, U.BCs, config)
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, config)
    # @reset rho_eqn.preconditioner = set_preconditioner(
    #                 solvers.rho.preconditioner, rho_eqn, p.BCs, config)

    @info "Pre-allocating solvers..."
     
    @reset U_eqn.solver = solvers.U.solver(_A(U_eqn), _b(U_eqn, XDir()))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))
    # @reset rho_eqn.solver = solvers.rho.solver(_A(rho_eqn), _b(rho_eqn))
  
    @info "Initialising energy model..."
    energyModel = Energy.initialise(model.energy, model, mdotf, rho, p_eqn, config)

    @info "Initialising turbulence model..."
    turbulenceModel = Turbulence.initialise(model.turbulence, model, mdotf, p_eqn, config)

    R_ux, R_uy, R_uz, R_p, R_e, model  = solver_variant(
    model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config; resume=resume, pref=pref)

    return R_ux, R_uy, R_uz, R_p, R_e, model    
end # end function

function CPISO(
    model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config; resume=resume, pref=pref)
    
    # Extract model variables and configuration
    (; U, p) = model.momentum
    (; rho, rhof, nu) = model.fluid
    (; dpdt) = model.energy
    mesh = model.domain
    p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware
    
    # divmdotf = get_source(rho_eqn, 1)
    mdotf = get_flux(U_eqn, 2)
    mueff = get_flux(U_eqn, 3)
    mueffgradUt = get_source(U_eqn, 2)
    rhorDf = get_flux(p_eqn, 2)
    divHv = get_source(p_eqn, 1)
    # ddtrho = get_source(p_eqn, 2)
    # psidpdt = get_source(p_eqn, 3)

    @info "Initialise VTKWriter (Store mesh in host memory)"

    VTKMeshData = initialise_writer(model.domain)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT)
    S2 = ScalarField(mesh)

    n_cells = length(mesh.cells)
    n_faces = length(mesh.faces)
    Uf = FaceVectorField(mesh)
    pf = FaceScalarField(mesh)
    nueff = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    Psi = ScalarField(mesh)
    Psif = FaceScalarField(mesh)

    divmdotf = ScalarField(mesh)

    mugradUTx = FaceScalarField(mesh)
    mugradUTy = FaceScalarField(mesh)
    mugradUTz = FaceScalarField(mesh)

    divmugradUTx = ScalarField(mesh)
    divmugradUTy = ScalarField(mesh)
    divmugradUTz = ScalarField(mesh)

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)
    prev = _convert_array!(prev, backend) 

    corr = zeros(TF, n_faces)
    corr = _convert_array!(corr, backend) 

    # Pre-allocate vectors to hold residuals 
    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    
    # Initial calculations
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, U.BCs, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, p.BCs, config)
    thermo_Psi!(model, Psi); thermo_Psi!(model, Psif, config);
    @. rho.values = Psi.values * p.values
    @. rhof.values = Psif.values * pf.values
    flux!(mdotf, Uf, rhof, config)

    # grad limiter test
    limit_gradient!(∇p, p, config)

    update_nueff!(nueff, nu, model.turbulence, config)
    @. mueff.values = rhof.values*nueff.values

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Staring PISO loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    volumes = getproperty.(mesh.cells, :volume)

    @time for iteration ∈ 1:iterations

        println("Max. CFL : ", maximum((U.x.values.^2+U.y.values.^2).^0.5*runtime.dt./volumes.^(1/3)))

        ## CHECK GRADU AND EXPLICIT STRESSES
        grad!(gradU, Uf, U, U.BCs, config)
        
        # Set up and solve momentum equations
        explicit_shear_stress!(mugradUTx, mugradUTy, mugradUTz, mueff, gradU)
        div!(divmugradUTx, mugradUTx, config)
        div!(divmugradUTy, mugradUTy, config)
        div!(divmugradUTz, mugradUTz, config)

        @. mueffgradUt.x.values = divmugradUTx.values
        @. mueffgradUt.y.values = divmugradUTy.values
        @. mueffgradUt.z.values = divmugradUTz.values

        solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config)

        energy!(energyModel, model, prev, mdotf, rho, mueff, config)


        thermo_Psi!(model, Psi); thermo_Psi!(model, Psif, config);

        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rhorDf, rD, config)
        @. rhorDf.values *= rhof.values

        remove_pressure_source!(U_eqn, ∇p, config)
        
        for i ∈ 1:10
            H!(Hv, U, U_eqn, config)
            
            # Interpolate faces
            interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
            correct_boundaries!(Uf, Hv, U.BCs, config)

            if typeof(model.fluid) <: Compressible
                flux!(pconv, Uf, config)
                @. pconv.values *= Psif.values
                corr = 0.0#fvc::ddtCorr(rho, U, phi)
                flux!(mdotf, Uf, config)
                @. mdotf.values *= rhof.values
                @. mdotf.values += rhorDf.values*corr
                interpolate!(pf, p, config)
                correct_boundaries!(pf, p, p.BCs, config)
                @. mdotf.values -= mdotf.values*Psif.values*pf.values/rhof.values
                div!(divHv, mdotf, config)

            elseif typeof(model.fluid) <: WeaklyCompressible
                @. corr = mdotf.values
                flux!(mdotf, Uf, config)
                @. mdotf.values *= rhof.values
                @. corr -= mdotf.values
                @. corr *= 0.0/runtime.dt
                @. mdotf.values += rhorDf.values*corr/rhof.values
                div!(divHv, mdotf, config)
            end
            
            # Pressure calculations
            @. prev = p.values
            solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)

            # Gradient
            grad!(∇p, pf, p, p.BCs, config) 

            # grad limiter test
            limit_gradient!(∇p, p, config)

            correct = false
            if correct
                ncorrectors = 1
                for i ∈ 1:ncorrectors
                    discretise!(p_eqn)
                    apply_boundary_conditions!(p_eqn, p.BCs)
                    setReference!(p_eqn.equation, pref, 1)
                    interpolate!(gradpf, ∇p, p)
                    nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
                    correct!(p_eqn.equation, p_model.terms.term1, pf)
                    solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)
                    grad!(∇p, pf, p, pBCs) 
                end
            end

            explicit_relaxation!(p, prev, solvers.p.relax, config)

            if ~isempty(solvers.p.limit)
                pmin = solvers.p.limit[1]; pmax = solvers.p.limit[2]
                clamp!(p.values, pmin, pmax)
            end

            pgrad = face_normal_gradient(p, pf)
        
            if typeof(model.fluid) <: Compressible
                @. mdotf.values += (pconv.values*(pf.values) - pgrad.values*rhorDf.values)  
            elseif typeof(model.fluid) <: WeaklyCompressible
                @. mdotf.values -= pgrad.values*rhorDf.values
            end
   
            @. rho.values = max.(Psi.values * p.values, 0.001)
            @. rhof.values = max.(Psif.values * pf.values, 0.001)

            # Velocity and boundaries correction
            correct_velocity!(U, Hv, ∇p, rD, config)
            interpolate!(Uf, U, config)
            correct_boundaries!(Uf, U, U.BCs, config)
            
            @. dpdt.values = (p.values-prev)/runtime.dt

            grad!(gradU, Uf, U, U.BCs, config)
            turbulence!(turbulenceModel, model, S, S2, prev, config) 
            update_nueff!(nueff, nu, model.turbulence, config)
        end # corrector loop end

    residual!(R_ux, U_eqn, U.x, iteration, xdir, config)
    residual!(R_uy, U_eqn, U.y, iteration, ydir, config)
    if typeof(mesh) <: Mesh3
        residual!(R_uz, U_eqn, U.z, iteration, zdir, config)
    end
    residual!(R_p, p_eqn, p, iteration, nothing, config)
        
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
            model2vtk(model, VTKMeshData, @sprintf "timestep_%.6d" iteration)
        end

    end # end for loop
    model_out = adapt(CPU(), model)
    return R_ux, R_uy, R_uz, R_p, model_out
end

function limit_gradient!(∇F, F, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    minPhi0 = maximum(F.values) # use min value so all values compared are larger
    maxPhi0 = minimum(F.values)

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(∇F, F, cells, cell_neighbours, cell_faces, cell_nsign, faces, minPhi0, maxPhi0, ndrange=length(cells))
    KernelAbstractions.synchronize(backend)
end

@kernel function _limit_gradient!(∇F, F, cells, cell_neighbours, cell_faces, cell_nsign, faces, minPhi, maxPhi)
    cID = @index(Global)
    # mesh = F.mesh
    # (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    # minPhi0 = maximum(F.values) # use min value so all values compared are larger
    # maxPhi0 = minimum(F.values)

    # for (cID, cell) ∈ enumerate(cells)
        cell = cells[cID]
        # minPhi = minPhi0 # reset for next cell
        # maxPhi = maxPhi0

        # find min and max values around cell
        faces_range = cell.faces_range
        
        phiP = F[cID]
        # minPhi = phiP # reset for next cell
        # maxPhi = phiP
        for fi ∈ faces_range
            nID = cell_neighbours[fi]
            phiN = F[nID]
            maxPhi = max(phiN, maxPhi)
            minPhi = min(phiN, minPhi)
        end

        g0 = ∇F[cID]
        cc = cell.centre

        for fi ∈ faces_range 
            fID = cell_faces[fi]
            face = faces[fID]
            nID = face.ownerCells[2]
            # phiN = F[nID]
            normal = face.normal
            nsign = cell_nsign[fi]
            na = nsign*normal

            fc = face.centre 
            cc_fc = fc - cc
            n0 = cc_fc/norm(cc_fc)
            gn = g0⋅n0
            δϕ = g0⋅cc_fc
            gτ = g0 - gn*n0
            if (maxPhi > phiP) && (δϕ > maxPhi - phiP)
                g0 = gτ + na*(maxPhi - phiP)
            elseif (minPhi < phiP) && (δϕ < minPhi - phiP)
                g0 = gτ + na*(minPhi - phiP)
            end            
        end
        ∇F.result.x.values[cID] = g0[1]
        ∇F.result.y.values[cID] = g0[2]
        ∇F.result.z.values[cID] = g0[3]
    # end
end

function explicit_shear_stress!(mugradUTx::FaceScalarField, mugradUTy::FaceScalarField, mugradUTz::FaceScalarField, mueff, gradU)
    mesh = mugradUTx.mesh
    (; faces, cells) = mesh
    nbfaces = length(mesh.boundary_cellsID) #boundary_faces(mesh)
    start_faceID = nbfaces + 1
    last_faceID = length(faces)
    for fID ∈ start_faceID:last_faceID
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        cell1 = cells[cID1]
        cell2 = cells[cID2]
        gradUxxf = 0.5*(gradU.result.xx[cID1]+gradU.result.xx[cID2])
        gradUxyf = 0.5*(gradU.result.xy[cID1]+gradU.result.xy[cID2])
        gradUxzf = 0.5*(gradU.result.xz[cID1]+gradU.result.xz[cID2])
        gradUyxf = 0.5*(gradU.result.yx[cID1]+gradU.result.yx[cID2])
        gradUyyf = 0.5*(gradU.result.yy[cID1]+gradU.result.yy[cID2])
        gradUyzf = 0.5*(gradU.result.yz[cID1]+gradU.result.yz[cID2])
        gradUzxf = 0.5*(gradU.result.zx[cID1]+gradU.result.zx[cID2])
        gradUzyf = 0.5*(gradU.result.zy[cID1]+gradU.result.zy[cID2])
        gradUzzf = 0.5*(gradU.result.zz[cID1]+gradU.result.zz[cID2])
        mugradUTx[fID] = mueff[fID] * (normal[1]*gradUxxf + normal[2]*gradUyxf + normal[3]*gradUzxf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
        mugradUTy[fID] = mueff[fID] * (normal[1]*gradUxyf + normal[2]*gradUyyf + normal[3]*gradUzyf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
        mugradUTz[fID] = mueff[fID] * (normal[1]*gradUxzf + normal[2]*gradUyzf + normal[3]*gradUzzf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
        # mugradUTx[fID] = mueff[fID] * (normal[1]*gradUxxf + normal[2]*gradUyxf + normal[3]*gradUzxf) * area
        # mugradUTy[fID] = mueff[fID] * (normal[1]*gradUxyf + normal[2]*gradUyyf + normal[3]*gradUzyf) * area
        # mugradUTz[fID] = mueff[fID] * (normal[1]*gradUxzf + normal[2]*gradUyzf + normal[3]*gradUzzf) * area
    end
    
    # Now deal with boundary faces
    for fID ∈ 1:nbfaces
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        cell1 = cells[cID1]
        cell2 = cells[cID2]
        gradUxxf = (gradU.result.xx[cID1])
        gradUxyf = (gradU.result.xy[cID1])
        gradUxzf = (gradU.result.xz[cID1])
        gradUyxf = (gradU.result.yx[cID1])
        gradUyyf = (gradU.result.yy[cID1])
        gradUyzf = (gradU.result.yz[cID1])
        gradUzxf = (gradU.result.zx[cID1])
        gradUzyf = (gradU.result.zy[cID1])
        gradUzzf = (gradU.result.zz[cID1])
        mugradUTx[fID] = mueff[fID] * (normal[1]*gradUxxf + normal[2]*gradUyxf + normal[3]*gradUzxf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
        mugradUTy[fID] = mueff[fID] * (normal[1]*gradUxyf + normal[2]*gradUyyf + normal[3]*gradUzyf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
        mugradUTz[fID] = mueff[fID] * (normal[1]*gradUxzf + normal[2]*gradUyzf + normal[3]*gradUzzf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
    end
end 