# TO DO: These functions needs to be organised in a more sensible manner
function bound!(field, config)
    # Extract hardware configuration
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; values, mesh) = field
    (; cells, cell_neighbours) = mesh

    # set up and launch kernel
    kernel! = _bound!(backend, workgroup)
    kernel!(values, cells, cell_neighbours, ndrange = length(values))
    KernelAbstractions.synchronize(backend)
end

@kernel function _bound!(values, cells, cell_neighbours)
    i = @index(Global)

    sum_flux = 0.0
    sum_area = 0
    average = 0.0
    @uniform mzero = eps(eltype(values)) # machine zero

    @inbounds begin
        for fi ∈ cells[i].faces_range
            cID = cell_neighbours[fi]
            sum_flux += max(values[cID], mzero) # bounded sum
            sum_area += 1
        end
        average = sum_flux/sum_area

        values[i] = max(
            max(
                values[i],
                average*signbit(values[i])
            ),
            mzero
        )
    end
end

y_plus_laminar(E, kappa) = begin
    yL = 11.0; for i ∈ 1:10; yL = log(max(yL*E, 1.0))/kappa; end
    yL
end

ω_vis(nu, y, beta1) = 6*nu/(beta1*y^2)

ω_log(k, y, cmu, kappa) = sqrt(k)/(cmu^0.25*kappa*y)

y_plus(k, nu, y, cmu) = cmu^0.25*y*sqrt(k)/nu

sngrad(Ui, Uw, delta, normal) = begin
    Udiff = (Ui - Uw)
    Up = Udiff - (Udiff⋅normal)*normal # parallel velocity difference
    grad = Up/delta
    return grad
end

mag(vector) = sqrt(vector[1]^2 + vector[2]^2 + vector[3]^2) 

nut_wall(nu, yplus, kappa, E::T) where T = begin
    max(nu*(yplus*kappa/log(max(E*yplus, 1.0 + 1e-4)) - 1), zero(T))
end

@generated constrain_equation!(eqn, fieldBCs, model, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction
            call = quote
                constrain!(eqn, fieldBCs[$i], model, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function constrain!(eqn, BC, model, config)

    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Access equation data and deconstruct sparse array
    A = _A(eqn)
    b = _b(eqn, nothing)
    rowval = _rowval(A)
    colptr = _colptr(A)
    nzval = _nzval(A)
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundaries, boundary_cellsID) = mesh

    fluid = model.fluid 
    # turbFields = model.turbulence.fields
    turbulence = model.turbulence

    facesID_range = get_boundaries(BC, boundaries)
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _constrain!(backend, workgroup)
    kernel!(
        turbulence, fluid, BC, faces, start_ID, boundary_cellsID, rowval, colptr, nzval, b, ndrange=length(facesID_range)
    )
end

@kernel function _constrain!(turbulence, fluid, BC, faces, start_ID, boundary_cellsID, rowval, colptr, nzval, b)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    @uniform begin
        # nu = _nu(model.fluid)
        # k = model.turbulence.k
        nu = _nu(fluid)
        k = turbulence.k
        (; kappa, beta1, cmu, B, E) = BC.value
        ylam = y_plus_laminar(E, kappa)
    end
    ωc = zero(eltype(nzval))
    
    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu) 

        if yplus > ylam 
            ωc = ωlog
        else
            ωc = ωvis
        end
        # Line below is weird but worked
        # b[cID] = A[cID,cID]*ωc

        
        # Classic approach
        # b[cID] += A[cID,cID]*ωc
        # A[cID,cID] += A[cID,cID]
        
        nzIndex = spindex(colptr, rowval, cID, cID)
        Atomix.@atomic b[cID] += nzval[nzIndex]*ωc
        Atomix.@atomic nzval[nzIndex] += nzval[nzIndex] 
    end
end

@generated constrain_boundary!(field, fieldBCs, model, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction
            call = quote
                set_cell_value!(field, fieldBCs[$i], model, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function set_cell_value!(field, BC, model, config)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundaries, boundary_cellsID) = mesh
    (; fluid, turbulence) = model
    # turbFields = turbulence.fields

    facesID_range = get_boundaries(BC, boundaries)
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _set_cell_value!(backend, workgroup)
    kernel!(
        field, turbulence, fluid, BC, faces, start_ID, boundary_cellsID, ndrange=length(facesID_range)
    )
end

@kernel function _set_cell_value!(field, turbulence, fluid, BC, faces, start_ID, boundary_cellsID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    @uniform begin
        nu = _nu(fluid)
        (; k) = turbulence
        # k= _k(turbulence)
        (; kappa, beta1, cmu, B, E) = BC.value
        (; values) = field
        ylam = y_plus_laminar(E, kappa)
    end
    ωc = zero(eltype(values))

    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu) 

        if yplus > ylam 
            ωc = ωlog
        else
            ωc = ωvis
        end

        values[cID] = ωc # needs to be atomic?
    end
end

@generated correct_production!(P, fieldBCs, model, gradU, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: KWallFunction
            call = quote
                set_production!(P, fieldBCs[$i], model, gradU, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function set_production!(P, BC, model, gradU, config)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID, boundaries) = mesh

    facesID_range = get_boundaries(BC, boundaries)
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _set_production!(backend, workgroup)
    kernel!(
        P.values, BC, model, faces, boundary_cellsID, start_ID, gradU, ndrange=length(facesID_range)
    )
end

@kernel function _set_production!(values, BC, model, faces, boundary_cellsID, start_ID, gradU)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E) = BC.value
    (; U, nu) = model
    (; k, nut) = model.turbulence

    ylam = y_plus_laminar(E, kappa)
    # Uw = SVector{3,_get_float(mesh)}(0.0,0.0,0.0)
    Uw = SVector{3}(0.0,0.0,0.0)
        cID = boundary_cellsID[fID]
        face = faces[fID]
        nuc = nu[cID]
        (; delta, normal)= face
        uStar = cmu^0.25*sqrt(k[cID])
        dUdy = uStar/(kappa*delta)
        yplus = y_plus(k[cID], nuc, delta, cmu)
        nutw = nut_wall(nuc, yplus, kappa, E)
        # mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
        mag_grad_U = mag(gradU[cID]*normal)
        if yplus > ylam
            values[cID] = (nu[cID] + nutw)*mag_grad_U*dUdy
        else
            values[cID] = 0.0
        end
end

@generated correct_eddy_viscosity!(νtf, nutBCs, model, config) = begin
    BCs = nutBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: NutWallFunction
            call = quote
                correct_nut_wall!(νtf, nutBCs[$i], model, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function correct_nut_wall!(νtf, BC, model, config)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.mesh
    (; faces, boundary_cellsID, boundaries) = mesh

    facesID_range = get_boundaries(BC, boundaries)
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _correct_nut_wall!(backend, workgroup)
    kernel!(
        νtf.values, model, BC, faces, boundary_cellsID, start_ID, ndrange=length(facesID_range)
    )
end

@kernel function _correct_nut_wall!(values, model, BC, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E) = BC.value
    (; nu) = model
    (; k) = model.turbulence
    
    ylam = y_plus_laminar(E, kappa)
        cID = boundary_cellsID[fID]
        face = faces[fID]
        # nuf = nu[fID]
        (; delta)= face
        # yplus = y_plus(k[cID], nuf, delta, cmu)
        nuc = nu[cID]
        yplus = y_plus(k[cID], nuc, delta, cmu)
        nutw = nut_wall(nuc, yplus, kappa, E)
        if yplus > ylam
            values[fID] = nutw
        else
            values[fID] = 0.0
        end
end