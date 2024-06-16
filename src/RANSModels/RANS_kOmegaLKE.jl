export KOmegaLKE

# Model type definition (hold fields)
struct KOmegaLKE{S1,S2,S3,S4,F1,F2,F3,F4,C1,C2,Y} <: AbstractTurbulenceModel 
    k::S1
    omega::S2
    kl::S3
    nut::S4
    kf::F1
    omegaf::F2
    klf::F3
    nutf::F4
    coeffs::C1
    Tu::C2
    y::Y
end 
Adapt.@adapt_structure KOmegaLKE

# Model type definition (hold equation definitions and internal data)
struct KOmegaLKEModel{
    E1,E2,E3,F1,F2,F3,S1,S2,S3,S4,S5,S6,S7,V1,V2} <: AbstractTurbulenceModel
    k_eqn::E1
    ω_eqn::E2
    kl_eqn::E3
    nueffkLS::F1
    nueffkS::F2
    nueffωS::F3
    nuL::S1
    nuts::S2
    Ω::S3
    γ::S4
    fv::S5
    normU::S6
    Reυ::S7
    ∇k::V1
    ∇ω::V2
end 
Adapt.@adapt_structure KOmegaLKEModel

# Model API constructor
RANS{KOmegaLKE}(; Tu, walls) = begin
    args = (Tu=Tu, walls=walls)
    ARG = typeof(args)
    RANS{KOmegaLKE,ARG}(args)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(rans::RANS{KOmegaLKE, ARG})(mesh) where ARG = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    kl = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    klf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = (
        C1 = 0.02974,
        C2 = 59.79,
        C3 = 1.191,
        C4 = 1.65*10^-13,
        Cμ = 0.09,
        Cω1 = 0.52,
        Cω2 = 0.0708,
        Ccrit = 76500,
        CSS = 1.45,
        Cv = 0.43,
        σk = 0.5,
        σd = 0.125,
        σkL = 0.0125,
        σω = 0.5
    )
    Tu = rans.args.Tu

    # Allocate wall distance "y" and setup boundary conditions
    y = ScalarField(mesh)
    walls = rans.args.walls
    BCs = []
    for boundary ∈ mesh.boundaries
        for namedwall ∈ walls
            if boundary.name == namedwall
                push!(BCs, Dirichlet(boundary.name, 0.0))
            else
                push!(BCs, Neumann(boundary.name, 0.0))
            end
        end
    end
    y = assign(y, BCs...)

    KOmegaLKE(k, omega, kl, nut, kf, omegaf, klf, nutf, coeffs, Tu, y)
end

# Model initialisation
function initialise(
    turbulence::KOmegaLKE, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}

    @info "Initialising k-ω LKE model..."

    # unpack turbulent quantities and configuration
    (; k, omega, kl, y, kf, omegaf, klf) = model.turbulence
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    nueffkLS = ScalarField(mesh)
    nueffkS = ScalarField(mesh)
    nueffωS = ScalarField(mesh)
    nueffkL = FaceScalarField(mesh)
    nueffk = FaceScalarField(mesh)
    nueffω = FaceScalarField(mesh)
    DkLf = ScalarField(mesh)
    Dkf = ScalarField(mesh)
    Dωf = ScalarField(mesh)
    PkL = ScalarField(mesh)
    Pk = ScalarField(mesh)
    Pω = ScalarField(mesh)
    dkdomegadx = ScalarField(mesh)
    normU = ScalarField(mesh)
    Reυ = ScalarField(mesh)
    nuL = ScalarField(mesh)
    nuts = ScalarField(mesh)
    Ω = ScalarField(mesh)
    γ = ScalarField(mesh)
    fv = ScalarField(mesh)
    ∇k = Grad{schemes.k.gradient}(k)
    ∇ω = Grad{schemes.p.gradient}(omega)

    kl_eqn = (
            Time{schemes.kl.time}(kl)
            + Divergence{schemes.kl.divergence}(mdotf, kl) 
            - Laplacian{schemes.kl.laplacian}(nueffkL, kl) 
            + Si(DkLf,kl) # DkLf = 2*nu/y^2
            ==
            Source(PkL)
        ) → eqn

    k_eqn = (
            Time{schemes.k.time}(k)
            + Divergence{schemes.k.divergence}(mdotf, k) 
            - Laplacian{schemes.k.laplacian}(nueffk, k) 
            + Si(Dkf,k) # Dkf = β⁺*omega
            ==
            Source(Pk)
        ) → eqn
    
    ω_eqn = (
            Time{schemes.omega.time}(omega)
            + Divergence{schemes.omega.divergence}(mdotf, omega) 
            - Laplacian{schemes.omega.laplacian}(nueffω, omega)
            + Si(Dωf,omega)  # Dωf = β1*omega
            ==
            Source(Pω)
            + Source(dkdomegadx)
    ) → eqn

    
    # Set up preconditioners

    @reset kl_eqn.preconditioner = set_preconditioner(
                solvers.kl.preconditioner, kl_eqn, kl.BCs, config)

    @reset k_eqn.preconditioner = set_preconditioner(
                solvers.k.preconditioner, k_eqn, k.BCs, config)

    @reset ω_eqn.preconditioner = set_preconditioner(
                solvers.omega.preconditioner, ω_eqn, omega.BCs, config)
    
    # preallocating solvers

    @reset kl_eqn.solver = solvers.kl.solver(_A(kl_eqn), _b(kl_eqn))
    @reset k_eqn.solver = solvers.k.solver(_A(k_eqn), _b(k_eqn))
    @reset ω_eqn.solver = solvers.omega.solver(_A(ω_eqn), _b(ω_eqn))

    grad!(∇ω, omegaf, omega, omega.BCs, config)
    grad!(∇k, kf, k, k.BCs, config)

    # float_type = _get_float(mesh)
    # coeffs = get_LKE_coeffs(float_type)

    # Wall distance calculation
    # y.values .= wall_distance(model, config)
    wall_distance!(model, config)

    return KOmegaLKEModel(
        k_eqn,
        ω_eqn,
        kl_eqn,
        nueffkLS,
        nueffkS,
        nueffωS,
        nuL,
        nuts,
        Ω,
        γ,
        fv,
        normU,
        Reυ,
        ∇k,
        ∇ω
    )
end

# Model solver call (implementation)
function turbulence!(rans::KOmegaLKEModel, model::Physics{T,F,M,Turb,E,D,BI}, S, S2, prev, config
    ) where {T,F,M,Turb<:KOmegaLKE,E,D,BI}
    mesh = model.domain
    (; momentum, turbulence) = model
    U = momentum.U
    (; k, omega, kl, nut, y, kf, omegaf, klf, nutf, coeffs, Tu) = turbulence
    nu = _nu(model.fluid)
    
    (; k_eqn, ω_eqn, kl_eqn, nueffkLS, nueffkS, nueffωS, nuL, nuts, Ω, γ, fv, ∇k, ∇ω, normU, Reυ) = rans
    (; solvers, runtime) = config

    nueffkL = get_flux(kl_eqn, 3)
    DkLf = get_flux(kl_eqn, 4)
    PkL = get_source(kl_eqn, 1)

    nueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    nueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)
    dkdomegadx = get_source(ω_eqn, 2) # cross diffusion term

    magnitude2!(Pk, S, config, scale_factor=2.0) # this is S^2

    # Update kl fluxes and terms

    # for i ∈ eachindex(normU.values)
    #     normU.values[i] = norm(U[i])
    # end
    magnitude!(normU, U, config)

    ReLambda = @. normU.values*y.values/nu.values;
    @. Reυ.values = (2*nu.values^2*kl.values/(y.values^2))^0.25*y.values/nu.values;
    η = coeffs.C1*tanh(coeffs.C2*(Tu^coeffs.C3)+coeffs.C4)
    @. PkL.values = sqrt(Pk.values)*η*kl.values*Reυ.values^(-1.30)*ReLambda^(0.5)
    @. DkLf.values = (2*nu.values)/(y.values^2)
    @. nueffkLS.values = nu.values+(coeffs.σkL*sqrt(kl.values)*y.values)
    interpolate!(nueffkL, nueffkLS, config)
    correct_boundaries!(nueffkL, nueffkLS, nut.BCs, config)

    # Solve kl equation
    prev .= kl.values
    discretise!(kl_eqn, prev, config)
    apply_boundary_conditions!(kl_eqn, kl.BCs, nothing, config)
    implicit_relaxation!(kl_eqn, kl.values, solvers.kl.relax, nothing, config)
    update_preconditioner!(kl_eqn.preconditioner, mesh, config)
    solve_system!(kl_eqn, solvers.kl, kl, nothing, config)
    bound!(kl, config)

    #Damping and trigger
    @. fv.values = 1-exp(-sqrt(k.values/(nu.values*omega.values))/coeffs.Cv)
    @. γ.values = min((kl.values/(min(nu.values,nuL.values)*sqrt(S2.values)))^2,coeffs.Ccrit)/coeffs.Ccrit
    fSS = @. exp(-(coeffs.CSS*nu.values*sqrt(S2.values)/k.values)^2) # should be Ω but S works

    #Update ω fluxes
    # double_inner_product!(Pk, S, gradU) # multiplied by 2 (def of Sij) (Pk = S² at this point)
    interpolate!(kf, k, config)
    correct_boundaries!(nutf, k, k.BCs, config)
    interpolate!(omegaf, omega, config)
    correct_boundaries!(nutf, omega, omega.BCs, config)
    grad!(∇ω, omegaf, omega, omega.BCs, config)
    grad!(∇k, kf, k, k.BCs, config)
    inner_product!(dkdomegadx, ∇k, ∇ω, config)
    @. Pω.values = coeffs.Cω1 * Pk.values * nut.values * (omega.values / k.values)
    @. dkdomegadx.values = max((coeffs.σd / omega.values) * dkdomegadx.values, 0.0)
    @. Dωf.values = coeffs.Cω2 * omega.values
    # @. nueffωS.values = nu.values+(coeffs.σω*nut_turb.values*γ.values)
    @. nueffωS.values = nu.values+(coeffs.σω*(k.values/omega.values)*γ.values)
    interpolate!(nueffω, nueffωS, config)
    correct_boundaries!(nueffω, nueffωS, nut.BCs, config)

    #Update k fluxes
    @. Dkf.values = coeffs.Cμ*omega.values*γ.values
    @. nueffkS.values = nu.values+(coeffs.σk*(k.values/omega.values)*γ.values)
    interpolate!(nueffk, nueffkS, config)
    correct_boundaries!(nueffk, nueffkS, nut.BCs, config)
    @. Pk.values = nut.values*Pk.values*γ.values*fv.values
    correct_production!(Pk, k.BCs, model, S.gradU, config)

    # Solve omega equation
    prev .= omega.values
    discretise!(ω_eqn, prev, config)
    apply_boundary_conditions!(ω_eqn, omega.BCs, nothing, config)
    implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    constrain_equation!(ω_eqn, omega.BCs, model, config) # active with WFs only
    update_preconditioner!(ω_eqn.preconditioner, mesh, config)
    solve_system!(ω_eqn, solvers.omega, omega, nothing, config)
    constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only
    bound!(omega, config)

    # Solve k equation
    prev .= k.values
    discretise!(k_eqn, prev, config)
    apply_boundary_conditions!(k_eqn, k.BCs, nothing, config)
    implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    solve_system!(k_eqn, solvers.k, k, nothing, config)
    bound!(k, config)

    # grad!(∇ω, omegaf, omega, omega.BCs, config)
    # grad!(∇k, kf, k, k.BCs, config)

    # @. nut_turb.values = k.values/omega.values
    ReLambda = @. normU.values*y.values/nu.values
    @. Reυ.values = (2*nu.values^2*kl.values/(y.values^2))^0.25*y.values/nu.values;
    @. PkL.values = sqrt(Pk.values)*η*kl.values*Reυ.values^(-1.30)*ReLambda^(0.5) # update
    @. nuL.values = PkL.values/max(S2.values,(normU.values/y.values)^2)

    fSS = @. exp(-(coeffs.CSS*nu.values*sqrt(S2.values)/k.values)^2) # should be Ω but S works
    @. nuts.values = fSS*(k.values/omega.values)
    @. nut.values = nuts.values+nuL.values

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)
end

# Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, name) where {T,F,M,Tu<:KOmegaLKE,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p),
        ("k", model.turbulence.k),
        ("omega", model.turbulence.omega),
        ("kl", model.turbulence.kl),
        ("nut", model.turbulence.nut),
        ("y", model.turbulence.y)
    )
    write_vtk(name, model.domain, args...)
end