export KOmega

# Reference:
# Wilcox, D. C., Turbulence Modeling for CFD, 2nd edition, DCW Industries, Inc., La Canada CA, 1998

# Model type definition
struct KOmega{S1,S2,S3,F1,F2,F3,C} <: AbstractTurbulenceModel
    k::S1
    omega::S2
    nut::S3
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
end
Adapt.@adapt_structure KOmega

struct KOmegaModel{E1,E2}
    k_eqn::E1 
    ω_eqn::E2
end
Adapt.@adapt_structure KOmegaModel

# Model API constructor (pass user input as keyword arguments and process as needed)
RANS{KOmega}(; β⁺=0.09, α1=0.52, β1=0.072, σk=0.5, σω=0.5) = begin 
    coeffs = (β⁺=β⁺, α1=α1, β1=β1, σk=σk, σω=σω)
    ARG = typeof(coeffs)
    RANS{KOmega,ARG}(coeffs)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(rans::RANS{KOmega, ARG})(mesh) where ARG = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = rans.args
    KOmega(k, omega, nut, kf, omegaf, nutf, coeffs)
end

# Model initialisation
function initialise(
    turbulence::KOmega, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}

    (; k, omega, nut) = turbulence
    (; rho) = model.fluid
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    # define fluxes and sources
    mueffk = FaceScalarField(mesh)
    mueffω = FaceScalarField(mesh)
    Dkf = ScalarField(mesh)
    Dωf = ScalarField(mesh)
    Pk = ScalarField(mesh)
    Pω = ScalarField(mesh)
    
    k_eqn = (
            Time{schemes.k.time}(rho, k)
            + Divergence{schemes.k.divergence}(mdotf, k) 
            - Laplacian{schemes.k.laplacian}(mueffk, k) 
            + Si(Dkf,k) # Dkf = β⁺rho*omega
            ==
            Source(Pk)
        ) → eqn
    
    ω_eqn = (
            Time{schemes.omega.time}(rho, omega)
            + Divergence{schemes.omega.divergence}(mdotf, omega) 
            - Laplacian{schemes.omega.laplacian}(mueffω, omega) 
            + Si(Dωf,omega)  # Dωf = rho*β1*omega
            ==
            Source(Pω)
    ) → eqn

    # Set up preconditioners
    @reset k_eqn.preconditioner = set_preconditioner(
                solvers.k.preconditioner, k_eqn, k.BCs, config)

    # @reset ω_eqn.preconditioner = set_preconditioner(
    #             solvers.omega.preconditioner, ω_eqn, omega.BCs, config)

    @reset ω_eqn.preconditioner = k_eqn.preconditioner
    
    # preallocating solvers
    @reset k_eqn.solver = solvers.k.solver(_A(k_eqn), _b(k_eqn))
    @reset ω_eqn.solver = solvers.omega.solver(_A(ω_eqn), _b(ω_eqn))

    return KOmegaModel(k_eqn, ω_eqn)
end

# Model solver call (implementation)
function turbulence!(
    rans::KOmegaModel{E1,E2}, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time, config
    ) where {T,F,M,Tu<:KOmega,E,D,BI,E1,E2}

    mesh = model.domain
    
    (; rho, rhof, nu, nuf) = model.fluid
    (;k, omega, nut, kf, omegaf, nutf, coeffs) = model.turbulence
    (;k_eqn, ω_eqn) = rans
    (; solvers, runtime) = config

    mueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    mueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)

    # nu = _nu(model.fluid)

    # update fluxes and sources
    magnitude2!(Pk, S, config, scale_factor=2.0) # multiplied by 2 (def of Sij)
    constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only
    correct_production!(Pk, k.BCs, model, S.gradU, config)
    @. Pω.values = rho.values*coeffs.α1*Pk.values
    @. Pk.values = rho.values*nut.values*Pk.values
    @. Dωf.values = rho.values*coeffs.β1*omega.values
    @. mueffω.values = rhof.values * (nuf.values + coeffs.σω*nutf.values)
    @. Dkf.values = rho.values*coeffs.β⁺*omega.values
    @. mueffk.values = rhof.values * (nuf.values + coeffs.σk*nutf.values)

    # Solve omega equation
    prev .= omega.values
    discretise!(ω_eqn, omega, config)
    apply_boundary_conditions!(ω_eqn, omega.BCs, nothing, config)
    constrain_equation!(ω_eqn, omega.BCs, model, config) # active with WFs only
    implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    # implicit_relaxation_diagdom!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    update_preconditioner!(ω_eqn.preconditioner, mesh, config)
    solve_system!(ω_eqn, solvers.omega, omega, nothing, config)
   
    constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only
    bound!(omega, config)

    # Solve k equation
    prev .= k.values
    discretise!(k_eqn, k, config)
    apply_boundary_conditions!(k_eqn, k.BCs, nothing, config)
    implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    # implicit_relaxation_diagdom!(k_eqn, k.values, solvers.k.relax, nothing, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    solve_system!(k_eqn, solvers.k, k, nothing, config)
    bound!(k, config)

    @. nut.values = k.values/omega.values

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)
end

# Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, VTKWriter, name
    ) where {T,F,M,Tu<:KOmega,E,D,BI}
    if typeof(model.fluid)<:AbstractCompressible
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("T", model.energy.T),
            ("k", model.turbulence.k),
            ("omega", model.turbulence.omega),
            ("nut", model.turbulence.nut)
        )
    else
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("k", model.turbulence.k),
            ("omega", model.turbulence.omega),
            ("nut", model.turbulence.nut)
        )
    end
    write_vtk(name, model.domain, VTKWriter, args...)
end