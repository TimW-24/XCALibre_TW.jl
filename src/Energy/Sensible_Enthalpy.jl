export SensibleEnthalpy
export Ttoh, htoT!, Ttoh!, thermo_Psi!

# Model type definition
struct SensibleEnthalpy{S1,S2,F1,F2,F,C} <: AbstractEnergyModel
    h::S1
    T::S2
    hf::F1
    Tf::F2
    update_BC::F
    coeffs::C
end
Adapt.@adapt_structure SensibleEnthalpy

struct Sensible_Enthalpy_Model{E1}
    energy_eqn::E1 
end
Adapt.@adapt_structure Sensible_Enthalpy_Model

# Model API constructor
ENERGY{SensibleEnthalpy}(; Tref = 288.15) = begin
    coeffs = (Tref=Tref, other=nothing)
    ARG = typeof(coeffs)
    ENERGY{SensibleEnthalpy,ARG}(coeffs)
end

# Functor as consturctor
(energy::ENERGY{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:SensibleEnthalpy,ARG} = begin
    h = ScalarField(mesh)
    T = ScalarField(mesh)
    hf = FaceScalarField(mesh)
    Tf = FaceScalarField(mesh)
    update_BC =  return_thingy(EnergyModel, fluid, energy.args.Tref)
    coeffs = energy.args
    SensibleEnthalpy(h, T, hf, Tf, update_BC, coeffs)
end

return_thingy(::Type{SensibleEnthalpy}, fluid, Tref) = begin
    function Ttoh(T)
        Cp = _Cp(fluid)
        h = Cp.values*(T-Tref)
        return h
    end
end

# function Ttoh(
#     model::Physics{T1,F,M,Tu,E,D,BI}, T
#     ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
#     (; coeffs) = model.energy
#     (; Tref) = coeffs
#     # h = ScalarField(model.domain)
#     Cp = _Cp(model.fluid)
#     h = Cp.values*(T-Tref)
#     return h
# end

function initialise(
    energy::SensibleEnthalpy, model::Physics{T1,F,M,Tu,E,D,BI}, mdotf, rho, peqn, config
    ) where {T1,F,M,Tu,E,D,BI}

    (; h, T) = energy
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation
    
    # rho = ScalarField(mesh)
    keff_by_cp = FaceScalarField(mesh)
    divK = ScalarField(mesh)

    Ttoh!(model, T, h)

    # println(h.BCs[1])

    energy_eqn = (
        Time{schemes.h.time}(rho, h)
        + Divergence{schemes.h.divergence}(mdotf, h) 
        - Laplacian{schemes.h.laplacian}(keff_by_cp, h) 
        == 
        -Source(divK)
    ) → eqn
    
    # Set up preconditioners
    @reset energy_eqn.preconditioner = set_preconditioner(
                solvers.h.preconditioner, energy_eqn, h.BCs, config)
    
    # preallocating solvers
    @reset energy_eqn.solver = solvers.h.solver(_A(energy_eqn), _b(energy_eqn))

    return Sensible_Enthalpy_Model(energy_eqn)
end

function energy!(
    energy::Sensible_Enthalpy_Model{E1}, model::Physics{T1,F,M,Tu,E,D,BI}, prev, mdotf, rho, mueff, config
    ) where {T1,F,M,Tu,E,D,BI,E1}

    mesh = model.domain

    (;U) = model.momentum
    (;h, hf, T) = model.energy
    (;energy_eqn) = energy
    (; solvers, runtime) = config

    println("Maxh ", maximum(h.values), " minh ", minimum(h.values))

    # rho = get_flux(energy_eqn, 1)
    keff_by_cp = get_flux(energy_eqn, 3)
    divK = get_source(energy_eqn, 1)

    Uf = FaceVectorField(mesh)
    Kf = FaceScalarField(mesh)
    K = ScalarField(mesh)
    Kbounded = ScalarField(mesh)
    Pr = _Pr(model.fluid)

    volumes = getproperty.(mesh.cells, :volume)

    # println("Minmdot ", minimum(rho.values), ", Maxdoot ", maximum(rho.values))

    @. keff_by_cp.values = mueff.values/Pr.values

    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, U.BCs, config)
    for i ∈ eachindex(K)
        K.values[i] = 0.5*(U.x.values[i]^2 + U.y.values[i]^2 + U.z.values[i]^2)
    end
    interpolate!(Kf, K, config)
    for i ∈ eachindex(Kf)
        Kf.values[i] = 0.5*(Uf.x.values[i]^2 + Uf.y.values[i]^2 + Uf.z.values[i]^2)
    end
    correct_face_interpolation!(Kf, K, mdotf) # This forces KE to be upwind 
    @. Kf.values *= mdotf.values
    div!(divK, Kf, config)
    div!(Kbounded, mdotf, config)
    println("MaxdivK ", maximum(divK.values), " mindivK ", minimum(divK.values))
    @. divK.values .- Kbounded.values * K.values # This might need dividing by the volume, unsure
    println("MaxdivK ", maximum(divK.values), " mindivK ", minimum(divK.values))

    # solve_equation!(energy_eqn, h, solvers.h, config) # This doesn't work for this scalarfield yet
    # Set up and solve energy equation
    @. prev = h.values
    discretise!(energy_eqn, h, config)
    apply_boundary_conditions!(energy_eqn, h.BCs, nothing, config)
    implicit_relaxation_diagdom!(energy_eqn, h.values, solvers.h.relax, nothing, config)
    update_preconditioner!(energy_eqn.preconditioner, mesh, config)
    solve_system!(energy_eqn, solvers.h, h, nothing, config)

    println("Maxh ", maximum(h.values), " minh ", minimum(h.values))
    
    # println("Maxh ", maximum(h.values), " minh ", minimum(h.values))

    Tmin = 100.0; Tmax = 2000.0
    thermoClamp!(model, h, Tmin, Tmax)

    htoT!(model, h, T)
    interpolate!(hf, h, config)
    correct_boundaries!(hf, h, h.BCs, config)
    # thermoClamp!(model, hf, Tmin, Tmax)
end

function thermo_Psi!(
    model::Physics{T,F,M,Tu,E,D,BI}, Psi::ScalarField
    ) where {T,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs, h) = model.energy
    (; Tref) = coeffs
    Cp = _Cp(model.fluid); R = _R(model.fluid)
    @. Psi.values = Cp.values/(R.values*(h.values + Cp.values*Tref))
end

function thermo_Psi!(
    model::Physics{T,F,M,Tu,E,D,BI}, Psif::FaceScalarField, config
    ) where {T,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs, hf, h) = model.energy
    interpolate!(hf, h, config)
    correct_boundaries!(hf, h, h.BCs, config)
    (; Tref) = coeffs
    Cp = _Cp(model.fluid); R = _R(model.fluid)
    @. Psif.values = Cp.values/(R.values*(hf.values + Cp.values*Tref))
end


# function thermo_rho!(
#     model::Physics{T,F,M,Tu,E,D,BI}, rhof::FaceScalarField, Psif
#     ) where {T,F<:AbstractCompressible,M,Tu,E,D,BI}

# end

function Ttoh!(
    model::Physics{T1,F,M,Tu,E,D,BI}, T::ScalarField, h::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = _Cp(model.fluid)
    @. h.values = Cp.values*(T.values-Tref)
end

# function Ttoh(
#     model::Physics{T1,F,M,Tu,E,D,BI}, T
#     ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
#     (; coeffs) = model.energy
#     (; Tref) = coeffs
#     # h = ScalarField(model.domain)
#     Cp = _Cp(model.fluid)
#     h = Cp.values*(T-Tref)
#     return h
# end

function htoT!(
    model::Physics{T1,F,M,Tu,E,D,BI}, h::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = _Cp(model.fluid)
    @. T.values = (h.values/Cp.values) + Tref
end



function thermoClamp!(
    model::Physics{T1,F,M,Tu,E,D,BI}, h::ScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = _Cp(model.fluid)
    hmin = Cp.values*(Tmin-Tref)
    hmax = Cp.values*(Tmax-Tref)
    clamp!(h.values, hmin, hmax)
end

function thermoClamp!(
    model::Physics{T1,F,M,Tu,E,D,BI}, hf::FaceScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = _Cp(model.fluid)
    hmin = Cp.values*(Tmin-Tref)
    hmax = Cp.values*(Tmax-Tref)
    clamp!(hf.values, hmin, hmax)
end

function correct_face_interpolation!(phif::FaceScalarField, phi, Uf::FaceScalarField)
    mesh = phif.mesh
    (; faces, cells) = mesh
    for fID ∈ eachindex(faces)
        face = faces[fID]
        (; ownerCells, area, normal) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        phi1 = phi[cID1]
        phi2 = phi[cID2]
        flux = Uf[fID]
        if flux >= 0.0
            phif.values[fID] = phi1
        else
            phif.values[fID] = phi2
        end
    end
end
