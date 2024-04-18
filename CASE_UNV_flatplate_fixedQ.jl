using Plots

using FVM_1D

using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/flatplate_2D_laminar.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)

velocity = [0.5, 0.0, 0.0]
nu = 1E-4
Cp = 1005
Re = velocity[1]*1/nu
pressure = 100000
h_inf = 300*Cp

model = RANS{Laminar_rho}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Neumann(:top, 0.0)
)

 @assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, pressure),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model energy (
    Dirichlet(:inlet, h_inf),
    Neumann(:outlet, 0.0),
    # Neumann(:wall, 0.0*Cp),#,200.0*Cp),#-20.0),
    # Dirichlet(:wall, 1.2*h_inf),#,200.0*Cp),#-20.0),
    FixedGradient(:wall, 1.2*h_inf),#,200.0*Cp),#-20.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind),
    p = set_schemes(divergence=Upwind),
    energy = set_schemes(divergence=Upwind)
)


solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.8,
        atol        = 1e-5,
        rtol        = 1e-2,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.2,
        atol        = 1e-6,
        rtol        = 1e-3,
    ),
    energy = set_solver(
        model.energy;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.8,
        atol        = 1e-6,
        rtol        = 1e-3,
    ),
)

runtime = set_runtime(iterations=1000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, pressure)
initialise!(model.energy, h_inf)

Rx, Ry, Rz, Rp, Re = simple_rho_K!(model, config)

using DelimitedFiles
using LinearAlgebra

# OF_data = readdlm("flatplate_OF_wall_laminar.csv", ',', Float64, skipstart=1)
# oRex = OF_data[:,7].*velocity[1]./nu[1]
# oCf = sqrt.(OF_data[:,9].^2 + OF_data[:,10].^2)/(0.5*velocity[1]^2)

# tauw, pos = wall_shear_stress(:wall, model)
# tauMag = [norm(tauw[i]) for i ∈ eachindex(tauw)]
# x = [pos[i][1] for i ∈ eachindex(pos)]

# Rex = velocity[1].*x/nu[1]
# Cf = 0.664./sqrt.(Rex)
# plot(; xaxis="Rex", yaxis="Cf")
# plot!(Rex, Cf, color=:red, ylims=(0, 0.05), xlims=(0,2e4), label="Blasius",lw=1.5)
# plot!(oRex, oCf, color=:green, lw=1.5, label="OpenFOAM")
# plot!(Rex,tauMag./(0.5*velocity[1]^2), color=:blue, lw=1.5,label="Code")

plot(; xlims=(0,runtime.iterations))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Re), Re, yscale=:log10, label="energy")