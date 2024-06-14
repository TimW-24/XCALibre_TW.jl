using FVM_1D
using CUDA

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/flatplate_transition.unv"
# mesh_file = "unv_sample_meshes/flatplate_2D_lowRe.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"


mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)

# Turbulence Model
velocity = [5.4,0,0]
nu = 1.48e-5
Re = 10*1/nu
νR = 15
Tu = 0.03
k_inlet = 0.0575 #3/2*(Tu*velocity[1])^2
kL_inlet = 0.0115 #1/2*(Tu*velocity[1])^2
ω_inlet = 275 #k_inlet/(νR*nu)

# model = RANS{KOmegaLKE}(mesh=mesh, viscosity=ConstantScalar(nu), Tu=Tu)

model = Physics(
    time = Steady(),
    fluid = Incompressible(nu = ConstantScalar(nu)),
    turbulence = RANS{KOmegaLKE}(),
    energy = nothing,
    domain = mesh
    )
phi = ScalarField(mesh)
phi = assign(
    phi, model,
        Neumann(:inlet, 0.0),
        Neumann(:outlet, 0.0),
        Dirichlet(:wall,0.0),
        # Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        Neumann(:freestream, 0.0)

        # Neumann(:inlet, 0.0),
        # Neumann(:outlet, 0.0),
        # Dirichlet(:cylinder, 0.0),
        # Neumann(:bottom, 0.0),
        # Neumann(:top, 0.0)
)

schemes = (
    phi = set_schemes(gradient=Midpoint),
)

solvers = (
    phi = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver,# CgSolver, BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-6,
        itmax = 10,
        relax = 0.99,
        rtol = 1e-10,
        atol = 1e-15
    ),

)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=32)

runtime = set_runtime(
    iterations=2000, write_interval=100, time_step=1)
    
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

# y = wall_distance(model, config; walls=(:wall))
y = wall_distance(phi, model, config)
write_vtk("wallDist", model.domain, ("y", y))
# Boundary Conditions
noSlip = [0.0, 0.0, 0.0]

@assign! model U (
    FVM_1D.Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    FVM_1D.Dirichlet(:wall, [0.0, 0.0, 0.0]),
    FVM_1D.Dirichlet(:bottom, velocity),
    Neumann(:freestream, 0.0),
)

@assign! model p (
    Neumann(:inlet, 0.0),
    FVM_1D.Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:freestream, 0.0)
)

@assign! model phi (
    Neumann(:inlet, 0.0),
    Neumann(:outlet, 0.0),
    FVM_1D.Dirichlet(:wall,0.0),
    Neumann(:bottom, 0.0),
    Neumann(:freestream, 0.0)
)

@assign! model turbulence kL (
    FVM_1D.Dirichlet(:inlet, kL_inlet),
    Neumann(:outlet, 0.0),
    FVM_1D.Dirichlet(:wall, 1e-15),
    Neumann(:bottom, 0.0),
    Neumann(:freestream, 0.0)
)

@assign! model turbulence k (
    FVM_1D.Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    FVM_1D.Dirichlet(:wall, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:freestream, 0.0)
)

@assign! model turbulence omega (
    FVM_1D.Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    Neumann(:bottom, 0.0),
    Neumann(:freestream, 0.0)
)

@assign! model turbulence nut (
    FVM_1D.Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    FVM_1D.Dirichlet(:wall, 0.0), 
    Neumann(:bottom, 0.0),
    Neumann(:freestream, 0.0),
)

schemes = (
    U = set_schemes(divergence=Upwind),
    p = set_schemes(divergence=Upwind),
    k = set_schemes(divergence=Upwind),
    phi = set_schemes(gradient=Midpoint),
    kL = set_schemes(divergence=Upwind,gradient=Midpoint),
    omega = set_schemes(divergence=Upwind)
)


solvers = (
    U = set_solver(
        model.U;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-2,
        atol = 1e-6
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-3,
        atol = 1e-6
    ),
    phi = set_solver(
        model.phi;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.85,
    ),
    kL = set_solver(
        model.turbulence.kL;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.6,
        rtol = 1e-2,
        atol = 1e-6
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.6,
        rtol = 1e-2,
        atol = 1e-6
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.6,
        rtol = 1e-2,
        atol = 1e-6
    )
)

runtime = set_runtime(
    iterations=2000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)
initialise!(model.phi, 0.0)
initialise!(model.turbulence.kL, kL_inlet)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

Rx, Ry, Rp = simple!(model, config) # 9.39k allocs

let
    p = plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
    plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
    plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
    plot!(1:length(Rp), Rp, yscale=:log10, label="p")
    display(p)
end

using DelimitedFiles
using LinearAlgebra

# OF_data = readdlm("flatplate_OF_wall_kOmega_lowRe.csv", ',', Float64, skipstart=1)
# oRex = OF_data[:,7].*velocity[1]./nu[1]
# oCf = sqrt.(OF_data[:,12].^2 + OF_data[:,13].^2)/(0.5*velocity[1]^2)

tauw, pos = wall_shear_stress(:wall, model)
tauMag = [norm(tauw[i]) for i ∈ eachindex(tauw)]
x = [pos[i][1] for i ∈ eachindex(pos)]
Rex = velocity[1].*x./nu

x_corr = [0:0.0002:2;]
Rex_corr = velocity[1].*x_corr/nu
Cf_corr = 0.0576.*(Rex_corr).^(-1/5)
Cf_laminar = 0.664.*(Rex_corr).^(-1/2)
plot(; xaxis="Rex", yaxis="Cf")
plot!(Rex_corr, Cf_corr, color=:red, ylims=(0, 0.01), xlims=(0,6e5), label="Turbulent",lw=1.5)
plot!(Rex_corr, Cf_laminar, color=:green, ylims=(0, 0.01), xlims=(0,6e5), label="Laminar",lw=1.5)
# plot!(oRex, oCf, color=:green, lw=1.5, label="OpenFOAM") # |> display
plot!(Rex,tauMag./(0.5*velocity[1]^2), color=:blue, lw=1.5,label="Code") |> display

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p") |> display