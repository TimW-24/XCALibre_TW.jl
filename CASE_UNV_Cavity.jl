using Plots
using FVM_1D
using Krylov

# quad and trig 40 and 100
mesh_file = "unv_sample_meshes/trig100.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_gpu = adapt(CUDABackend(), mesh)

velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = 1*velocity[1]/nu

model = Physics(
    time = Steady(),
    fluid = Incompressible(nu = ConstantScalar(nu)),
    turbulence = RANS{Laminar}(),
    energy = nothing,
    domain = mesh_gpu
    )

@assign! model momentum U (
    Dirichlet(:inlet, noSlip),
    Dirichlet(:outlet, noSlip),
    Dirichlet(:bottom, noSlip),
    Dirichlet(:top, velocity)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Neumann(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(gradient=Midpoint),
    p = set_schemes(gradient=Midpoint)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.8,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.2,
    )
)

runtime = set_runtime(iterations=2000, write_interval=2000)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, [0, 0, 0])
initialise!(model.momentum.p, 0.0)

Rx, Ry, Rz, Rp, model_out = run!(model, config, pref=0.0) # 9.39k allocs

plot(1:runtime.iterations, Rx, yscale=:log10)
plot!(1:runtime.iterations, Ry, yscale=:log10)
plot!(1:runtime.iterations, Rp, yscale=:log10)