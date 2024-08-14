using Plots
using FVM_1D
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_2mm.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_10-7.5-2mm.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# mesh_gpu = adapt(CUDABackend(), mesh)

# Inlet conditions

velocity = [5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu
gamma = 1.4
cp = 1005.0
R = 287.0
temp = 300.0
pressure = 100000
Pr = 0.7

model = Physics(
    time = Transient(),
    fluid = FLUID{WeaklyCompressible}(
        nu = nu,
        cp = cp,
        gamma = gamma,
        Pr = Pr
        ),
    turbulence = RANS{Laminar}(),
    energy = ENERGY{SensibleEnthalpy}(),
    domain = mesh
    )

@assign! model momentum U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:cylinder, noSlip),
    # Dirichlet(:cylinder, noSlip),
    Symmetry(:bottom, 0.0),
    Symmetry(:top, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, pressure),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

@assign! model energy h (
    FixedTemperature(:inlet, T=300.0, model=model.energy),
    Neumann(:outlet, 0.0),
    # Neumann(:cylinder, 0.0),
    FixedTemperature(:cylinder, T=310.0, model=model.energy),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

solvers = (
    rho = set_solver(
        model.fluid.rho;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1,
        rtol = 1e-4,
        atol = 1e-2
    ),
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1,
        rtol = 1e-4,
        atol = 1e-2
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1,
        rtol = 1e-4,
        atol = 1e-3
    ),
    h = set_solver(
        model.energy.h;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1,
        rtol = 1e-4,
        atol = 1e-2
    )
)

schemes = (
    rho = set_schemes(time=Euler),
    U = set_schemes(divergence=Upwind, gradient=Midpoint, time=Euler),
    p = set_schemes(divergence=Upwind, gradient=Midpoint, time=Euler),
    h = set_schemes(divergence=Upwind, gradient=Midpoint, time=Euler)
)

runtime = set_runtime(iterations=5000, write_interval=100, time_step=0.0001)

hardware = set_hardware(backend=CPU(), workgroup=4)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, pressure)
initialise!(model.energy.T, temp)
initialise!(model.fluid.rho, pressure/(R*temp))

println("Maxh ", maximum(model.energy.T.values), " minh ", minimum(model.energy.T.values))

Rx, Ry, Rz, Rp, Rh, model = run!(model, config); #, pref=0.0)

plot(; xlims=(0,runtime.iterations), ylims=(1e-8,0))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Rh), Rh, yscale=:log10, label="h")