using Plots
using FVM_1D
using Krylov
using CUDA

#mesh_file="src/UNV_3D/5_cell_new_boundaries.unv"
mesh_file="src/UNV_3D/5_cell_new_boundaries.unv"
mesh_file="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.06m.unv"

mesh=build_mesh3D(mesh_file)

velocity = [0.01,0.0,0.0]
nu=1e-3
Re=velocity[1]*0.1/nu
noSlip = [0.0, 0.0, 0.0]

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, noSlip),
    Dirichlet(:top, noSlip),
    Dirichlet(:side1, noSlip),
    Dirichlet(:side2, noSlip)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0),
    Neumann(:side1, 0.0),
    Neumann(:side2, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind),
    p = set_schemes(divergence=Upwind)
)

solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-4
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-4

    )
)

runtime = set_runtime(
    iterations=1000, time_step=1, write_interval=10)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

# backend = CUDABackend()

Rx, Ry, Rz, Rp = simple!(model, config)

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rz), Rz, yscale=:log10, label="Uz")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

using Profile, PProf

GC.gc()
initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 begin 
    Rx, Ry, Rz, Rp = simple!(model, config)
end

PProf.Allocs.pprof()
