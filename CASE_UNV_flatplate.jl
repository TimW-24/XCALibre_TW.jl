using Plots

using FVM_1D

using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/flatplate_2D_laminar.unv"
mesh = build_mesh(mesh_file, scale=0.001)


velocity = [0.2, 0.0, 0.0]
nu = 1e-5
Re = velocity[1]*1/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Neumann(:top, 0.0)
)

 @assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Linear),
    p = set_schemes(divergence=Linear)
)


solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = DILU(),
        convergence = 1e-7,
        relax       = 0.8,
    ),
    p = set_solver(
        model.p;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.2,
    )
)

runtime = set_runtime(iterations=2000, write_interval=-1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Rx, Ry, Rp = isimple!(model, config) # 9.39k allocs

write_vtk("results", mesh, ("U", model.U), ("p", model.p))

using DelimitedFiles
using LinearAlgebra

OF_data = readdlm("flatplate_OF_wall_laminar.csv", ',', Float64, skipstart=1)
oRex = OF_data[:,7].*velocity[1]./nu[1]
oCf = sqrt.(OF_data[:,9].^2 + OF_data[:,10].^2)/(0.5*velocity[1]^2)

tauw, pos = wall_shear_stress(:wall, model)
tauMag = [norm(tauw[i]) for i ∈ eachindex(tauw)]
x = [pos[i][1] for i ∈ eachindex(pos)]

Rex = velocity[1].*x/nu[1]
Cf = 0.664./sqrt.(Rex)
plot(; xaxis="Rex", yaxis="Cf")
plot!(Rex, Cf, color=:red, ylims=(0, 0.05), xlims=(0,2e4), label="Blasius",lw=1.5)
plot!(oRex, oCf, color=:green, lw=1.5, label="OpenFOAM")
plot!(Rex,tauMag./(0.5*velocity[1]^2), color=:blue, lw=1.5,label="Code") |> display

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p") |> display