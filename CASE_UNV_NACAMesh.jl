using Plots, XCALibre, AerofoilOptimisation, LinearAlgebra
using BayesianOptimization

#%% REYNOLDS & Y+ CALCULATIONS

#=yplus_init,BL_layers = 2.0,55
laminar = false
velocity,BL_mesh = BL_calcs(Re,nu,ρ,chord,yplus_init,BL_layers,laminar) #Returns (BL mesh thickness, BL mesh growth rate)

#%% CFD CASE SETUP & SOLVE

iter = 1
aero_eff = Array{Float64,1}(undef,iter)
C_l = Array{Float64,1}(undef,iter)
C_d = Array{Float64,1}(undef,iter)
for i ∈ 1:iter
    α = i-1
    writes = α > 10 ? 50 : 1000
=#
# Aerofoil Mesh
chord = 250.0

create_NACA_mesh(
    chord = chord, #[mm]
    α = 0, #[°]
    cutoff = 0.5*(chord/100), #Min thickness of TE [mm]. Default = 0.5 @ 100mm chord; reduce for aerofoils with very thin TE
    vol_size = (16,10), #Total fluid volume size (x,y) in chord multiples [aerofoil located in the vertical centre at the 1/3 position horizontally]
    ratio = 0.75,
    BL_thick = 1, #Boundary layer mesh thickness [%c]
    BL_layers = 35, #Boundary layer mesh layers [-]
    BL_stretch = 1.2, #Boundary layer stretch factor (successive multiplication factor of cell thickness away from wall cell) [-]
    py_lines = (14,37,44,248,358,391,405,353), #SALOME python script relevant lines (notebook path, chord line, points line, splines line, BL thickness, foil end BL fidelity, .unv path)
    dat_path = "/home/tim/Documents/MEng Individual Project/Julia/AerofoilOptimisation/foil_dats/NACA0012.dat",
    py_path = "/home/tim/Documents/MEng Individual Project/Julia/AerofoilOptimisation/foil_pythons/NACAMesh.py", #Path to SALOME python script
    salome_path = "/home/tim/Downloads/InstallationFiles/SALOME-9.11.0/mesa_salome", #Path to SALOME installation
    unv_path = "/home/tim/Documents/MEng Individual Project/Julia/XCALibre_TW.jl/unv_sample_meshes/NACAMesh.unv", #Path to .unv destination
    note_path = "/home/tim/Documents/MEng Individual Project/SALOME", #Path to SALOME notebook (.hdf) destination
    GUI = false #SALOME GUI selector
)
mesh_file = "unv_sample_meshes/NACAMesh.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# Velocity calculation for given α
function vel_calc(Re,α,nu,chord)
    vel = [0.0,0.0,0.0]
    Umag = (Re*nu)/(chord*0.001)
    vel[1] = Umag*cos(α[1]*π/180)
    vel[2] = Umag*sin(α[1]*π/180)
    return vel
end
#function foil_optim(α::Vector{Float64})
#    println(α)
    # Parameters
    Re = 1000000
    α = 2
    nu,ρ = 1.48e-5,1.225
    velocity = vel_calc(Re,α,nu,chord)
    νR = 10
    Tu = 0.025
    k_inlet = 3/2*(Tu*norm(velocity))^2
    ω_inlet = k_inlet/(νR*nu)

    # Boundary Conditions
    noSlip = [0.0, 0.0, 0.0]

    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu = nu),
        turbulence = RANS{KOmega}(),
        energy = Energy{Isothermal}(),
        domain = mesh
        )

    @assign! model momentum U ( 
        Dirichlet(:inlet, velocity),
        Neumann(:outlet, 0.0),
        Dirichlet(:top, velocity),
        Dirichlet(:bottom, velocity),
        Dirichlet(:foil, noSlip)
    )

    @assign! model momentum p (
        Neumann(:inlet, 0.0),
        Dirichlet(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        Neumann(:foil, 0.0)
    )

    @assign! model turbulence k (
        Dirichlet(:inlet, k_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        Dirichlet(:foil, 1e-15)
    )

    @assign! model turbulence omega (
        Dirichlet(:inlet, ω_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        OmegaWallFunction(:foil) # need constructor to force keywords
    )

    @assign! model turbulence nut (
        Dirichlet(:inlet, k_inlet/ω_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0), 
        Dirichlet(:foil, 0.0)
    )


    schemes = (
        U = set_schemes(divergence=Upwind,gradient=Midpoint),
        p = set_schemes(divergence=Upwind),
        k = set_schemes(divergence=Upwind,gradient=Midpoint),
        omega = set_schemes(divergence=Upwind,gradient=Midpoint)
    )

    solvers = (
        U = set_solver(
            model.momentum.U;
            solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
            preconditioner = Jacobi(),
            convergence = 1e-7,
            relax       = 0.5,
            rtol = 1e-2,
            atol = 1e-10
        ),
        p = set_solver(
            model.momentum.p;
            solver      = CgSolver, # BicgstabSolver, GmresSolver
            preconditioner = Jacobi(),
            convergence = 1e-7,
            relax       = 0.3,
            rtol = 1e-3,
            atol = 1e-10
        ),
        k = set_solver(
            model.turbulence.k;
            solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
            preconditioner = Jacobi(),
            convergence = 1e-7,
            relax       = 0.4,
            rtol = 1e-2,
            atol = 1e-10
        ),
        omega = set_solver(
            model.turbulence.omega;
            solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
            preconditioner = Jacobi(),
            convergence = 1e-7,
            relax       = 0.4,
            rtol = 1e-2,
            atol = 1e-10
        )
    )

    runtime = set_runtime(iterations=1000, write_interval=1000, time_step=1)

    hardware = set_hardware(backend=CPU(), workgroup=8)

    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

    GC.gc()

    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    initialise!(model.turbulence.k, k_inlet)
    initialise!(model.turbulence.omega, ω_inlet)
    initialise!(model.turbulence.nut, k_inlet/ω_inlet)

    Rx, Ry, Rp, model_out = run!(model, config) #, pref=0.0)

    #%% POST-PROCESSING
    let
        plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
        plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
        plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
        plot!(1:length(Rp), Rp, yscale=:log10, label="p")
    end

    C_l,C_d = aero_coeffs(:foil, chord, ρ, velocity, model.momentum.p)
    aero_eff = lift_to_drag(:foil, ρ, model)

    if isnan(aero_eff)
        aero_eff = 0
    end

    vtk_files = filter(x->endswith(x,".vtk"), readdir("vtk_results/"))
    for file ∈ vtk_files
        filepath = "vtk_results/"*file
        dest = "vtk_loop/NACA_Optimisation/$(aero_eff),$(α)"*file
        mv(filepath, dest,force=true)
    end
#    return aero_eff
#end
model = ElasticGPE(1,                            # 2 input dimensions
                   mean = MeanConst(0.0),         
                   kernel = SEArd([0.0], 5.0),
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.
set_priors!(model.mean, [Normal(1, 2)])

modeloptimizer = MAPGPOptimizer(every = 10,       
                                maxeval = 40)

opt = BOpt(foil_optim,
            model,
            UpperConfidenceBound(),
            modeloptimizer,                        
            [0.0], [15.0],       
            repetitions = 1,
            maxiterations = 100,
            sense = Max,
            initializer_iterations = 10,   
            verbosity = Progress)

result = boptimize!(opt)

#=
paraview_vis(paraview_path = "paraview", #Path to paraview
             vtk_path = "/home/tim/Documents/MEng Individual Project/Julia/FVM_1D_TW/vtk_results/iteration_..vtk") #Path to vtk files
=#
