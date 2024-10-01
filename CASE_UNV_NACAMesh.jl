using Plots, XCALibre, BayesianOptimization
using LinearAlgebra, GaussianProcesses, Distributions

chord = 250.0

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
function foil_optim(α::Vector{Float64})
    println(α)
    # Parameters
    Re = 1000000
    α = α
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
        turbulence = RANS{KOmega}(β⁺=0.09),
        energy = Energy{Isothermal}(),
        domain = mesh
        )

    @assign! model momentum U ( 
        XCALibre.Dirichlet(:inlet, velocity),
        Neumann(:outlet, 0.0),
        XCALibre.Dirichlet(:top, velocity),
        XCALibre.Dirichlet(:bottom, velocity),
        XCALibre.Dirichlet(:foil, noSlip)
    )

    @assign! model momentum p (
        Neumann(:inlet, 0.0),
        XCALibre.Dirichlet(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        Neumann(:foil, 0.0)
    )

    @assign! model turbulence k (
        XCALibre.Dirichlet(:inlet, k_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        XCALibre.Dirichlet(:foil, 1e-15)
    )

    @assign! model turbulence omega (
        XCALibre.Dirichlet(:inlet, ω_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        OmegaWallFunction(:foil) # need constructor to force keywords
    )

    @assign! model turbulence nut (
        XCALibre.Dirichlet(:inlet, k_inlet/ω_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0), 
        XCALibre.Dirichlet(:foil, 0.0)
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
            relax       = 0.7,
            rtol = 1e-3,
            atol = 1e-10
        ),
        p = set_solver(
            model.momentum.p;
            solver      = CgSolver, # BicgstabSolver, GmresSolver
            preconditioner = Jacobi(),
            convergence = 1e-7,
            relax       = 0.3,
            rtol = 1e-4,
            atol = 1e-10
        ),
        k = set_solver(
            model.turbulence.k;
            solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
            preconditioner = Jacobi(),
            convergence = 1e-7,
            relax       = 0.3,
            rtol = 1e-3,
            atol = 1e-10
        ),
        omega = set_solver(
            model.turbulence.omega;
            solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
            preconditioner = Jacobi(),
            convergence = 1e-7,
            relax       = 0.3,
            rtol = 1e-3,
            atol = 1e-10
        )
    )

    runtime = set_runtime(iterations=500, write_interval=500, time_step=1)

    hardware = set_hardware(backend=CPU(), workgroup=6)

    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

    GC.gc()

    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    initialise!(model.turbulence.k, k_inlet)
    initialise!(model.turbulence.omega, ω_inlet)
    initialise!(model.turbulence.nut, k_inlet/ω_inlet)

    Rx, Ry, Rz, Rp, model_out = run!(model, config) #, pref=0.0)

    #%% POST-PROCESSING
    let
        plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
        plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
        plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
        plot!(1:length(Rp), Rp, yscale=:log10, label="p")
    end

    C_l,C_d = aero_coeffs(:foil, chord, velocity, model, ρ, nu)
    aero_eff = lift_to_drag(:foil, model, ρ, nu)

    if isnan(aero_eff)
        aero_eff = 0
    end
    aero_eff = round(aero_eff,digits=3)
    α = round(α[1],digits=3)
    vtk_files = filter(x->endswith(x,".vtk"), readdir("vtk_results/"))
    for file ∈ vtk_files
        filepath = "vtk_results/"*file
        dest = "vtk_loop/NACA_Optimisation/$(aero_eff),$(α)"*file
        mv(filepath, dest,force=true)
    end
    return aero_eff
end
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

alpha_inputs = [2.8125 10.3125 14.0625 6.5625 4.6875 12.1875 8.4375 0.9375 1.40625 8.90625 3.4432126486612407 0.0 2.361306954903304 15.0 5.476514715973348 11.26402659402207 7.39241592658447 13.133937006330672 2.7830815990127813 0.422884724292046 9.38721928615226 3.4024415656772113 1.1987256130185693 14.684527104647096 13.535519587314896 6.929927822198579 14.698546513842413 3.4990688331549125 0.0869761499534637 0.7112198842942964 6.338859238458447 7.1651765143230435 5.475010255776448 14.15347629116867 4.544399245216521 5.255120935630782 9.03719296405648 1.807185497127921 2.719939384552249 0.39405408470667214 6.142924905046002 14.324171433543196 4.608655359771077 5.298885247860957 9.613924812184823 7.444739911521388 12.720833156006066 9.889514857297746 8.17079235179106 3.728882092443396 0.7226152801680674 0.4738547802690282 7.825244758100422 9.117285891015088 2.4133035700392496 2.0123057103010584 5.159388324942617 5.65935815598451 4.034283167891717 14.239635603789512 0.25044779686362056 6.492884747695639 6.747496857595797 0.3391480736236435 5.756697367982879 1.0920079522146666 13.032460047664308 14.906616140638016 11.385716373741317 11.154855611851975 10.072838682793048]
aeroeff_outputs = [124.162, -7.413, -7.434, -12.953, -24.594, -6.669, -9.205, 7.203, 12.417, -8.634, -92.213, -0.015, 40.596, -9.342, -17.59, -6.88, -10.915, -6.825, 110.738, 2.991, -8.142, -102.806, 9.86, -8.646, -6.964, -11.949, -8.686, -80.908, 0.593, 5.229, -13.669, -11.396, -17.599, -7.566, -26.629, -19.068, -8.491, 19.334, 89.392, 2.778, -14.379, -7.898, -25.669, -18.751, -7.939, -10.812, -6.723, -7.713, -9.577, -54.452, 5.321, 3.368, -10.121, -8.407, 44.384, 24.707, -19.795, -16.554, -38.626, -7.727, 1.743, -13.167, -12.423, 2.379, -16.057, 8.711, -6.797, -9.101, -6.832, -6.925, -7.576]
aeroeff_outputs = reshape(aeroeff_outputs,1,71)
#=
paraview_vis(paraview_path = "paraview", #Path to paraview
             vtk_path = "/home/tim/Documents/MEng Individual Project/Julia/FVM_1D_TW/vtk_results/iteration_..vtk") #Path to vtk files
=#
scatter(alpha_inputs,abs.(aeroeff_outputs),legend=false)