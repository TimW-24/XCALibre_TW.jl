using Plots, XCALibre, Krylov, AerofoilOptimisation
using BayesianOptimization, GaussianProcesses, Distributions

function foil_optim(y::Vector{Float64})
    println(y)
    #%% AEROFOIL GEOMETRY DEFINITION
    foil,ctrl_p = spline_foil(FoilDef(
        chord   = 100, #[mm]
        LE_h    = 0, #[%c, at α=0°]
        TE_h    = 0, #[%c, at α=0°]
        peak    = [25,y[1]], #[%c]
        trough  = [75,-y[2]], #[%c]
        xover = 50, #[%c]
        α = 5 #[°]
    )) #Returns aerofoil MCL & control point vector (spline method)

    #%% REYNOLDS & Y+ CALCULATIONS
    chord = 100.0
    Re = 10000
    nu,ρ = 1.48e-5,1.225
    yplus_init,BL_layers = 2.0,35
    laminar = false
    velocity,BL_mesh = BL_calcs(Re,nu,ρ,chord,yplus_init,BL_layers,laminar) #Returns (BL mesh thickness, BL mesh growth rate)

    #%% AEROFOIL MESHING
    lines = update_mesh(
        chord = foil.chord, #[mm]
        ctrl_p = ctrl_p, #Control point vector
        vol_size = (16,10), #Total fluid volume size (x,y) in chord multiples [aerofoil located in the vertical centre at the 1/3 position horizontally]
        thickness = 1, #Aerofoil thickness [%c]
        BL_thick = 1, #Boundary layer mesh thickness [mm]
        BL_layers = BL_layers, #Boundary layer mesh layers [-]
        BL_stretch = 1.2, #Boundary layer stretch factor (successive multiplication factor of cell thickness away from wall cell) [-]
        ratio = 1.15,
        py_lines = (13,44,51,59,36,68,225,247,284), #SALOME python script relevant lines (notebook path, 3 B-Spline lines,chord line, thickness line, BL line .unv path)
        py_path = "/home/tim/Documents/MEng Individual Project/Julia/AerofoilOptimisation/foil_pythons/FoilMesh.py", #Path to SALOME python script
        salome_path = "/home/tim/Downloads/InstallationFiles/SALOME-9.11.0/mesa_salome", #Path to SALOME installation
        unv_path = "/home/tim/Documents/MEng Individual Project/Julia/FVM_1D_TW/unv_sample_meshes/FoilMesh.unv", #Path to .unv destination
        note_path = "/home/tim/Documents/MEng Individual Project/SALOME", #Path to SALOME notebook (.hdf) destination
        GUI = false #SALOME GUI selector
    ) #Updates SALOME geometry and mesh to new aerofoil MCL definition


    #%% CFD CASE SETUP & SOLVE
    # Aerofoil Mesh
    mesh_file = "unv_sample_meshes/FoilMesh.unv"
    mesh = build_mesh(mesh_file, scale=0.001)
    mesh = update_mesh_format(mesh)

    # Turbulence Model
    νR = 10
    Tu = 0.025
    k_inlet = 3/2*(Tu*velocity[1])^2
    ω_inlet = k_inlet/(νR*nu)
    model = RANS{KOmega}(mesh=mesh, viscosity=ConstantScalar(nu))

    # Boundary Conditions
    noSlip = [0.0, 0.0, 0.0]

    @assign! model U ( 
        FVM_1D.FVM_1D.Dirichlet(:inlet, velocity),
        Neumann(:outlet, 0.0),
        FVM_1D.Dirichlet(:top, velocity),
        FVM_1D.Dirichlet(:bottom, velocity),
        FVM_1D.Dirichlet(:foil, noSlip)
    )

    @assign! model p (
        Neumann(:inlet, 0.0),
        FVM_1D.Dirichlet(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        Neumann(:foil, 0.0)
    )

    @assign! model turbulence k (
        FVM_1D.Dirichlet(:inlet, k_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        FVM_1D.Dirichlet(:foil, 1e-15)
    )

    @assign! model turbulence omega (
        FVM_1D.Dirichlet(:inlet, ω_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        OmegaWallFunction(:foil) # need constructor to force keywords
    )

    @assign! model turbulence nut (
        FVM_1D.Dirichlet(:inlet, k_inlet/ω_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0), 
        FVM_1D.Dirichlet(:foil, 0.0)
    )


    schemes = (
        U = set_schemes(divergence=Upwind,gradient=Midpoint),
        p = set_schemes(divergence=Upwind),
        k = set_schemes(divergence=Upwind,gradient=Midpoint),
        omega = set_schemes(divergence=Upwind,gradient=Midpoint)
    )

    solvers = (
        U = set_solver(
            model.U;
            solver      = GmresSolver, # BicgstabSolver, GmresSolver
            preconditioner = ILU0(),
            convergence = 1e-7,
            relax       = 0.7,
        ),
        p = set_solver(
            model.p;
            solver      = GmresSolver, # BicgstabSolver, GmresSolver
            preconditioner = LDL(),
            convergence = 1e-7,
            relax       = 0.4,
        ),
        k = set_solver(
            model.turbulence.k;
            solver      = GmresSolver, # BicgstabSolver, GmresSolver
            preconditioner = ILU0(),
            convergence = 1e-7,
            relax       = 0.4,
        ),
        omega = set_solver(
            model.turbulence.omega;
            solver      = GmresSolver, # BicgstabSolver, GmresSolver
            preconditioner = ILU0(),
            convergence = 1e-7,
            relax       = 0.4,
        )
    )

    runtime = set_runtime(
        iterations=1000, write_interval=1000, time_step=1)

    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime)

    GC.gc()

    initialise!(model.U, velocity)
    initialise!(model.p, 0.0)
    initialise!(model.turbulence.k, k_inlet)
    initialise!(model.turbulence.omega, ω_inlet)
    initialise!(model.turbulence.nut, k_inlet/ω_inlet)

    Rx, Ry, Rp = simple!(model, config) #, pref=0.0)

    #%% POST-PROCESSING
    C_l,C_d = aero_coeffs(:foil, chord, ρ, velocity, model)
    aero_eff = lift_to_drag(:foil, ρ, model)

    if isnan(aero_eff)
        aero_eff = 0
    end
    
    let
        p = plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
        plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
        plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
        plot!(1:length(Rp), Rp, yscale=:log10, label="p")
        display(p)
    end
    vtk_files = filter(x->endswith(x,".vtk"), readdir("vtk_results/"))
    for file ∈ vtk_files
        filepath = "vtk_results/"*file
        dest = "vtk_loop/Optimisation (Re=10k,k-w,2 var,5 AoA)/$(aero_eff)"*file
        mv(filepath, dest,force=true)
    end
    return aero_eff
end
model = ElasticGPE(2,                            # 2 input dimensions
                   mean = MeanConst(0.0),         
                   kernel = SEArd([0.0,0.0], 5.0),
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.
set_priors!(model.mean, [Normal(1, 2)])

modeloptimizer = MAPGPOptimizer(every = 10,       
                                maxeval = 40)

#STORAGE OF COMPUTED VALUES
x_vals_2var = [11.08938933397953 15.48478524358638 12.164002984073132 12.665267317724187 21.26068771860812 1.0 15.34740845994921 1.0 9.42885874978036 2.653259572153325 30.0 4.267729125036729 1.0 1.0 15.786762146780873 30.0 2.737493768136838 19.683572162740344 1.0 30.0 30.0 9.820257249439429 11.673746894446964 26.370495531304375 9.506574498660447 19.614447203812354 23.359331007436392 14.526194474816485 7.874144974523727 5.1273253647317425 5.158061558328708 3.777361361168503 14.805483677559673 25.38145327836131 7.27134740125583 3.814643337577621 3.817745360018021 7.386916377159381 3.8177116209126263 5.4689482993908705 3.8177309533844097 6.944983285733449 5.907533032980499 22.698796100917292 10.679562425058732 8.198202442543723 3.8282901913616167 8.27302852625749 12.365596846815482 3.811898651545394 12.830994420292395; 26.06594970426061 21.331303626714156 24.90839448225568 26.537190362714103 14.746335375614578 1.0 24.564036677111638 17.622546497028218 1.0 1.0 25.35950065789937 4.245708413936353 30.0 1.0 27.827264459813676 1.0 23.259636774424752 7.190881749756314 11.199431193728273 19.065928433441623 10.330779652929618 15.038064383513078 11.2039025271804 30.0 18.616053221186974 1.0 21.892195186099517 15.369866610835304 30.0 1.0 14.671291693458903 1.0 2.406499206419488 6.862059691266949 3.2203207577392394 1.0 1.0 22.481362025274223 1.0 26.554820836314928 1.0 11.00964175066907 18.486474202587498 26.93671100363466 21.326905624162503 13.976892280854305 1.0 16.431639017531314 6.790357502460945 1.0 30.0]
y_vals_2var = [5.237414628100506, 3.5703363658222647, 4.970811997633325, 4.289373133638538, 0.787067699901464, 9.55434224325726, 3.17, 3.323, 5.945, 10.80000677639661, 0.622, 4.267, 2.188, 9.554, 2.738175907960623, 0.0747785746782299, 3.8530104100151146, 0.7702909925201151, 2.074455808807779, 0.058414465505498336, -0.09805447703354525, 6.027244507415013, 3.903059865115453, 0.75318843484064, 6.095487481230004, 1.3534367383145045, 0.9585882800209506, 3.0633350932189476, 3.2116405372735892, 10.617440022682384, 5.329122504203392, 11.4909115412491, 2.1746956893396368, 0.16561338917773344, 5.26112380967508, 11.544570483172807, 11.55438194511509, 5.5446278551129415, 11.55415271974972, 4.438819947934602, 12.102091341963055, 4.710621171250421, 5.355218569819767, 0.9179811619637491, 5.653514861410564, 6.154540244727485, 11.526923763536402, 6.278113560560658, 2.736036208435878, 11.55088205030062, 3.823900732054509]
x_vals_5var = [34.6925504388446 14.141151873774689 11.119011445168764 11.08080982623967 33.346798874453334 24.56151475033733 21.306278552213467 35.10385162217908 27.889964837576986 24.3618997406178 35.76685963541211 17.43327343659165 23.004589657680963 29.043545553301588 26.602118909716253 18.29250237591447 31.537515225361872 36.694661209788016 19.137277800130654 23.469260384839718 10.151475601543307 29.650674003252174 13.404871529268792 12.160209321284379 35.519881617411244 10.592483061424145 20.57334313212654 34.22159255499805 32.78539251478561 31.687849131333017 21.73276018381634 20.41910294634201 15.77870908830227 30.051993913784365 36.438463802593205 33.943512945223944 36.41522680186885 14.03569399514691 35.447685623946015 25.91594885564416 13.436015130347709 38.053423646107404 22.146066162293774 10.920470508904828 29.36662471532775 35.482135369407935 21.550312606242613 11.823717624678677 24.056651652293557 13.183071342007835 19.833826560893336 20.68762135725582 35.87584552187393 29.461902732099574 20.494658817815235 39.55356093137995 35.61184678781933 24.661930936590945 17.713240371685593 25.8053055970471 22.667905480261936 39.76186530764601 19.107492041590763 37.79127327581547 29.967753869294388 19.729112919004983 31.87805588640749 34.46905807458484 36.04631058824618 20.91385267511066 17.027873531706238 26.390164743892882 23.014609936857262 22.91501102705249 10.581676130646363 39.04747303464983 29.275018830063498 21.316186564417706 22.430499820176987 18.41567824513947 26.51495684503582 34.209428366987034 40.0 21.242749464842912 30.994498549559932 35.6560978709805 24.13491640188798 28.701280701206443 10.631827300896008 10.0 10.0 40.0 40.0 10.0 20.943947479020025 10.0 40.0; 12.04526641678037 3.8900108248817085 17.146457107772253 19.7559869738956 5.800499067771768 9.68743515049813 15.479352139574361 16.134113875196704 13.934932419907634 11.638876710887482 25.289790759205285 21.20183778495785 22.29518506856916 25.503111247710724 19.44130813855018 27.598847536892965 18.119731666675648 10.668768902789628 16.42380841611522 14.746145646956442 1.7280619388017264 6.0697656541388705 28.503716795016054 10.508566413220068 7.163200376059657 9.596456770550718 29.188511195689998 26.00959661718847 11.788234205794014 2.6649944762215454 2.2373224571936823 7.087372255159611 23.05811282380155 9.247747210138227 15.721554306005745 3.9781130286352617 18.958479564711165 19.80758761886793 11.713840363556155 19.036797892464612 16.708040866521912 1.0208873298390095 18.43802569616377 9.106642954931587 21.597370293806442 13.154500695746972 4.518023999433719 28.892911203541335 9.942623548462876 2.6675765686679904 4.25270973649415 27.820168809156097 25.188721498833214 22.093994776180875 14.668720142357563 14.356057167866338 5.025543324712591 4.606023670848989 11.532818121810417 6.028816257835819 20.548421704331822 6.408406564802942 20.36853478458226 14.096352910046587 24.648406707691084 3.6624564892490716 6.854388257428047 28.396304300021214 11.782095029105585 1.0028151297985417 10.763298121301142 19.41778218761861 18.595270694303625 18.753516043947275 21.403336423072737 13.484920056839314 3.5779107857997197 10.721023322944538 9.747106224859792 11.653219561393861 11.355522971090812 16.248303222950263 1.0 9.277249072708667 3.7592743833826248 1.0 5.692883371794694 7.343839317574373 10.526235743253242 30.0 8.203243216276247 30.0 11.951049523148894 15.772988620471892 29.790626449140834 30.0 13.896734983189134; 63.94093894618316 79.07508279233092 70.19449799981741 83.65563186913839 68.56399463296665 80.07794926849088 71.35052480250245 69.10845753741178 73.79934949696182 62.284246628294326 65.54958394743679 64.36110547747697 62.27045415437432 69.55982760113707 77.22495181453907 71.83375542632098 66.50089377885763 74.89040824939563 71.84279804301885 66.08832157227405 81.75737654471536 73.49091800290904 72.59565013003095 83.8933413211225 88.4416141309925 88.58495367944073 75.07158675082138 65.57018375787786 66.78363351995664 64.4223436237155 64.76949373899518 61.237163428854274 75.72731810659093 88.33078647168492 61.03603566128841 60.829461838554515 74.663434253052 89.71042234072993 75.09784765234161 74.88150049346832 83.65306617190416 80.19054672484191 83.27093066373826 74.4196575912131 87.44687071192114 61.09886467272664 83.5969832827067 61.04766127206749 87.98517134662909 69.9639528522035 68.82827542635032 75.59877145676776 88.44555689971347 73.34642917925802 61.269145981011235 60.3599433998925 76.39949192999336 88.23203150324436 73.15073261790214 72.52968318689095 69.77385853405661 73.4229171090308 83.61810431985906 74.90370499191026 63.473185646700145 85.43403490722724 89.80846504390784 82.78200532842092 70.70023582840177 86.09614832753485 73.9358878274546 75.98747029516716 82.5096421139157 70.2785395775042 64.841336976501 74.92653395663676 85.15157058710618 83.75226452585603 73.66112815773536 90.0 90.0 90.0 60.0 90.0 90.0 79.45448708498662 90.0 83.57899224746674 90.0 90.0 60.0 90.0 90.0 60.0 71.38356809519529 60.0 60.0; 14.777268408971338 4.1441809588331395 27.34716012192077 28.88188567553844 16.301977234442546 25.36967445474609 5.667229086954997 3.1245970942350394 6.286783757013786 24.618462917629635 3.8756927129288408 5.340666599307309 7.694461111566954 18.885226496828494 21.33332644682954 8.428344645932185 11.555453345697451 13.134247084395373 18.03900389804433 8.154898192733596 3.410138720372147 23.67206840401379 7.284059220688346 26.46757917505253 11.058140567470192 4.936264170395133 10.844181301578752 7.544448737492424 29.647638432627843 15.028303066610224 6.809383327141516 25.10168418465857 26.9994437469655 17.310004110580692 1.6489698586196906 5.086496047166792 26.754829106898352 23.462093020954487 27.19762873354031 2.0868101442653746 5.12575009624558 24.39213652155618 5.756550844023442 25.977857595769677 23.682343303273743 9.314067423845735 20.192498026339855 4.363866866347251 16.061699099950154 20.775498552631355 7.642589956251491 8.184932821413115 22.66182983012378 5.353959997864284 25.01968447320318 7.196736330011332 28.302471671406504 2.2057093853127503 13.575001910683172 16.636825154779157 19.09220650077936 19.678151944620236 12.313976195003534 8.304708957642458 23.09232892045435 12.63734630186532 21.072437845696957 24.86476946380455 7.587957776033578 22.464893083958444 12.825983132601229 24.882953593573028 6.872023817708933 20.117533358823685 5.540010823344057 7.186805020286565 5.8253963529453925 25.564351525592738 9.277915748743512 23.010435790811268 1.0 1.0 17.257880764232443 1.0 1.0 1.0 1.0 1.0 1.0 30.0 30.0 30.0 30.0 1.0 8.1951799548139 30.0 30.0; 37.47833220805155 24.797556896032276 60.31445918583698 71.71790724948227 58.318389456457986 62.19557732491994 55.66078819353905 40.07940388164356 25.01396937453621 42.44803228997699 48.59346463228787 21.54081468766213 46.42967493374713 41.52211673719107 50.95561774249659 42.51615467687345 36.19121205019181 40.12010600493336 38.22874057928705 66.32066233110122 47.65581313868041 76.95907544973636 49.145237008356375 27.62147525177927 25.887083125272497 28.94206749754308 52.25782486786459 76.20536659362332 33.858552167564895 31.038397938050462 70.35098738688907 50.92901838376328 52.27206663857078 79.84492080467172 24.478267090078607 60.75524714332487 43.340801179830464 51.05711771042278 46.6944733096132 65.52810850287327 28.04178042862251 27.325176722660885 74.5538889501612 23.23055943553785 26.687807839779715 51.62363387363267 54.959570629584995 27.984369786939872 23.409185842156738 38.49625021844113 68.74868462001956 61.980918699558316 53.14348670426122 37.2678057575401 56.055692851232436 26.498394335486648 41.570974421189156 44.567242440634345 62.61698897170759 51.0140067091186 67.92623039990133 67.19907950128719 33.91359057420012 77.71795295457875 79.22995073088319 75.7201301383756 58.73288485822815 23.986852616129205 38.36403756219049 59.55751882320149 63.08862542626935 77.13299356470195 75.04381897947984 66.9188054387005 41.69582175890748 78.39253064710275 44.5802575545297 22.066472876681495 74.5474205121602 80.0 80.0 80.0 80.0 66.63517186707011 80.0 80.0 80.0 80.0 80.0 20.0 80.0 80.0 20.0 80.0 48.67323623257573 80.0 74.80540126425058]
y_vals_5var = [3.0406242760316085, 2.384811848924405, 3.8872148442806354, 3.363701993591951, 6.740635579767181, 8.633076607787693, 2.2226868055839217, 1.5241266096969925, 1.9097397620417325, 5.63750496566302, 0.21077898595029163, -0.4966898056346044, -0.6377594644938074, 0.5928120525694253, 2.7756209686115327, -0.7404075584683661, 0.9713937287081377, 5.309028579709205, 1.8395647198954566, 4.630703394291649, 4.258507657068549, 6.741996864117234, -1.1047635803213676, 2.9170064039905683, 7.983898050207942, 2.804869817837126, 0.03596057914506517, -0.685127253533448, 3.658444933189657, 1.9295210201768933, 3.333061412146498, 5.326730373297367, 1.0753618682353119, 10.786387250573048, 2.742458115468455, 7.062892024654798, 2.4852434017494014, 0.0, 4.419749284425611, 2.8316074718622963, 0.3184389615813964, 4.396481938956758, 8.137110471994598, 2.2463062118353085, 2.291662516894049, 2.003405197211167, 7.083666335950684, -0.7918714314659735, 5.926733217803263, 4.00230254837684, 4.505011100337667, -0.49262726568416043, 2.3084965519621634, 0.3450988990759704, 6.412450540342336, 0.9233498999614167, 5.2231217123976315, 7.334421541398417, 9.208820496525817, 7.2981412771719745, 4.527637176488729, 7.683349221128427, 1.2917246179676387, 6.497906393507288, 1.9410380544030394, 7.744735357505569, 0.0, 1.9908294447930923, 3.687165537959401, 4.1981341919386095, 9.537288769678417, 5.715773133524079, 8.146080430751931, 6.0044590785736816, -0.6770743171291433, 6.02411408496835, 2.428607407255862, 4.010546653708594, 8.690191590680902, 10.09269002544873, 16.805101433177548, 5.887774529390494, 1.4839488682339612, 11.804984564568656, 16.928359382080142, 10.510669341245176, 17.941526417730202, 16.947059704892755, 11.918314770389413, -0.047668155572821475, 4.8955226223608, 0.0, 4.309710676711314, 0.8747953765591551, -0.7577031694656682, -0.521389675163371, 4.5583114033199985]
x_vals_2var_10k = [3.625 10.625 14.125 7.125 5.375 12.375 8.875 1.875 2.3125 9.3125 6.504549775298427 7.281079462320841 2.3145685826030187 8.699316286790467 4.2894463430661345 1.8949058007104944 7.0342679367041345 9.801623427931014 6.150544400808255 4.220118168059278 15.0 15.0 1.0 15.0 4.831684373239959 15.0 2.796828645737749 6.272103196907788 1.0 1.0 9.884256986497622 4.906151677260143 12.605101023806473 10.184166349793728 3.4953265661859314 8.195629066180972 6.186518625130257 9.765746228790325 1.0 5.392812758733502 11.803032685954125 4.544786934593868 4.129101190137086 4.097677939493877 8.597122964806513 3.9883495756443637 4.135536235404519 4.139346344922236 4.140372622449566 4.140503463105925 8.052346610713705 13.326547393798654 12.25968626767324 14.477882839583126 5.6593143318753345; 5.375 12.375 1.875 8.875 3.625 10.625 7.125 14.125 7.5625 14.5625 5.552179022822979 11.114256077029788 3.165984710549236 3.86652036575109 1.199025544278957 1.0 1.4168854183536812 1.2403596729539 13.426967333396432 11.40175413242729 5.955950681069222 8.403042841422575 9.986038784350812 15.0 15.0 13.182534920728367 1.8757838938686118 1.0 4.418154840510948 12.378817844841896 9.004323041944762 9.909061642546312 3.8462797386503222 5.493179907326777 1.0 12.901593842708088 7.1681966023714585 2.428612901208271 15.0 2.1530994126261667 15.0 1.0 1.0 1.0 1.0 2.6848525913605243 1.0 1.0 1.0 1.0 12.446807846311495 14.920425894370672 11.414305836213606 13.725337105424671 4.650037817962839]
y_vals_2var_10k = [5.1962615558634635, 4.666416986941233, 2.6957808316429217, 6.142650744764306, 5.94517546552795, 2.903801266694992, 5.068312459089726, 4.059087661321873, 0.0, 4.952205793908455, 5.875689796239696, 5.75843165532398, 5.605886569807023, 5.377855940070088, 7.574798299082679, 6.972419974785948, 6.900860328787445, 5.7035671731050215, 5.556331677890857, 5.603972427171128, 1.54774906071742, 1.9030053304474295, 3.719915271181682, 2.294501490994527, 5.034823037521235, 2.0202648549562103, 6.505763915130686, 7.134147200202139, 4.2128136262399085, 3.0735696982150147, 4.3351268901447595, 5.993209123255962, 2.846197384005066, 4.255808423952203, 7.595776332775121, 5.362996832191748, 6.195847479766792, 5.088839908110157, 3.116914689803957, 6.682887810858866, 3.92283386548021, 7.64871821208746, 7.704684272054273, 7.673146953386399, 6.218214915408527, 6.109102013515681, 7.6722588082787615, 7.669070160138154, 7.674327437675778, 7.645671607887962, 5.436832797905276, 2.934507651415876, 3.235230472038454, 2.249470378669872, 5.978515515375169]
#append!(model,x_2var_10k,y_2var_10k)

opt = BOpt(foil_optim,
           model,
           UpperConfidenceBound(),
           modeloptimizer,                        
           [1.0,1.0], [15.0,15.0],       
           repetitions = 1,
           maxiterations = 100,
           sense = Max,
           initializer_iterations = 10,   
            verbosity = Progress)

result = boptimize!(opt)