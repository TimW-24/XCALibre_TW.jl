export AbstractScheme, AbstractBoundary
export AbstractDirichlet, AbstractNeumann
export Dirichlet, Neumann, Wall
export KWallFunction, OmegaWallFunction, NutWallFunction
export Constant, Linear, Upwind
export Steady, Euler, CrankNicolson
export Orthogonal, Midpoint
export assign, @assign!
export set_schemes

# SUPPORTED DISCRETISATION SCHEMES 

abstract type AbstractScheme end
struct Constant <: AbstractScheme end
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end
struct Orthogonal <: AbstractScheme end
struct Midpoint <: AbstractScheme end
struct Steady <: AbstractScheme end 
struct Euler <: AbstractScheme end 
struct CrankNicolson <: AbstractScheme end



# SUPPORTED BOUNDARY CONDITIONS 

abstract type AbstractBoundary end
abstract type AbstractDirichlet <: AbstractBoundary end
abstract type AbstractNeumann <: AbstractBoundary end

struct Dirichlet{I,V} <: AbstractBoundary
    ID::I
    value::V
end

function fixedValue(BC::Dirichlet, ID::I, value::V) where {I<:Integer,V}
    if V <: Number
        return Dirichlet{I,eltype(value)}(ID, value)
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Dirichlet{I,typeof(nvalue)}(ID, nvalue)
        else
            throw("Only vectors with three components can be used")
        end
    else
        throw("The value provided should be a scalar or a vector")
    end
end

struct Neumann{I,V} <: AbstractBoundary
    ID::I 
    value::V 
end

function fixedValue(BC::Neumann, ID::I, value::V) where {I<:Integer,V}
    if V <: Number
        return Neumann{I,eltype(value)}(ID, value)
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Neumann{I,typeof(nvalue)}(ID, nvalue)
        else
            throw("Only vectors with three components can be used")
        end
        throw("The value provided should be a scalar or a vector")
    end
end

struct Wall{I,V} <: AbstractBoundary
    ID::I 
    value::V 
end

function fixedValue(BC::Wall, ID::I, value::V) where {I<:Integer,V}
    if V <: Number
        return Wall{I,eltype(value)}(ID, value)
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Wall{I,typeof(nvalue)}(ID, nvalue)
        else
            throw("Only vectors with three components can be used")
        end
        throw("The value provided should be a scalar or a vector")
    end
end

struct KWallFunction{I,V} <: AbstractBoundary
    ID::I 
    value::V 
end
KWallFunction(name::Symbol) = begin
    KWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end

struct OmegaWallFunction{I,V} <: AbstractBoundary
    ID::I 
    value::V 
end
OmegaWallFunction(name::Symbol) = begin
    OmegaWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end

struct NutWallFunction{I,V} <: AbstractBoundary 
    ID::I 
    value::V 
end
NutWallFunction(name::Symbol) = begin
    NutWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end

assign(vec::VectorField, model, args...) = begin
    float = _get_float(vec.mesh)
    boundaries = vec.mesh.boundaries
    @reset vec.x.BCs = ()
    @reset vec.y.BCs = ()
    @reset vec.z.BCs = ()
    @reset vec.BCs = ()
    for arg ∈ args
        # bc_type = Base.typename(typeof(arg)).wrapper #NOT NEEDED WITH MULTIPLE DISPATCH OF VIXED VALUE
        # a = typeof(arg)
        # println("$a")
        # idx = boundary_index(boundaries, arg.ID)
        # idx = @time begin get(model.boundary_info,arg.ID,nothing) end
        idx = boundary_index(model.boundary_info, arg.ID)
        bname = boundaries[idx].name
        println("Setting boundary $idx: ", bname)
        if typeof(arg.value) <: AbstractVector
            length(arg.value) == 3 || throw("Vector must have 3 components")
            # println("",typeof(bc_type))
            xBCs = (vec.x.BCs..., fixedValue(arg, idx, float(arg.value[1])))
            yBCs = (vec.y.BCs..., fixedValue(arg, idx, float(arg.value[2])))
            zBCs = (vec.z.BCs..., fixedValue(arg, idx, float(arg.value[3])))
            uBCs = (vec.BCs..., fixedValue(arg, idx, float.(arg.value)))
            @reset vec.x.BCs = xBCs
            @reset vec.y.BCs = yBCs
            @reset vec.z.BCs = zBCs
            @reset vec.BCs = uBCs
        else
            xBCs = (vec.x.BCs..., fixedValue(arg, idx, float(arg.value)))
            yBCs = (vec.y.BCs..., fixedValue(arg, idx, float(arg.value)))
            zBCs = (vec.z.BCs..., fixedValue(arg, idx, float(arg.value)))
            uBCs = (vec.BCs..., fixedValue(arg, idx, float(arg.value)))
            @reset vec.x.BCs = xBCs
            @reset vec.y.BCs = yBCs
            @reset vec.z.BCs = zBCs
            @reset vec.BCs = uBCs
        end
    end
    return vec
end

assign(scalar::ScalarField, model, args...) = begin
    float = _get_float(scalar.mesh)
    boundaries = scalar.mesh.boundaries
    @reset scalar.BCs = ()
    for arg ∈ args
        # bc_type = Base.typename(typeof(arg)).wrapper
        # idx = boundary_index(boundaries, arg.ID) #returns index number of mesh boundary with same name as boundary condition ID
        # idx = @time begin get(model.boundary_info,arg.ID,nothing) end
        idx = boundary_index(model.boundary_info, arg.ID)
        bname = boundaries[idx].name
        println("Setting boundary $idx: ", bname)

        # Exception 1: value is a number
        if typeof(arg.value) <: Number
            BCs = (fixedValue(arg, idx, float(arg.value))) # doesn't work with tuples
            @reset scalar.BCs = (scalar.BCs..., BCs)

        # Exception 2: value is a named tuple (used in wall functions)
        elseif typeof(arg.value) <: NamedTuple
            BCs_vals = arg.value
            for entry ∈ typeof(arg.value).parameters[1] # access names
                val = float(getproperty(arg.value, entry)) # type conversion
                BCs_vals = set(BCs_vals, PropertyLens{entry}(), val)
            end
            BCs = (fixedValue(arg, idx, BCs_vals))
            @reset scalar.BCs = (scalar.BCs..., BCs)
        else
            error("Value given to boundary $idx ($bname) is not recognised")
        end
    end
    return scalar
end

macro assign!(model, field, BCs)
    emodel = esc(model)
    efield = Symbol(field)
    eBCs = esc(BCs)
    # esymbol_mapping = esc(symbol_mapping)
    quote
        f = $emodel.$efield
        f = assign(f, $emodel, $eBCs...)
        $emodel = @set $emodel.$efield = f
    end
end

macro assign!(model, turb, field, BCs)
    emodel = esc(model)
    eturb = Symbol(turb)
    efield = Symbol(field)
    eBCs = esc(BCs)
    # esymbol_mapping = esc(symbol_mapping)
    quote
        f = $emodel.$eturb.$efield
        f = assign(f, $emodel, $eBCs...)
        # @reset $emodel.$eturb.$efield = f
        $emodel = @set $emodel.$eturb.$efield = f
    end
end

set_schemes(;
    time=Steady,
    divergence=Linear, 
    laplacian=Linear, 
    gradient=Orthogonal) = begin
    (
        time=time,
        divergence=divergence,
        laplacian=laplacian,
        gradient=gradient
    )
end