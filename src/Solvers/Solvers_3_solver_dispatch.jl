export run!

# Incompressible solver (steady)
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config
    ) where{T<:Steady,F<:Incompressible,M,Tu,E,D,BI} = 
begin
    Rx, Ry, Rz, Rp, model = simple!(model, config); #, pref=0.0)
    return Rx, Ry, Rz, Rp, model
end

# Incompressible solver (transient)
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config
    ) where{T<:Transient,F<:Incompressible,M,Tu,E,D,BI} = 
begin
    Rx, Ry, Rz, Rp, model = piso!(model, config); #, pref=0.0)
    return Rx, Ry, Rz, Rp, model
end

# Compressible solver (steady)
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config
    ) where{T<:Steady,F<:Compressible,M,Tu,E,D,BI} = 
begin
    Rx, Ry, Rz, Rp, Re, model = simple_comp!(model, config); #, pref=0.0)
    return Rx, Ry, Rz, Rp, model
end

# # Compressible solver (transient)
# run!(
#     model::Physics{T,F,M,Tu,E,D,BI}, config
#     ) where{T<:Transient,F<:Compressible,M,Tu,E,D,BI} = 
# begin
#     Rx, Ry, Rz, Rp, model = piso_comp!(model, config); #, pref=0.0)
#     return Rx, Ry, Rz, Rp, model
# end