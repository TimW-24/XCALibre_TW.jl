@define_boundary Neumann Laplacian{Linear} begin
    phi = term.phi 
    values = get_values(phi, component)
    J = term.flux[fID]
    (; area, delta) = face 

    flux = J*area/delta
    ap = term.sign[1]*(-flux)

    # ap, ap*values[cellID] # original
    0.0, 0.0 
end

@define_boundary Neumann Divergence{Linear} begin
    # original
    ap = term.sign[1]*(term.flux[fID]) 
    ap, 0.0 # original

    # test
    # phi = term.phi 
    # values = get_values(phi, component)
    # ap = 0.5*term.sign[1]*(term.flux[fID])

    # ap, -ap*values[cellID]
end

@define_boundary Neumann Divergence{Upwind} begin
    # original
    # ap = term.sign[1]*(term.flux[fID]) 
    # ap, 0.0 # original

    # test
    phi = term.phi 
    values = get_values(phi, component)
    ap = 0.5*term.sign[1]*(term.flux[fID])

    ap, -ap*values[cellID]
end

@define_boundary Neumann Divergence{BoundedUpwind} begin
    phi = term.phi 
    ap = term.sign[1]*(term.flux[fID])
    vol = 1#cell.volume

    ap-term.flux[fID], 0.0
end

@define_boundary Neumann Si begin
    # nothing
    0.0, 0.0
end