using FVM_1D 

macro define_boundary(boundary, operator, definition)
    quote
        @inline (bc::$boundary)(
            term::Operator{F,P,I,$operator}, cellID, zcellID, cell, face, fID, i, component=nothing
            ) where {F,P,I} = $definition
    end |> esc
end

@macroexpand @define_boundary Neumann Laplacian{Linear} begin
    (; T, energy_model) = bc.value
    h = energy_model.update_BC(T)
    ap = term.sign[1]*(term.flux[fID])
    0.0, -ap*h
end

