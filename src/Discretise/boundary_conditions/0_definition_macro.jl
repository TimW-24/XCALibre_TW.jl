macro define_boundary(boundary, operator, definition)
    quote
        @inline (bc::$boundary)(
            term::Operator{F,P,I,$operator}, rowval, colptr, nzval, b, cellID, zcellID, cell, face, fID, i, component=nothing
            ) where {F,P,I} = $definition
    end |> esc
end