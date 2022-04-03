export @discretise
export @discretise2

macro discretise(Model_type, nTerms::Integer, nSources::Integer)
    aP! = Expr(:block)
    aN! = Expr(:block)
    b!  = Expr(:block)
    for t ∈ 1:nTerms
        push!(aP!.args, :(aP!(
            model.terms.$(Symbol("term$t")), A, cell, face, nsign, cID)
            ))
        push!(aN!.args, :(aN!(
            model.terms.$(Symbol("term$t")), A, cell, face, nsign, cID, nID)
            ))
        push!(b!.args, :(
            b!(model.terms.$(Symbol("term$t")), b, cell, cID)
            ))
    end 
    
    quote 
        function discretise!(equation, model::$Model_type, mesh)
            (; faces, cells) = mesh
            (; A, b) = equation
            @inbounds for cID ∈ eachindex(cells)
                cell = cells[cID]
                A[cID,cID] = zero(0.0)
                @inbounds for fi ∈ eachindex(cell.facesID)
                    fID = cell.facesID[fi]
                    nsign = cell.nsign[fi]
                    face = faces[fID]
                    nID = cell.neighbours[fi]
                    A[cID,nID] = zero(0.0)
                    $aP!
                    $aN!                    
                end
                b[cID] = zero(0.0)
                $b!
            end
            nothing
        end # end function
    end |> esc # end quote and escape!
end # end macro

macro discretise2(Model_type, nTerms::Integer, nSources::Integer)
    assignment_block_1 = [] #Expr(:block)
    assignment_block_2 = [] #Expr(:block)
    for t ∈ 1:nTerms
        function_call = :(
            scheme!(model.terms.$(Symbol("term$t")), A, cell, face, ns, cID, nID)
            )
        # ap_assignment = :(A[cID, cID] += coeffs[1])
        # an_assignment = :(A[cID, nID] += coeffs[2])
        push!(assignment_block_1, function_call)

        assign_source = :(
            scheme_source!(model.terms.$(Symbol("term$t")), b, cell, cID)
            )
        push!(assignment_block_2, assign_source)
    end 
    
    func = quote 
        function discretise2!(equation, model::$Model_type, mesh)
            (; faces, cells) = mesh
            (; A, b) = equation
            @inbounds for cID ∈ eachindex(cells)
                cell = cells[cID]
                (; facesID, nsign, neighbours) = cell
                A[cID,cID] = zero(0.0)
                @inbounds for fi ∈ eachindex(cell.facesID)
                    fID = cell.facesID[fi]
                    ns = cell.nsign[fi] # normal sign
                    face = faces[fID]
                    nID = cell.neighbours[fi]
                    A[cID,nID] = zero(0.0)
                    $(assignment_block_1...)    
                end
                b[cID] = zero(0.0)
                $(assignment_block_2...)
            end
            nothing
        end # end function
    end |> esc # end quote and escape!
    return func
end # end macro