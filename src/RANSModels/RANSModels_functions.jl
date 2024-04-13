export StrainRate
export double_inner_product!, double_inner_product2!
export magnitude!, magnitude2!

struct StrainRate{G, GT} <: AbstractTensorField
    gradU::G
    gradUT::GT
end

function (S::StrainRate)(i)
    0.5.*(S.gradU[i] .+ S.gradUT[i])
end

double_inner_product!(
    magS::ScalarField, t1::AbstractTensorField, t2::AbstractTensorField; scale_factor =1.0) = 
begin
    sum = 0.0
    for i ∈ eachindex(magS.values)
        # t1 = t0[i] .- (1/3)*t0[i]*I
        #t1 = 2.0.*t0[i] .- (2/3)*t0[i]*I
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                sum +=   t1[i][j,k]*t2[i][j,k]
                # sum +=   t1[j,k]*t2[i][j,k]
            end
        end
        magS.values[i] = sqrt(sum*scale_factor)
    end
end #For shear rate

double_inner_product!(
    magS::ScalarField, t0::AbstractTensorField, t2) = 
begin
    sum = 0.0
    for i ∈ eachindex(magS)
        # t1 = t0[i] .- (1/3)*t0[i]*I
        t1 = 2.0.*t0[i] .- (2/3)*t0[i]*I
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                sum +=   t1[j,k]*t2[i][k,j]
                # sum +=   t1[j,k]*t2[i][j,k]
            end
        end
        magS[i] = sum
    end
end #For Pk

function magnitude!(magS::ScalarField, S::AbstractTensorField; scale_factor=1.0)
    sum = 0.0
    for i ∈ eachindex(magS.values)
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                # sum +=   S[i][j,k]*S[i][k,j]
                sum +=   S[i][j,k]*S[i][k,j]
            end
        end
        magS.values[i] =   sqrt(sum*scale_factor)
    end
end #For PkL

function magnitude!(magS::ScalarField, S::AbstractVectorField; scale_factor=1.0)
    sum = 0.0
    for i ∈ eachindex(magS.values)
        sum = 0.0
        for j ∈ 1:3
                sum +=   S[i][j]^2
        end
        magS.values[i] =   sqrt(sum*scale_factor)
    end
end #For U

function magnitude2!(
    magS::ScalarField, S::AbstractTensorField; scale_factor=1.0
    )
    sum = 0.0
    for i ∈ eachindex(magS.values)
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                # sum +=   S[i][j,k]*S[i][k,j]
                sum +=   S[i][j,k]*S[i][j,k]
            end
        end
        magS.values[i] = sum*scale_factor
    end
end

bound!(field, bound) = begin
    mesh = field.mesh
    # (; cells, faces) = mesh
    (; cells, cell_neighbours) = mesh
    for i ∈ eachindex(field)
        sum_flux = 0.0
        sum_area = 0
        average = 0.0
        
        # Cell based average
        # cellsID = cells[i].neighbours
        # for cID ∈ cellsID
        for fi ∈ cells[i].faces_range
            cID = cell_neighbours[fi]
            sum_flux += max(field[cID], eps()) # bounded sum
            sum_area += 1
        end
        average = sum_flux/sum_area

        field[i] = max(
            max(
                field[i],
                average*signbit(field[i])
            ),
            bound
        )
    end
end