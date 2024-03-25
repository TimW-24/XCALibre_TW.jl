using FVM_1D
using FVM_1D.UNV_3D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")

unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
unv_mesh="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.08m.unv"
unv_mesh="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.06m.unv"
unv_mesh="unv_sample_meshes/3d_streamtube_1.0x0.1x0.1_0.04m.unv"
#unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"


points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
faces
volumes
boundaryElements

mesh=build_mesh3D(unv_mesh)

#Priority
#1) all_cell_faces (unsuccsessful)
#2) face_ownerCells (unsuccsessful)
#3) cell neighbours

@time nodes=generate_nodes(points,volumes) #0.067947 seconds

@time node_cells=generate_node_cells(points,volumes) #0.051790 seconds

@time faces,cell_face_nodes=generate_tet_internal_faces(volumes,faces) #0.065681 seconds
#faces=quad_internal_faces(volumes,faces)

@time face_nodes=Vector{Int}(generate_face_nodes(faces)) #0.014925 seconds
@time cell_nodes=Vector{Int}(generate_cell_nodes(volumes)) #0.011821 seconds

@time all_cell_faces=generate_all_cell_faces(faces,cell_face_nodes) #0.526907 seconds

@time cell_nodes_range=generate_cell_nodes_range(volumes) #0.008669 seconds
@time face_nodes_range=generate_face_nodes_range(faces) #0.011004 seconds
@time all_cell_faces_range=generate_all_faces_range(volumes) #0.010706 seconds

@time cells_centre=calculate_centre_cell(volumes,nodes) #0.026527 seconds

@time boundary_faces,boundary_face_range=generate_boundary_faces(boundaryElements) #0.036406 seconds
@time boundary_cells=generate_boundary_cells(boundary_faces,all_cell_faces,all_cell_faces_range) #0.093407 seconds

@time cell_faces,cell_faces_range=generate_cell_faces(boundaryElements,volumes,all_cell_faces) #0.055045 seconds

@time boundaries=generate_boundaries(boundaryElements,boundary_face_range) #0.009460 seconds 

@time face_ownerCells=generate_face_ownerCells(faces,all_cell_faces,volumes,all_cell_faces_range) #0.535271 seconds

@time faces_area=calculate_face_area(nodes,faces) #0.037004 seconds
@time faces_centre=calculate_face_centre(faces,nodes) #0.026016 seconds
@time faces_normal=calculate_face_normal(nodes,faces,face_ownerCells,cells_centre,faces_centre) #0.050983 seconds 
@time faces_e,faces_delta,faces_weight=calculate_face_properties(faces,face_ownerCells,cells_centre,faces_centre,faces_normal) #0.061823 seconds

@time cells_volume=calculate_cell_volume(volumes,all_cell_faces_range,all_cell_faces,faces_normal,cells_centre,faces_centre,face_ownerCells,faces_area) #0.030618 seconds

@time cells=generate_cells(volumes,cells_centre,cells_volume,cell_nodes_range,cell_faces_range) #0.011763 seconds
@time cell_neighbours=generate_cell_neighbours(cells,cell_faces) #0.497284 seconds
@time faces=generate_faces(faces,face_nodes_range,faces_centre,faces_normal,faces_area,face_ownerCells,faces_e,faces_delta,faces_weight) #0.034309 seconds

@time cell_nsign=calculate_cell_nsign(cells,faces,cell_faces) #0.027957 seconds 

#work

generate_all_cell_faces(faces,cell_face_nodes)

function generate_all_cell_faces(faces,cell_face_nodes)
    all_cell_faces=Int[]
    sorted_faces=Vector{Vector{Int64}}(undef,length(faces))
    for i=1:length(faces)
        sorted_faces[i]=sort(faces[i].faces)
    end

    all_cell_faces=zeros(Int,length(cell_face_nodes)) #May only work for Tet
    for i=1:length(cell_face_nodes)
        all_cell_faces[i]=findfirst(x->x==cell_face_nodes[i],sorted_faces)
    end
    return all_cell_faces
end

@time al2=generate_all_cell_faces_1(faces,cell_face_nodes)



# DEFINE FUNCTIONS
function calculate_face_properties(faces,face_ownerCells,cell_centre,face_centre,face_normal)
    store_e=SVector{3,Float64}[]
    store_delta=Float64[]
    store_weight=Float64[]
    for i=1:length(faces) #Boundary Face
        if face_ownerCells[i,2]==face_ownerCells[i,1]
            cc=cell_centre[face_ownerCells[i,1]]
            cf=face_centre[i]

            d_cf=cf-cc

            delta=norm(d_cf)
            push!(store_delta,delta)
            e=d_cf/delta
            push!(store_e,e)
            weight=one(Float64)
            push!(store_weight,weight)

        else #Internal Face
            c1=cell_centre[face_ownerCells[i,1]]
            c2=cell_centre[face_ownerCells[i,2]]
            cf=face_centre[i]
            d_1f=cf-c1
            d_f2=c2-cf
            d_12=c2-c1

            delta=norm(d_12)
            push!(store_delta,delta)
            e=d_12/delta
            push!(store_e,e)
            weight=abs((d_1f⋅face_normal[i])/(d_1f⋅face_normal[i] + d_f2⋅face_normal[i]))
            push!(store_weight,weight)

        end
    end
    return store_e,store_delta,store_weight
end

function calculate_face_normal(nodes,faces,face_ownerCells,cell_centre,face_centre)
    face_normal=SVector{3,Float64}[]
    for i=1:length(faces)
        n1=nodes[faces[i].faces[1]].coords
        n2=nodes[faces[i].faces[2]].coords
        n3=nodes[faces[i].faces[3]].coords

        t1x=n2[1]-n1[1]
        t1y=n2[2]-n1[2]
        t1z=n2[3]-n1[3]

        t2x=n3[1]-n1[1]
        t2y=n3[2]-n1[2]
        t2z=n3[3]-n1[3]

        nx=t1y*t2z-t1z*t2y
        ny=-(t1x*t2z-t1z*t2x)
        nz=t1x*t2y-t1y*t2x

        magn2=(nx)^2+(ny)^2+(nz)^2

        snx=nx/sqrt(magn2)
        sny=ny/sqrt(magn2)
        snz=nz/sqrt(magn2)

        normal=SVector(snx,sny,snz)
        push!(face_normal,normal)

        if face_ownerCells[i,2]==face_ownerCells[i,1]
            cc=cell_centre[face_ownerCells[i,1]]
            cf=face_centre[i]

            d_cf=cf-cc

            if d_cf⋅face_normal[i]<0
                face_normal[i]=-1.0*face_normal[i]
            end
        else
            c1=cell_centre[face_ownerCells[i,1]]
            c2=cell_centre[face_ownerCells[i,2]]
            cf=face_centre[i]
            #d_1f=cf-c1
            #d_f2=c2-cf
            d_12=c2-c1

            if d_12⋅face_normal[i]<0
                face_normal[i]=-1.0*face_normal[i]
            end
        end
    end
    return face_normal
end

function calculate_face_centre(faces,nodes)
    centre_store=SVector{3,Float64}[]
    for i=1:length(faces)
        face_store=SVector{3,Float64}[]
        for ic=1:length(faces[i].faces)
            push!(face_store,nodes[faces[i].faces[ic]].coords)
        end
        centre=(sum(face_store)/length(face_store))
        push!(centre_store,centre)
    end
    return centre_store
end

function calculate_face_area(nodes,faces)
    area_store=Float64[]
    for i=1:length(faces)
        if faces[i].faceCount==3
            n1=nodes[faces[i].faces[1]].coords
            n2=nodes[faces[i].faces[2]].coords
            n3=nodes[faces[i].faces[3]].coords

            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2
            push!(area_store,area)
        end

        if faces[i].faceCount>3
            n1=nodes[faces[i].faces[1]].coords
            n2=nodes[faces[i].faces[2]].coords
            n3=nodes[faces[i].faces[3]].coords

            t1x=n2[1]-n1[1]
            t1y=n2[2]-n1[2]
            t1z=n2[3]-n1[3]

            t2x=n3[1]-n1[1]
            t2y=n3[2]-n1[2]
            t2z=n3[3]-n1[3]

            area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
            area=sqrt(area2)/2

            for ic=4:faces[i].faceCount
                n1=nodes[faces[i].faces[ic]].coords
                n2=nodes[faces[i].faces[2]].coords
                n3=nodes[faces[i].faces[3]].coords

                t1x=n2[1]-n1[1]
                t1y=n2[2]-n1[2]
                t1z=n2[3]-n1[3]

                t2x=n3[1]-n1[1]
                t2y=n3[2]-n1[2]
                t2z=n3[3]-n1[3]

                area2=(t1y*t2z-t1z*t2y)^2+(t1x*t2z-t1z*t2x)^2+(t1y*t2x-t1x*t2y)^2
                area=area+sqrt(area2)/2

            end

            push!(area_store,area)

        end
    end
    return area_store
end

function generate_cell_faces(boundaryElements,volumes,all_cell_faces)
    cell_faces=Vector{Int}[]
    cell_face_range=UnitRange{Int64}[]
    counter_start=0
    x=0
    max=0

    for ib=1:length(boundaryElements)
        max_store=maximum(boundaryElements[ib].elements)
        if max_store>=max
            max=max_store
        end
    end

    for i=1:length(volumes)
        push!(cell_faces,all_cell_faces[counter_start+1:counter_start+length(volumes[i].volumes)])
        counter_start=counter_start+length(volumes[i].volumes)
        cell_faces[i]=filter(x-> x>max,cell_faces[i])

        if length(cell_faces[i])==1
            push!(cell_face_range,UnitRange(x+1,x+1))
            x=x+1
        else
            push!(cell_face_range,UnitRange(x+1,x+length(cell_faces[i])))
            x=x+length(cell_faces[i])
        end
    end
    cell_faces=reduce(vcat,cell_faces)

    return cell_faces,cell_face_range
end

function calculate_cell_nsign(cells,faces1,cell_faces)
    cell_nsign=Int[]
    for i=1:length(cells)
        centre=cells[i].centre 
        for ic=1:length(cells[i].faces_range)
            fcentre=faces1[cell_faces[cells[i].faces_range][ic]].centre
            fnormal=faces1[cell_faces[cells[i].faces_range][ic]].normal
            d_cf=fcentre-centre
            fnsign=zero(Int)

            if d_cf⋅fnormal > zero(Float64)
                fnsign = one(Int)
            else
                fnsign = -one(Int)
            end
            push!(cell_nsign,fnsign)
        end

    end
    return cell_nsign
end

function calculate_cell_volume(volumes,all_cell_faces_range,all_cell_faces,face_normal,cell_centre,face_centre,face_ownerCells,face_area)
    volume_store = Float64[]
    for i=1:length(volumes)
        volume = zero(Float64) # to avoid type instability
        for f=all_cell_faces_range[i]
            findex=all_cell_faces[f]

            normal=face_normal[findex]
            cc=cell_centre[i]
            fc=face_centre[findex]
            d_fc=fc-cc

            if  face_ownerCells[findex,1] ≠ face_ownerCells[findex,2]
                if dot(d_fc,normal)<0.0
                    normal=-1.0*normal
                end
            end


            volume=volume+(normal[1]*face_centre[findex][1]*face_area[findex])
            
        end
        push!(volume_store,volume)
    end
    return volume_store
end

function calculate_centre_cell(volumes,nodes)
    centre_store=SVector{3,Float64}[]
    for i=1:length(volumes)
        cell_store=typeof(nodes[volumes[1].volumes[1]].coords)[]
        for ic=1:length(volumes[i].volumes)
            push!(cell_store,nodes[volumes[i].volumes[ic]].coords)
        end
        centre=(sum(cell_store)/length(cell_store))
        push!(centre_store,centre)
    end
    return centre_store
end

function generate_cell_neighbours(cells,cell_faces)
    cell_neighbours=Int64[]
    for ID=1:length(cells) 
        for i=cells[ID].faces_range 
            faces=cell_faces[i]
            for ic=1:length(i)
                face=faces[ic]
                index=findall(x->x==face,cell_faces)
                if length(index)==2
                    if i[1]<=index[1]<=i[end]
                        for ip=1:length(cells)
                            if cells[ip].faces_range[1]<=index[2]<=cells[ip].faces_range[end]
                                push!(cell_neighbours,ip)
                            end
                        end
                    end
                    if i[1]<=index[2]<=i[end]
                        for ip=1:length(cells)
                            if cells[ip].faces_range[1]<=index[1]<=cells[ip].faces_range[end]
                                push!(cell_neighbours,ip)
                            end
                        end
                    end
                end
                if length(index)==1
                    x=0
                    push!(cell_neighbours,x)
                end
            end
        end
    end
    return cell_neighbours
end

function generate_tet_internal_faces(volumes,faces)
    cell_face_nodes=Vector{Int}[]

    for i=1:length(volumes)
        cell_faces=zeros(Int,4,3)
        cell=sort(volumes[i].volumes)

        cell_faces[1,1:3]=cell[1:3]
        cell_faces[2,1:2]=cell[1:2]
        cell_faces[2,3]=cell[4]
        cell_faces[3,1]=cell[1]
        cell_faces[3,2:3]=cell[3:4]
        cell_faces[4,1:3]=cell[2:4]

        for ic=1:4
            push!(cell_face_nodes,cell_faces[ic,:])
        end
    end

    sorted_faces=Vector{Int}[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end

    internal_faces=setdiff(cell_face_nodes,sorted_faces)

    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces, cell_face_nodes
end

function quad_internal_faces(volumes,faces)
    store_cell_faces1=Int64[]

    for i=1:length(volumes)
        cell_faces=zeros(Int,6,4)

        cell_faces[1,1:4]=volumes[i].volumes[1:4]
        cell_faces[2,1:4]=volumes[i].volumes[5:8]
        cell_faces[3,1:2]=volumes[i].volumes[1:2]
        cell_faces[3,3:4]=volumes[i].volumes[5:6]
        cell_faces[4,1:2]=volumes[i].volumes[3:4]
        cell_faces[4,3:4]=volumes[i].volumes[7:8]
        cell_faces[5,1:2]=volumes[i].volumes[2:3]
        cell_faces[5,3:4]=volumes[i].volumes[6:7]
        cell_faces[6,1]=volumes[i].volumes[1]
        cell_faces[6,2]=volumes[i].volumes[4]
        cell_faces[6,3]=volumes[i].volumes[5]
        cell_faces[6,4]=volumes[i].volumes[8]

        for ic=1:6
            push!(store_cell_faces1,cell_faces[ic,:])
        end
    end

    sorted_cell_faces=Int64[]
    for i=1:length(store_cell_faces1)

        push!(sorted_cell_faces,sort(store_cell_faces1[i]))
    end

    sorted_faces=Int64[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end

    internal_faces=setdiff(sorted_cell_faces,sorted_faces)

    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces
end


function generate_boundaries(boundaryElements,boundary_face_range)
    boundaries=Boundary{Symbol, UnitRange{Int64}}[]
    for i=1:length(boundaryElements)
        push!(boundaries,Boundary(Symbol(boundaryElements[i].name),boundary_face_range[i]))
    end
    return boundaries
end

function generate_boundary_cells(boundary_faces,cell_faces,cell_faces_range)
    boundary_cells = Int64[]
    store = Int64[]
    for ic=1:length(boundary_faces)
        for i in eachindex(cell_faces)
                if cell_faces[i]==boundary_faces[ic]
                    push!(store,i)
                end
        end
    end
    store
    
    for ic=1:length(store)
        for i=1:length(cell_faces_range)
            if cell_faces_range[i][1]<=store[ic]<=cell_faces_range[i][end]
                push!(boundary_cells,i)
            end
        end
    end
    return boundary_cells
end

function generate_boundary_faces(boundaryElements)
    boundary_faces=Int64[]
    z=0
    wipe=Int64[]
    boundary_face_range=UnitRange{Int64}[]
    for i=1:length(boundaryElements)
        for n=1:length(boundaryElements[i].elements)
            push!(boundary_faces,boundaryElements[i].elements[n])
            push!(wipe,boundaryElements[i].elements[n])
        end
        if length(wipe)==1
            push!(boundary_face_range,UnitRange(boundaryElements[i].elements[1],boundaryElements[i].elements[1]))
            z=z+1
        elseif length(wipe) ≠1
            push!(boundary_face_range,UnitRange(boundaryElements[i].elements[1],boundaryElements[i].elements[end]))
            z=z+length(wipe)
        end
        wipe=Int64[]
    end
    return boundary_faces,boundary_face_range
end

function generate_face_ownerCells(faces,all_cell_faces,volumes,all_cell_faces_range)
    x=Vector{Int64}[]
    for i=1:length(faces)
        push!(x,findall(x->x==i,all_cell_faces))
    end
    y=zeros(Int,length(x),2)
    for ic=1:length(volumes)
        for i=1:length(x)
            #if length(x[i])==1
                if all_cell_faces_range[ic][1]<=x[i][1]<=all_cell_faces_range[ic][end]
                    y[i,1]=ic
                    y[i,2]=ic
                end
            #end

            if length(x[i])==2
                if all_cell_faces_range[ic][1]<=x[i][2]<=all_cell_faces_range[ic][end]
                    #y[i]=ic
                    y[i,2]=ic

                end
            end

        end
    end
    return y
end

function generate_nodes(points,volumes)
    nodes=Node{SVector{3,Float64}, UnitRange{Int64}}[]
    cells_range=nodes_cells_range!(points,volumes)
    @inbounds for i ∈ 1:length(points)
        #point=points[i].xyz
        push!(nodes,Node(points[i].xyz,cells_range[i]))
    end
    return nodes
end

#Generate Faces
function generate_face_nodes(faces)
    face_nodes=typeof(faces[1].faces[1])[] # Giving type to array constructor
    for n=1:length(faces)
        for i=1:faces[n].faceCount
            push!(face_nodes,faces[n].faces[i])
        end
    end
    return face_nodes
end

#Generate cells
function generate_cell_nodes(volumes)
    cell_nodes=typeof(volumes[1].volumes[1])[] # Giving type to array constructor
    for n=1:length(volumes)
        for i=1:volumes[n].volumeCount
            push!(cell_nodes,volumes[n].volumes[i])
        end
    end
    return cell_nodes
end

function generate_all_cell_faces(faces,cell_face_nodes)
    all_cell_faces=Int[]
    sorted_faces=Vector{Int}[]
    for i=1:length(faces)
        push!(sorted_faces,sort(faces[i].faces))
    end

    for i=1:length(cell_face_nodes)
        push!(all_cell_faces,findfirst(x -> x==cell_face_nodes[i],sorted_faces))
    end
    return all_cell_faces
end

#Nodes Range
function generate_cell_nodes_range(volumes)
    cell_nodes_range=UnitRange(0,0)
    store=typeof(cell_nodes_range)[]
    x=0
    for i=1:length(volumes)
        cell_nodes_range=UnitRange(x+1,x+length(volumes[i].volumes))
        x=x+length(volumes[i].volumes)
        push!(store,cell_nodes_range)
    end
    return store
end

function generate_face_nodes_range(faces)
    face_nodes_range=UnitRange(0,0)
    store=typeof(face_nodes_range)[]
    x=0
    for i=1:length(faces)
        face_nodes_range=UnitRange(x+1,x+faces[i].faceCount)
        x=x+faces[i].faceCount
        push!(store,face_nodes_range)
    end
    return store
end

function generate_all_faces_range(volumes)
    cell_faces_range=UnitRange(0,0)
    store=typeof(cell_faces_range)[]
    x=0
    @inbounds for i=1:length(volumes)
        #Tetra
        if length(volumes[i].volumes)==4
            cell_faces_range=UnitRange(x+1,x+4)
            x=x+4
            push!(store,cell_faces_range)
        end

        #Hexa
        if length(volumes[i].volumes)==8
                cell_faces_range=UnitRange(x+1,x+6)
                x=x+6
                push!(store,cell_faces_range)
        end
    end
    return store
end

#Generate cells
function generate_cells(volumes,centre_of_cells,volume_of_cells,cell_nodes_range,cell_faces_range)
    cells=Cell{Float64,SVector{3,Float64},UnitRange{Int64}}[]
    for i=1:length(volumes)
        push!(cells,Cell(centre_of_cells[i],volume_of_cells[i],cell_nodes_range[i],cell_faces_range[i]))
    end
    return cells
end

function generate_faces(faces,face_nodes_range,faces_centre,faces_normal,faces_area,face_ownerCells,faces_e,faces_delta,faces_weight)
    faces3D=Face3D{Float64,SVector{2,Int64},SVector{3,Float64},UnitRange{Int64}}[]
    for i=1:length(faces)
        push!(faces3D,Face3D(face_nodes_range[i],SVector(face_ownerCells[i,1],face_ownerCells[i,2]),faces_centre[i],faces_normal[i],faces_e[i],faces_area[i],faces_delta[i],faces_weight[i]))
    end
    return faces3D
end

#Node connectivity

function nodes_cells_range!(points,volumes)
    neighbour=Int64[]
    wipe=Int64[]
    cells_range=UnitRange{Int64}[]
    x=0
    @inbounds for in=1:length(points)
        @inbounds for iv=1:length(volumes)
            @inbounds for i=1:length(volumes[iv].volumes)
                if volumes[iv].volumes[i]==in
                    neighbour=iv
                    push!(wipe,neighbour)
                    
                end
                continue
                
            end
        end
        if length(wipe)==1
            #cells_range[in]=UnitRange(x+1,x+1)
            push!(cells_range,UnitRange(x+1,x+1))
            x=x+1
        elseif length(wipe) ≠1
            #cells_range[in]=UnitRange(x+1,x+length(wipe))
            push!(cells_range,UnitRange(x+1,x+length(wipe)))
            x=x+length(wipe)
        end
        #push!(mesh.nodes[in].cells_range,cells_range)
        wipe=Int64[]
    end
    return cells_range
end

function generate_node_cells(points,volumes)
    neighbour=Int64[]
    store=Int64[]
    @inbounds for in=1:length(points)
        @inbounds for iv=1:length(volumes)
            @inbounds for i=1:length(volumes[iv].volumes)
                if volumes[iv].volumes[i]==in
                    neighbour=iv
                    push!(store,neighbour)
                end
                continue
            end
        end
    end
    return store
end