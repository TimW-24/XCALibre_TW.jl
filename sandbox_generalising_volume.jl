using FVM_1D.UNV_3D
using StaticArrays
using Statistics
using LinearAlgebra

unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/800_cell_new_boundaries.unv"
#unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"

points,edges,faces,volumes,boundaryElements=load_3D(unv_mesh)

points
edges
faces
volumes
boundaryElements

#mesh=build_mesh3D(unv_mesh)



struct Node{SV3<:SVector{3,<:AbstractFloat}, UR<:UnitRange{<:Integer}}
    coords::SV3
    cells_range::UR # to access neighbour cells (can be dummy entry for now)
end

function generate_nodes(points,volumes)
    nodes=Node[]
    cells_range=nodes_cells_range!(points,volumes)
    @inbounds for i ∈ 1:length(points)
        #point=points[i].xyz
        push!(nodes,Node(points[i].xyz,cells_range[i]))
    end
    return nodes
end

function nodes_cells_range!(points,volumes)
    neighbour=Int64[]
    wipe=Int64[]
    cells_range=UnitRange[]
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

nodes=generate_nodes(points,volumes)

function generate_internal_faces(volumes,faces)
    store_cell_faces=Int[]
    store_faces=Int[]
    
    for i=1:length(volumes)
    cell=sort(volumes[i].volumes)
    push!(store_cell_faces,cell[1],cell[2],cell[3])
    push!(store_cell_faces,cell[1],cell[2],cell[4])
    push!(store_cell_faces,cell[1],cell[3],cell[4])
    push!(store_cell_faces,cell[2],cell[3],cell[4])
    end
    
    for i=1:length(faces)
        face=sort(faces[i].faces)
        push!(store_faces,face[1],face[2],face[3])
    end

    range=[]

    x=0
    for i=1:length(store_cell_faces)/3
        store=UnitRange(x+1:x+3)
        x=x+3
        push!(range,store)
    end

    faces1=[]

    for i=1:length(range)
        face1=store_cell_faces[range[i]]
        for ic=1:length(range)
            face2=store_cell_faces[range[ic]]
            store=[]
    
            push!(store,face2[1] in face1)
            push!(store,face2[2] in face1)
            push!(store,face2[3] in face1)
    
            count1=count(store)
    
            if count1!=3
                push!(faces1,face1)
            end
        end
    end

    all_faces=unique(faces1)

    store1_faces=[]
    for i=1:length(faces)
        push!(store1_faces,sort(faces[i].faces))
    end

    all_faces=sort(all_faces)
    store1_faces=sort(store1_faces)
    
    internal_faces=setdiff(all_faces,store1_faces)
    
    for i=1:length(internal_faces)
        push!(faces,UNV_3D.Face(faces[end].faceindex+1,faces[end].faceCount,internal_faces[i]))
    end
    return faces
end

faces=generate_internal_faces(volumes,faces)

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

face_area=calculate_face_area(nodes,faces)


function calculate_face_centre(faces,nodes)
    centre_store=[]
    for i=1:length(faces)
        face_store=[]
        for ic=1:length(faces[i].faces)
            push!(face_store,nodes[faces[i].faces[ic]].coords)
        end
        centre=(sum(face_store)/length(face_store))
        push!(centre_store,centre)
    end
    return centre_store
end

face_centre=calculate_face_centre(faces,nodes)

# normal_store=[]
# for i=1:length(faces)
#     n1=nodes[faces[i].faces[1]].coords
#     n2=nodes[faces[i].faces[2]].coords
#     n3=nodes[faces[i].faces[3]].coords

#     t1x=n2[1]-n1[1]
#     t1y=n2[2]-n1[2]
#     t1z=n2[3]-n1[3]

#     t2x=n3[1]-n1[1]
#     t2y=n3[2]-n1[2]
#     t2z=n3[3]-n1[3]

#     nx=t1y*t2z-t1z*t2y
#     ny=-(t1x*t2z-t1z*t2x)
#     nz=t1x*t2y-t2y*t2x

#     magn2=(nx)^2+(ny)^2+(nz)^2

#     snx=nx/sqrt(magn2)
#     sny=ny/sqrt(magn2)
#     snz=nz/sqrt(magn2)

#     normal=SVector(snx,sny,snz)
#     push!(normal_store,normal)
# end
# normal_store


function calculate_face_normal(faces,nodes)
    normal_store=[]
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
        push!(normal_store,normal)
    end
    return normal_store
end

face_normal=calculate_face_normal(faces,nodes)


function flip_face_normal(faces,face_ownerCells,cell_centre,face_centre,face_normal)
    for i=1:length(faces)
        if face_ownerCells[i,2]==face_ownerCells[i,1]
            cc=cell_centre[face_ownerCells[i,1]]
            cf=face_centre[i]

            d_cf=cf-cc

            if d_cf⋅face_normal[i]<0.0
                face_normal[i]=-1.0*face_normal[i]
            end
        else
            c1=cell_centre[face_ownerCells[i,1]]
            c2=cell_centre[face_ownerCells[i,2]]
            d_12=c2-c1

            if d_12⋅face_normal[i]<0.0
                face_normal[i]=-1.0*face_normal[i]
            end
        end
    end
    return face_normal
end

face_normal=flip_face_normal(faces,face_ownerCells,cell_centre,face_centre,face_normal)


# store_e=[]
# store_delta=[]
# store_weight=[]
# for i=1:length(faces)
#     if face_ownerCells[i,2]==face_ownerCells[i,1]
#         cc=cell_centre[face_ownerCells[i,1]]
#         cf=face_centre[i]

#         d_cf=cf-cc

#         delta=norm(d_cf)
#         push!(store_delta,delta)
#         e=d_cf/delta
#         push!(store_e,e)
#         weight=one(Float64)
#         push!(store_weight,weight)

#     else
#         c1=cell_centre[face_ownerCells[i,1]]
#         c2=cell_centre[face_ownerCells[i,2]]
#         cf=face_centre[i]
#         d_1f=cf-c1
#         d_f2=c2-cf
#         d_12=c2-c1

#         delta=norm(d_12)
#         push!(store_delta,delta)
#         e=d_12/delta
#         push!(store_e,e)
#         weight=abs((d_1f⋅face_normal[i])/(d_1f⋅face_normal[i] + d_f2⋅face_normal[i]))
#         push!(store_weight,weight)

#     end
# end

function calculate_face_properties(faces,face_ownerCells,cell_centre,face_centre,face_normal)
    store_e=[]
    store_delta=[]
    store_weight=[]
    for i=1:length(faces)
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

        else
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

face_properties=calculate_face_properties(faces,face_ownerCells,cell_centre,face_centre,face_normal)


function calculate_centre_cell(volumes,nodes)
    centre_store=[]
    for i=1:length(volumes)
        cell_store=[]
        for ic=1:length(volumes[i].volumes)
            push!(cell_store,nodes[volumes[i].volumes[ic]].coords)
        end
        centre=(sum(cell_store)/length(cell_store))
        push!(centre_store,centre)
    end
    return centre_store
end

cell_centre=calculate_centre_cell(volumes,nodes)


function generate_all_cell_faces(volumes,faces)
    cell_faces=[]
    for i=1:length(volumes)
        for ic=1:length(faces)
            bad=sort(volumes[i].volumes)
            good=sort(faces[ic].faces)
            store=[]

            push!(store,good[1] in bad)
            push!(store,good[2] in bad)
            push!(store,good[3] in bad)

            if store[1:3] == [true,true,true]
                push!(cell_faces,faces[ic].faceindex)
            end
            continue
        end
    end
    return cell_faces
end

all_cell_faces=Vector{Int}(generate_all_cell_faces(volumes,faces))


function generate_all_faces_range(volumes,faces)
    cell_faces_range=UnitRange(0,0)
    store=[]
    x=0
    @inbounds for i=1:length(volumes)
        #Tetra
        if length(volumes[i].volumes)==4
            #cell_faces_range=UnitRange(faces[(4*i)-3].faceindex,faces[4*i].faceindex)
            cell_faces_range=UnitRange(x+1,x+length(volumes[i].volumes))
            x=x+length(volumes[i].volumes)
            push!(store,cell_faces_range)
        end

        #Hexa
        if length(volumes[i].volumes)==8
                cell_faces_range=UnitRange(faces[6*i-5].faceindex,faces[6*i].faceindex)
                push!(store,cell_faces_range)
        end

        #wedge
        if length(volumes[i].volumes)==6
                cell_faces_range=UnitRange(faces[5*i-4].faceindex,faces[5*i].faceindex)
                push!(store,cell_faces_range)
        end
    end
    return store
end

all_cell_faces_range=generate_all_faces_range(volumes,faces)

function generate_face_ownerCells(faces,all_cell_faces,volumes,all_cell_faces_range)
    x=[]
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

face_ownerCells=generate_face_ownerCells(faces,all_cell_faces,volumes,all_cell_faces_range)


volume_store=[]
volume=0
for i=1:length(volumes)
    volume=0
    for f=all_cell_faces_range[i]
        findex=all_cell_faces[f]

        normal=face_normal[findex]

        volume=volume+(normal[1]*face_centre[findex][1]*face_area[findex])
    end
    push!(volume_store,volume)
end

volume_store

bfaceindex=0
for i=1:length(boundaryElements)
    bfaceindex=maximum(boundaryElements[i].elements)
end
bfaceindex

volume=0
for f=all_cell_faces_range[1]
    findex=all_cell_faces[f]

    normal=face_normal[findex]
    cc=cell_centre[1]

    if  findex>bfaceindex && face_ownerCells[findex,1] ≠ face_ownerCells[findex,2]
        if dot(cc,normal)<0.0
            normal=-1.0*normal
        end
    end


    volume=volume+(normal[1]*face_centre[findex][1]*face_area[findex])
end

volume

all_cell_faces
normal=face_normal[13]
cc=cell_centre[1]

if dot(cc,normal)<0.0
    normal=-1.0*normal
end

