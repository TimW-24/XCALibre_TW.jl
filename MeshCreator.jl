using AerofoilOptimisation

chord = 250.0

create_NACA_mesh(
    chord = chord, #[mm]
    α = 0, #[°]
    cutoff = 0.5*(chord/100), #Min thickness of TE [mm]. Default = 0.5 @ 100mm chord; reduce for aerofoils with very thin TE
    vol_size = (16,10), #Total fluid volume size (x,y) in chord multiples [aerofoil located in the vertical centre at the 1/3 position horizontally]
    ratio = 1.2, #Mesh cell size scaling
    BL_thick = 1, #Boundary layer mesh thickness [%c]
    BL_layers = 20, #Boundary layer mesh layers [-]
    BL_stretch = 1.2, #Boundary layer stretch factor (successive multiplication factor of cell thickness away from wall cell) [-]
    py_lines = (14,37,44,248,358,391,405,353), #SALOME python script relevant lines (notebook path, chord line, points line, splines line, BL thickness, foil end BL fidelity, .unv path)
    dat_path = "/home/tim/Documents/MEng Individual Project/Julia/AerofoilOptimisation/foil_dats/ClarkY.dat",
    py_path = "/home/tim/Documents/MEng Individual Project/Julia/AerofoilOptimisation/foil_pythons/NACAMesh.py", #Path to SALOME python script
    salome_path = "/home/tim/Downloads/InstallationFiles/SALOME-9.11.0/mesa_salome", #Path to SALOME installation
    unv_path = "/home/tim/Documents/MEng Individual Project/Julia/XCALibre_TW.jl/unv_sample_meshes/ClarkYMesh.unv", #Path to .unv destination
    note_path = "/home/tim/Documents/MEng Individual Project/SALOME", #Path to SALOME notebook (.hdf) destination
    GUI = true #SALOME GUI selector
)
