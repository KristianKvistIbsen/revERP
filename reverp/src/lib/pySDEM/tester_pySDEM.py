from ansys.dpf import core as dpf
import reverp.src.lib.hansen_dpf_functions as hdpf
import numpy as np
import pyvista as pv
from pySDEM import pySDEM
import trimesh

# Helper method to convert PyVista to DPF mesh (based on MeshHandler.stl_to_dpf_mesh)
def _pv_to_dpf_mesh(pv_grid):
    coordinates = pv_grid.points
    connectivity = pv_grid.cells_dict[5]  # Assuming triangle elements

    meshed_region = dpf.MeshedRegion(
        num_nodes=coordinates.shape[0],
        num_elements=connectivity.shape[0]
    )

    for i, coord in enumerate(coordinates):
        meshed_region.nodes.add_node(i, coord)

    for i, face in enumerate(connectivity):
        meshed_region.elements.add_element(i, "shell", face.tolist())

    return meshed_region

wrap = r"C:\Users\105849\Desktop\scala_wrap.stl"
path = r"C:\Users\105849\Desktop\file.rst"

# Load model and get skin mesh
model = dpf.Model(path)
skin_mesh1 = hdpf.get_skin_mesh_from_ns("FSI", model)

# Decimate mesh
op = dpf.operators.mesh.decimate_mesh()
op.inputs.mesh.connect(skin_mesh1)
op.inputs.preservation_ratio.connect(float(1))
skin_mesh = op.outputs.mesh()

# Get original mesh data
skin_mesh_grid = skin_mesh.grid
v = skin_mesh_grid.points
f = skin_mesh_grid.cells_dict
f = f[list(f.keys())[0]]

# Handle vertex mapping
v_used = np.unique(f)
v_map = np.zeros(len(v), dtype=int)
v_map[v_used] = np.arange(len(v_used))
f = v_map[f]
original_v = v[v_used]
original_f = f

# Check if mesh is genus-0
is_genus_zero = len(original_v) - 3*len(original_f)/2 + len(original_f) == 2

if not is_genus_zero:
    # Load wrap mesh when original isn't genus-0
    wrap_mesh = trimesh.load(wrap, file_type='stl')
    v = wrap_mesh.vertices / 1000  # Convert mm to m
    f = wrap_mesh.faces
    wrap_areas = wrap_mesh.area_faces
    print("Using shrink-wrapped mesh as input is not genus-0")
else:
    wrap_areas = np.ones(len(f))
    print("Using original mesh (genus-0 detected)")

# Create original grid (before SDEM)
original_grid = pv.UnstructuredGrid({5: original_f}, original_v)

# Create wrap grid
wrap_grid = pv.UnstructuredGrid({5: f}, v)

# Perform SDEM
population = wrap_areas if not is_genus_zero else np.ones(len(f))
try:
    S = pySDEM(v, f, population, dt=1, epsilon=1e-3, max_iter=0)
except ValueError as e:
    print(f"SDEM failed: {str(e)}")
    raise

# Create morphed mesh with S coordinates
morphed_v = S * np.sqrt(skin_mesh_grid.area / (4 * np.pi))  # Scale to preserve area
morphed_grid = pv.UnstructuredGrid({5: f}, morphed_v)

# Get field data
tfreq = model.metadata.time_freq_support.time_frequencies
nodal_normals = hdpf.get_normals(skin_mesh)
skin_mesh_scoping = hdpf.get_scoping_from_mesh(skin_mesh, "Nodal")
velocity_fc = hdpf.get_normal_velocitiy_fc_from_skin_mesh(model, skin_mesh, tfreq, skin_mesh_scoping, nodal_normals).eval()

# Map velocity to wrapped mesh (which might be the original topology if genus-0)
mapping_op = dpf.operators.mapping.prepare_mapping_workflow()
mapping_op.inputs.input_support.connect(skin_mesh)
mapping_op.inputs.output_support.connect(_pv_to_dpf_mesh(wrap_grid))  # Convert PyVista to DPF mesh
mapping_workflow = mapping_op.outputs.mapping_workflow()
mapping_workflow.connect('source', velocity_fc)
mapped_velocity_fc = mapping_workflow.get_output('target', output_type="fields_container")

# Export all three meshes with velocity field
# 1. Original skin mesh
hdpf.export_field_to_vtk(
    _pv_to_dpf_mesh(original_grid),
    velocity_fc,
    r"C:\Users\105849\Desktop\original_skin_mesh.vtk"
)

# 2. Wrapped mesh (or original topology if genus-0)
hdpf.export_field_to_vtk(
    _pv_to_dpf_mesh(wrap_grid),
    mapped_velocity_fc,
    r"C:\Users\105849\Desktop\wrapped_mesh.vtk"
)

# 3. Morphed mesh with S coordinates
hdpf.export_field_to_vtk(
    _pv_to_dpf_mesh(morphed_grid),
    mapped_velocity_fc,  # Use mapped velocity since topology matches wrap_grid
    r"C:\Users\105849\Desktop\morphed_mesh.vtk"
)



# Add the helper method to PyVista UnstructuredGrid class
# pv.UnstructuredGrid._as_dpf_mesh = _pv_to_dpf_mesh

# %%
# plotter = pv.Plotter(off_screen=True, window_size=[1080, 1080])
# frames = []

# PHASE1_FRAMES = 20
# PHASE2_FRAMES = 100
# PHASE3_FRAMES = 20

# # Phase 1: Show mesh with full faces
# for alpha in np.linspace(0, 1, PHASE1_FRAMES):
#     plotter.add_mesh(pv.UnstructuredGrid({5: f}, v), style='surface', color='gray', opacity=1-alpha)
#     plotter.add_mesh(pv.UnstructuredGrid({5: f}, v), style='wireframe', line_width=1, color='black', opacity=alpha)
#     plotter.camera_position = 'xz'
#     frames.append(plotter.screenshot())
#     plotter.clear()

# # Phase 3: Morph from original to final mesh in wireframe
# for t in np.linspace(0, 1, PHASE2_FRAMES):
#     t_smooth = 0.5 * (1 - np.cos(t * np.pi))
#     intermediate_vertices = (1 - t_smooth)*v + t_smooth*S
#     plotter.add_mesh(pv.UnstructuredGrid({5: f}, intermediate_vertices),
#                      style='wireframe', line_width=1, color='black')
#     plotter.camera_position = 'xz'
#     frames.append(plotter.screenshot())
#     plotter.clear()

# # # Phase 4: Fade wireframe to full faces at final position
# # for alpha in np.linspace(1, 0, PHASE3_FRAMES):
# #     # surface mesh with increasing opacity
# #     plotter.add_mesh(pv.UnstructuredGrid({5: f}, S), style='surface', color='gray', opacity=1 - alpha)
# #     # wireframe mesh with reducing opacity
# #     plotter.add_mesh(pv.UnstructuredGrid({5: f}, S), style='wireframe', line_width=1, color='black', opacity=alpha)
# #     plotter.camera_position = 'xy'
# #     frames.append(plotter.screenshot())
# #     plotter.clear()

# frames_pil = [Image.fromarray(f) for f in frames]
# frames_pil_reversed = list(reversed(frames_pil))
# frames_pil = frames_pil + frames_pil_reversed
# frames_pil[0].save(
#     r"C:\Users\105849\Desktop\mesh_morphing.gif",
#     save_all=True,
#     append_images=frames_pil[1:],
#     duration=80,
#     loop=True
# )

# # %%


# plotter = pv.Plotter()

# # Add the mesh to the plotter
# plotter.add_mesh(pv.UnstructuredGrid({5: f}, v), style='surface', color='gray', opacity=1)

# # Set the camera position
# plotter.camera_position = 'xy'

# # Show the plot
# plotter.show()

# # %%
# # Create animation showing morphing between original and final mesh in HD
# plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
# frames = []
# for t in np.linspace(0, 1, 100):
#     t_smooth = 0.5 * (1 - np.cos(t * np.pi))
#     intermediate_vertices = (1 - t_smooth)*v + t_smooth*S
#     grid = pv.UnstructuredGrid({5: f}, intermediate_vertices)
#     plotter.add_mesh(grid, style='wireframe', line_width=1, color='black')
#     plotter.camera_position = 'xy'
#     frame = plotter.screenshot()
#     frames.append(frame)
#     plotter.clear()

# frames_pil = [Image.fromarray(frame) for frame in frames]
# frames_pil_reversed = list(reversed(frames_pil))
# frames_pil = frames_pil + [frames_pil[-1]]*20 + frames_pil_reversed + [frames_pil_reversed[-1]]*20
# frames_pil[0].save(r"C:\Users\105849\Desktop\mesh_morphing.gif", save_all=True, append_images=frames_pil[1:], duration=80, loop=True)


# # %%

# nodal_normals = hdpf.get_normals(skin_mesh)
# # skin_mesh.nodes.coordinates_field.data[v_used] = S

# skin_mesh_scoping = hdpf.get_scoping_from_mesh(skin_mesh, "Nodal")
# nodal_normals = hdpf.get_normals(skin_mesh)
# tfreq = model.metadata.time_freq_support.time_frequencies
# disp = hdpf.get_normal_velocitiy_fc_from_skin_mesh(model, skin_mesh, tfreq, skin_mesh_scoping, nodal_normals)
# disp = disp.eval()
# disp = disp[200].data

# import pyvista as pv
# import numpy as np
# from PIL import Image

# # Assuming f, v, S, and disp are already defined
# # f: faces of the mesh
# # v: initial vertices of the mesh
# # S: morphed vertices of the mesh
# # disp: displacement field

# # Create the plotter object
# plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
# frames = []

# # Create animation showing displacements on the morphed geometry
# for t in np.linspace(0, 1, 100):
#     t_smooth = 0.5 * (1 - np.cos(t * np.pi))
#     intermediate_vertices = (1 - t_smooth) * v + t_smooth * S
#     grid = pv.UnstructuredGrid({5: f}, intermediate_vertices)

#     # Apply displacements to the morphed geometry
#     displacements = (1 - t_smooth) * np.zeros_like(disp) + t_smooth * disp
#     grid.point_data['displacement'] = np.linalg.norm(displacements, axis=1)

#     # Add the mesh with the jet color map
#     plotter.add_mesh(grid, scalars='displacement', cmap='jet', style='surface')
#     plotter.camera_position = 'xy'
#     frame = plotter.screenshot()
#     frames.append(frame)
#     plotter.clear()

# # Create the GIF
# frames_pil = [Image.fromarray(frame) for frame in frames]
# frames_pil_reversed = list(reversed(frames_pil))
# frames_pil = frames_pil + [frames_pil[-1]]*20 + frames_pil_reversed + [frames_pil_reversed[-1]]*20
# frames_pil[0].save(r"C:\Users\105849\Desktop\displacement_animation.gif", save_all=True, append_images=frames_pil[1:], duration=80, loop=True)