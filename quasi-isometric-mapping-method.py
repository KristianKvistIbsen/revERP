import numpy as np
import pyshtools as pysh
from scipy.interpolate import griddata
from ansys.dpf import core as dpf
from reverp.src.utils.spherical import (
    impedance_Z,
    spherical_node_mapping,
    create_spherical_grid,
    field_extrapolation,
    plot_extrapolated_field_plane,
    animate_extrapolated_field_plane
)
import pyvista as pv
import pyshtools as pysh
from scipy.interpolate import griddata
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
from reverp.src.lib.pySDEM import pySDEM
from reverp.src.lib.hansen_dpf_functions import (
    get_skin_mesh_from_ns,
    get_scoping_from_mesh,
    get_normals,
    compute_faces_area
)

n_harmonics = 60
nlat = 2 * n_harmonics
nlon = 2 * nlat

# Define acoustic constants
a = 0.025  # sphere radius in meters
freq = 1000  # frequency in Hz
c = 1500    # speed of sound in air, m/s
k = 2 * np.pi * freq / c  # wavenumber
v0 = 1     # surface normal velocity, m/s
sphere_location = [-25,-25, 0]
rho = 997  # air density, kg/m^3
omega = 2 * np.pi * freq

mesh_path = r"N:\sub_scatter_model_files\dp0\SYS\MECH\file.rst"

# Load model and skin mesh
model = dpf.Model(mesh_path)
skin_mesh = get_skin_mesh_from_ns("FSI", model)
normals = get_normals(skin_mesh).data
face_areas = compute_faces_area(skin_mesh, get_scoping_from_mesh(skin_mesh, "Elemental")).data

# Prepare skin mesh grid, removing midside nodes
skin_mesh_grid = skin_mesh.grid
v = skin_mesh_grid.points
f = skin_mesh_grid.cells_dict[list(skin_mesh_grid.cells_dict.keys())[0]]
v_used = np.unique(f)
v_unused = np.setdiff1d(np.arange(len(v)), v_used)
v = v[v_used]
normals = normals[v_used]

# Remap face indices
v_map = np.zeros(len(v_used) + len(v_unused), dtype=int)
v_map[v_used] = np.arange(len(v_used))
f = v_map[f].astype(int)

# Compute incoming wave normal velocity
r = np.linalg.norm(v - sphere_location, axis=1)
vr = -v0* (a**2)/(r**2)*(1-(1j*k*r))/(1-(1j*k*a))*np.exp(1j * k * r - 1j * k * a)

r_hat = (v - sphere_location) / r[:, np.newaxis]
dot_prod = np.sum(r_hat * normals, axis=1)
v_n = vr * dot_prod  # v_n_incoming

# Set surface velocity for scattering
v_n_surface = -v_n  # v_n_scattered = -v_n_incoming

# Compute spherical radius R
area = np.sum(face_areas)
R = np.sqrt(area / (4 * np.pi))

# SDEM application (inflates to sphere)
S = pySDEM.pySDEM(v, f, face_areas, 2000, 0, 10)
scaling_factor = np.sqrt(skin_mesh_grid.area / (4 * np.pi))
coordinates = S * scaling_factor  # Inflated sphere coordinates

# Get spherical coordinates
sphere_coords = spherical_node_mapping(
    coordinates,
    skin_mesh_grid.center
)

lats = 90 - np.degrees(sphere_coords[:, 4])  # Convert to degrees and adjust range
lons = 180 + np.degrees(sphere_coords[:, 5])  # Convert to degrees and adjust range
grid_lats, grid_lons = create_spherical_grid(nlat, nlon)

# Replace v_n_surface with v_n_surface_corrected in subsequent steps
lats = 90 - np.degrees(sphere_coords[:, 4])
lons = 180 + np.degrees(sphere_coords[:, 5])
grid_values = griddata(
    (lats, lons),
    v_n_surface,
    (grid_lats, grid_lons),
    method='nearest'
)


# 3. Compute spherical harmonics
grid = pysh.SHGrid.from_array(grid_values)
clm = grid.expand(normalization='ortho')

plm = np.zeros_like(clm.coeffs,dtype=np.complex128)
for id_l in range(n_harmonics):
    plm[:,id_l,:] = clm.coeffs[:,id_l,:]*impedance_Z(id_l,k*R,rho,c)
plm[np.isnan(plm)] = 0

# Define Hankel function
def h_l1(l, z):
    return spherical_jn(l, z) + 1j * spherical_yn(l, z)

def compute_incoming_pressure(radii, lats, lons, sphere_location, v0, k, omega, rho, a):
    nr = len(radii)
    n_angles = len(lats)
    p_incoming = np.zeros((nr, n_angles), dtype=complex)

    # Adjust longitude by 180 degrees to flip the field
    lons_adjusted = (lons + 180) % 360

    # Convert latitudes and adjusted longitudes to radians
    theta = np.radians(90 - lats)  # Colatitude
    phi = np.radians(lons_adjusted)

    # Loop over radial distances
    for i, ri in enumerate(radii):
        # Compute Cartesian coordinates of field points (centered at origin)
        x = ri * np.sin(theta) * np.cos(phi)
        y = ri * np.sin(theta) * np.sin(phi)
        z = ri * np.cos(theta)

        # Compute distance from each point to the sphere's location
        r_in = np.sqrt(
            (x - sphere_location[0])**2 +
            (y - sphere_location[1])**2 +
            (z - sphere_location[2])**2
        )
        phi_incoming = (a**2*v0*np.exp(1j*k*r_in-1j*k*a))/(r_in*(1-1j*k*a))
        p_incoming[i, :] = 1j * omega * rho * phi_incoming

    return p_incoming


# %%


# Parameters for extrapolation
nr = 500          # Number of radial points
r_final = 20  # Final radius (5 times the initial radius R)
plane = 'XY'     # Plane to visualize
n_angles = 500   # Number of angular points

p_scattered, radii, lats, lons, angles = field_extrapolation(
    plm, R, r_final, nr, k, plane=plane, plot=False ,n_angles=n_angles
)

p_incoming = compute_incoming_pressure(
    radii, lats, lons, sphere_location, v0, k, omega, rho, a
)
# %%
# Step 3: Compute the total pressure field
p_total = 0
p_total += p_incoming
p_total += p_scattered

user_v_max = np.max(np.abs(p_scattered))
user_v_min = -np.max(np.abs(p_scattered))
# Step 4: Plot the total pressure field
fig = plot_extrapolated_field_plane(p_total, R, r_final, nr, angles, plane, "real", user_cmap="ANSYS",vmin=user_v_min,vmax=user_v_max)
# fig = plot_extrapolated_field_plane(-p_incoming, R, r_final, nr, angles, plane, "real", user_cmap="jet",vmin=-v_range,vmax=v_range)
# fig = plot_extrapolated_field_plane(-p_scattered, R, r_final, nr, angles, plane, "real", user_cmap="jet",vmin=-v_range,vmax=v_range)
# %%


# anim = animate_extrapolated_field_plane(p_total, R, r_final, nr, angles, plane, omega, n_frames=20, interval=10, user_cmap="jet",vmin=-v_range,vmax=v_range)
# anim = animate_extrapolated_field_plane(p_incoming, R, r_final, nr, angles, plane, omega, n_frames=20, interval=10, user_cmap="jet",vmin=-v_range,vmax=v_range)
# anim = animate_extrapolated_field_plane(p_scattered, R, r_final, nr, angles, plane, omega, n_frames=20, interval=10, user_cmap="jet",vmin=-v_range,vmax=v_range)



# %%
# Step 1: Create the Original Skin Mesh with Used Vertices
faces_pv = np.hstack([np.full((f.shape[0], 1), 3, dtype=int), f]).ravel()
original_mesh = pv.PolyData(v, faces_pv)

# Step 2: Create the Inflated Spherical Mesh
inflated_mesh = pv.PolyData(coordinates, faces_pv)

# Step 3: Create the Source Ball (Vibrating Sphere)
source_ball = pv.Sphere(radius=a, center=sphere_location)

# Step 4: Attach Normal Velocity Data to Meshes
original_mesh.point_data['Vn'] = v_n_surface  # Normal velocity for original mesh
inflated_mesh.point_data['Vn'] = v_n_surface  # Normal velocity for inflated mesh

# Step 5: Set Up the Plotter
plotter = pv.Plotter()

# Add the original skin mesh with velocity coloring
# plotter.add_mesh(original_mesh, scalars='Vn', cmap='viridis', opacity=1.0, label='Original Skin Mesh',)

# Add the inflated spherical mesh with velocity coloring
plotter.add_mesh(inflated_mesh, scalars='Vn', cmap='viridis', opacity=1, label='Inflated Spherical Mesh')
#
# Add the source ball (no velocity, just red)
plotter.add_mesh(source_ball, color='red', label='Source Ball')

# Step 7: Adjust the Plot for Better Visualization
plotter.camera_position = 'xy'  # Set view to XY plane
plotter.camera.zoom(1.5)       # Zoom out to ensure all objects are visible

# Add a legend to identify each mesh
plotter.add_legend()

# Step 8: Display the Plot
plotter.show()


