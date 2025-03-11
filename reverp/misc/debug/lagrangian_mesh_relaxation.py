from ansys.dpf import core as dpf
import reverp.src.lib.hansen_dpf_functions as hdpf
import numpy as np
import trimesh
import robust_laplacian
import numpy as np
import trimesh
import scipy.optimize as optimize
from scipy.spatial import KDTree

def dpf_to_trimesh(coordinates, connectivities):
    # Reshape coordinates if they are 1D
    if len(coordinates.shape) == 1:
        n_vertices = len(coordinates) // 3
        vertices = coordinates.reshape(n_vertices, 3)
    else:
        vertices = np.array(coordinates)

    # Reshape connectivities into triangles
    n_triangles = len(connectivities) // 3
    faces = np.array(connectivities).reshape(n_triangles, 3)

    mesh = trimesh.Trimesh(vertices=vertices,
                          faces=faces,
                          process=False)
    return mesh

# path = r"C:\Users\105849\Desktop\file.rst"
# path = r"N:\PhD\LY_revERP\file.rst"
# path = r"N:\PhD\revERP\Simple_nonconvex_box_files\dp0\SYS\MECH\file.rst"
path = r"C:\Users\105849\Desktop\motor_rst\file.rst"

model = dpf.Model(path)
skin_mesh = hdpf.get_skin_mesh_from_ns("FSI", model)
nodes = skin_mesh.nodes
elements = skin_mesh.elements
coor = nodes.coordinates_field.data
connectivities_field = elements.connectivities_field.data

# Create mesh with reshaped vertices
mesh = dpf_to_trimesh(coor, connectivities_field)
total_area = np.sum(mesh.facets_area)
# mesh.vertices = mesh.vertices / np.sqrt(total_area/(4*np.pi))

L, M = robust_laplacian.mesh_laplacian(np.array(mesh.vertices), np.array(mesh.faces))
# L, M = robust_laplacian.point_cloud_laplacian(np.array(mesh.vertices))

sphere_vertices = mesh.vertices
# sphere_vertices = sphere_vertices / (np.sqrt(np.sum(sphere_vertices**2, axis=1, keepdims=True)))
nb_it = 10000
dt = 0.05

for it in range(nb_it):
    # print(it)
    sphere_vertices = sphere_vertices - dt * L.dot(sphere_vertices)# - 1000*dt * M.dot(sphere_vertices)
    norm_sph_vert = np.sqrt(np.sum(sphere_vertices * sphere_vertices, 1))
    sphere_vertices = sphere_vertices / np.tile(norm_sph_vert, (3, 1)).T

dpf_sphere_mesh = skin_mesh.deep_copy()
dpf_sphere_mesh.nodes.coordinates_field.data = sphere_vertices
dpf_sphere_mesh.plot(
    background_color='white',  # Set background to white
    color='white',            # Set mesh line color to black
    opacity=1.0,              # Full opacity
    show_edges=True           # Ensure edges are visible
)


# %%

import numpy as np
from scipy.optimize import minimize

def mobius_area_correction_spherical(v, f, map_coords):
    """
    Find an optimal Mobius transformation for reducing the area distortion of a
    spherical conformal parameterization.

    Parameters:
        v (np.ndarray): nv x 3 vertex coordinates of a genus-0 closed triangle mesh
        f (np.ndarray): nf x 3 triangulations of a genus-0 closed triangle mesh
        map_coords (np.ndarray): nv x 3 vertex coordinates of the spherical conformal parameterization

    Returns:
        tuple: (map_mobius, x) where:
            - map_mobius (np.ndarray): nv x 3 vertex coordinates of the updated spherical conformal parameterization
            - x (np.ndarray): optimal parameters for the Mobius transformation
    """

    def face_area(f, v):
        """Compute the area of every face of a triangle mesh."""
        v12 = v[f[:, 1]] - v[f[:, 0]]
        v23 = v[f[:, 2]] - v[f[:, 1]]
        v31 = v[f[:, 0]] - v[f[:, 2]]

        a = np.sqrt(np.sum(v12 * v12, axis=1))
        b = np.sqrt(np.sum(v23 * v23, axis=1))
        c = np.sqrt(np.sum(v31 * v31, axis=1))

        s = (a + b + c) / 2
        return np.sqrt(s * (s-a) * (s-b) * (s-c))

    def stereographic(u):
        """Stereographic projection."""
        if u.shape[1] == 2:
            x, y = u[:, 0], u[:, 1]
            z = 1 + x**2 + y**2
            return np.column_stack([2*x/z, 2*y/z, (-1 + x**2 + y**2)/z])
        else:
            x, y, z = u[:, 0], u[:, 1], u[:, 2]
            return np.column_stack([x/(1-z), y/(1-z)])

    def finite_mean(A):
        """Calculate mean avoiding Inf values."""
        return np.mean(A[np.isfinite(A)])

    # Compute the area with normalization
    area_v = face_area(f, v)
    area_v = area_v / np.sum(area_v)

    # Project the sphere onto the plane
    p = stereographic(map_coords)
    z = p[:, 0] + 1j * p[:, 1]

    def compute_mobius_transform(x):
        """Apply Mobius transformation with given parameters."""
        numerator = (x[0] + x[1]*1j)*z + (x[2] + x[3]*1j)
        denominator = (x[4] + x[5]*1j)*z + (x[6] + x[7]*1j)
        return numerator / denominator

    def area_map(x):
        """Calculate area after Mobius transformation."""
        fz = compute_mobius_transform(x)
        transformed_coords = stereographic(np.column_stack([np.real(fz), np.imag(fz)]))
        areas = face_area(f, transformed_coords)
        return areas / np.sum(areas)

    def objective(x):
        """Objective function for optimization."""
        return finite_mean(np.abs(np.log(area_map(x) / area_v)))

    # Optimization setup
    x0 = np.array([1, 0, 0, 0, 0, 0, 1, 0])  # initial guess
    bounds = [(-100, 100) for _ in range(8)]  # parameter bounds

    # Perform optimization
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    x = result.x

    # Compute final transformation
    fz = compute_mobius_transform(x)
    map_mobius = stereographic(np.column_stack([np.real(fz), np.imag(fz)]))

    return map_mobius, x



map_mobius, x =  mobius_area_correction_spherical(coor, np.array(mesh.faces), sphere_vertices)

dpf_mobius_mesh = skin_mesh.deep_copy()
dpf_mobius_mesh.nodes.coordinates_field.data = map_mobius
dpf_mobius_mesh.plot(
    background_color='white',  # Set background to white
    color='white',            # Set mesh line color to black
    opacity=1.0,              # Full opacity
    show_edges=True           # Ensure edges are visible
)

# # %%
# # sphere_mesh.show(viewer='gl')

# # %%
# tfreq = model.metadata.time_freq_support.time_frequencies
# scoping = hdpf.get_scoping_from_mesh(skin_mesh, "Nodal")
# normals = hdpf.get_normals(skin_mesh)
# vns = hdpf.get_normal_velocitiy_fc_from_skin_mesh(model, dpf_sphere_mesh, tfreq, scoping, normals)
# vns = vns.outputs.fields_container()
# # %%


# vns[250].plot()

# # %%
# tfreq = model.metadata.time_freq_support.time_frequencies
# scoping = hdpf.get_scoping_from_mesh(skin_mesh, "Nodal")
# normals = hdpf.get_normals(skin_mesh)
# vn = hdpf.get_normal_velocitiy_fc_from_skin_mesh(model, skin_mesh, tfreq, scoping, normals)
# vn = vn.outputs.fields_container()
# # %%
# vn[250].plot()
