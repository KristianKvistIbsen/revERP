# reverp/utils/spherical.py
from typing import Tuple, Any
import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.optimize import minimize
import trimesh
import pyvista as pv
import pyshtools as pysh
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import numpy as np
import pyshtools as pysh
from scipy.special import spherical_jn, spherical_yn

def field_extrapolation(alm, r0, r_final, nr, k, plane='XY', plot=False, plot_type="real", n_angles=360, vmin=None, vmax=None):
    """
    Extrapolate a field on a specified plane for polar plotting.

    Args:
        alm: Spherical harmonic coefficients
        r0: Initial radius (m)
        r_final: Final radius (m)
        nr: Number of radial points
        k: Wavenumber
        plane: 'XY', 'XZ', or 'YZ'
        plot: Boolean to enable plotting
        plot_type: 'real', 'imag', or 'abs'
        n_angles: Number of angular points
        vmin: Minimum value for colormap (optional)
        vmax: Maximum value for colormap (optional)

    Returns:
        field: Extrapolated field (nr, n_angles)
        radii: Radial points
        lats: Latitudes
        lons: Longitudes
        angles: Angular grid (degrees)
    """
    # print("DEBUG FLAG")
    # Define the base angles from 0 to 360 degrees
    angles = np.linspace(0, 360, n_angles, endpoint=False)

    # Define grid based on plane
    if plane == 'XY':
        lats = np.zeros(n_angles)
        lons = (angles + 180) % 360
    elif plane == 'XZ':
        # Offset angles by 180 degrees
        offset_angles = (angles + 180) % 360
        lats, lons = [], []
        for alpha in offset_angles:
            if alpha == 90:
                lats.append(90)
                lons.append(0)
            elif alpha == 270:
                lats.append(-90)
                lons.append(0)
            else:
                X = np.cos(np.radians(alpha))
                Z = np.sin(np.radians(alpha))
                theta = np.arccos(Z)
                phi = 0 if X >= 0 else 180
                lats.append(90 - np.degrees(theta))
                lons.append(phi)
        lats = np.array(lats)
        lons = np.array(lons)
    elif plane == 'YZ':
        # Offset angles by 180 degrees
        offset_angles = (angles + 180) % 360
        lats, lons = [], []
        for beta in offset_angles:
            if beta == 90:
                lats.append(90)
                lons.append(90)
            elif beta == 270:
                lats.append(-90)
                lons.append(90)
            else:
                Y = np.cos(np.radians(beta))
                Z = np.sin(np.radians(beta))
                theta = np.arccos(Z)
                phi = 90 if Y >= 0 else 270
                lats.append(90 - np.degrees(theta))
                lons.append(phi)
        lats = np.array(lats)
        lons = np.array(lons)
    else:
        raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")

    # Extrapolate field
    field = np.zeros((nr, n_angles), dtype=complex)
    radii = np.linspace(r0, r_final, nr)
    for i, ri in enumerate(radii):
        alm_i = np.zeros_like(alm, dtype=complex)
        for l in range(alm.shape[2]):
            jn_0 = spherical_jn(l, k * r0)
            yn_0 = spherical_yn(l, k * r0)
            h0 = jn_0 + 1j * yn_0
            jn_i = spherical_jn(l, k * ri)
            yn_i = spherical_yn(l, k * ri)
            hi = jn_i + 1j * yn_i
            alm_i[:, l, :] = (hi / h0) * alm[:, l, :]
        alm_i[np.isnan(alm_i)] = 0
        field[i, :] = pysh.expand.MakeGridPointC(alm_i, lats, lons, norm=4)

    if plot:
        import matplotlib.pyplot as plt
        fig = plot_extrapolated_field_plane(field, r0, r_final, nr, angles, plane, plot_type, vmin=vmin, vmax=vmax)
        plt.show()

    return field, radii, lats, lons, angles




def plot_extrapolated_field_plane(field, r0, r_final, nr, angles, plane, plot_type, vmin=None, vmax=None, user_cmap=None):
    """
    Plot the field on the specified plane as a polar plot with customizable colormap bounds.

    Args:
        field: Field data (nr, n_angles)
        r0: Initial radius
        r_final: Final radius
        nr: Number of radial points
        angles: Angular grid (degrees)
        plane: 'XY', 'XZ', or 'YZ'
        plot_type: 'real', 'imag', or 'abs'
        vmin: Minimum value for colormap (optional)
        vmax: Maximum value for colormap (optional)
        user_cmap: Custom colormap (optional, defaults to predefined scheme)

    Returns:
        fig: Matplotlib figure object
    """
    # Define custom color scheme (reversed order: blue at minimum, red at maximum)
    colors = [
        '#0000ff',  # Blue (minimum)
        '#00b0ff',  # Light Blue
        '#00fffe',  # Cyan
        '#00ffb3',  # Cyan-Green
        '#00ff00',  # Green
        '#b2ff00',  # Light Green
        '#feff00',  # Yellow
        '#ffb001',  # Orange
        '#fe0000'   # Red (maximum)
    ]

    if user_cmap == "ANSYS":
            custom_cmap = LinearSegmentedColormap.from_list('ansys_scheme', colors, N=9)
            user_cmap = custom_cmap

    # Process field data
    if plot_type == 'real':
        data = np.real(field)
    elif plot_type == 'imag':
        data = np.imag(field)
    else:  # 'abs' or default
        data = np.abs(field)

    radii = np.linspace(r0, r_final, nr)
    theta = np.radians(angles)
    R, Theta = np.meshgrid(radii, theta)

    # Create polar plot with vmin and vmax
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    pcm = ax.pcolormesh(Theta, R, data.T, cmap=user_cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_title(f'Field on {plane} Plane')
    fig.colorbar(pcm, label=f'{plot_type.capitalize()} Field Value')

    # Set direction labels
    ticks = np.radians([0, 90, 180, 270])
    if plane == 'XY':
        labels = ['+X', '+Y', '-X', '-Y']
    elif plane == 'XZ':
        labels = ['+X', '+Z', '-X', '-Z']
    else:  # YZ
        labels = ['+Y', '+Z', '-Y', '-Z']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_theta_zero_location('E')  # 0° on right
    plt.tight_layout()
    return fig

def animate_extrapolated_field_plane(field, r0, r_final, nr, angles, plane, omega, n_frames=100, interval=50, vmin=None, vmax=None, user_cmap="turbo"):
    """
    Animate the real part of a complex field on a specified plane with time dependency e^(-i omega t).

    Args:
        field: Complex field data (nr, n_angles)
        r0: Initial radius
        r_final: Final radius
        nr: Number of radial points
        angles: Angular grid (degrees)
        plane: 'XY', 'XZ', or 'YZ'
        omega: Angular frequency (rad/s)
        n_frames: Number of frames in the animation (default: 100)
        interval: Delay between frames in milliseconds (default: 50)
        vmin: Minimum value for colormap (optional)
        vmax: Maximum value for colormap (optional)
        user_cmap: Colormap name (default: "turbo")

    Returns:
        anim: Matplotlib animation object
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np

    # Compute spatial grid
    radii = np.linspace(r0, r_final, nr)
    theta = np.radians(angles)
    R, Theta = np.meshgrid(radii, theta)

    # Precompute real and imaginary parts of the field
    field_real = np.real(field)
    field_imag = np.imag(field)

    # Determine color limits if not provided
    if vmin is None or vmax is None:
        max_abs = np.max(np.abs(field))
        vmin = -max_abs if vmin is None else vmin
        vmax = max_abs if vmax is None else vmax

    # Set up the figure and polar axes
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Initial data (t=0)
    initial_data = field_real  # At t=0, e^(-i*omega*0) = 1, so real part is field_real

    # Create initial plot
    pcm = ax.pcolormesh(Theta, R, initial_data.T, cmap=user_cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_title(f'Real Field on {plane} Plane (Animated)')
    fig.colorbar(pcm, label='Real Field Value')

    # Set direction labels
    ticks = np.radians([0, 90, 180, 270])
    if plane == 'XY':
        labels = ['+X', '+Y', '-X', '-Y']
    elif plane == 'XZ':
        labels = ['+X', '+Z', '-X', '-Z']
    else:  # YZ
        labels = ['+Y', '+Z', '-Y', '-Z']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_theta_zero_location('E')  # 0° on right
    # ax.set_theta_direction(-1)  # Uncomment if clockwise direction is desired
    plt.tight_layout()

    # Compute time steps over one period T = 2π/ω
    T = 2 * np.pi / omega
    times = np.linspace(0, T, n_frames, endpoint=False)

    # Define update function for animation
    def update(t):
        # Compute real part: Re[F e^(-iωt)] = Re[F] cos(ωt) + Im[F] sin(ωt)
        data = field_real * np.cos(omega * t) + field_imag * np.sin(omega * t)
        pcm.set_array(data.T.ravel())
        return pcm,

    # Create animation
    anim = FuncAnimation(fig, update, frames=times, interval=interval, blit=True)

    # Display the animation
    plt.show()
    return anim

def calculate_sigma_sh(clm: Any, a: float, f: float, C: float) -> np.ndarray:

    ka = a * (2 * np.pi * f) / C
    N_SPHERICAL_HARMONICS = clm.coeffs.shape[2]
    sigma_sh = np.zeros(N_SPHERICAL_HARMONICS)

    for id_h in range(N_SPHERICAL_HARMONICS):
        jn = spherical_jn(id_h, ka)
        yn = spherical_yn(id_h, ka)
        d_jn = spherical_jn(id_h, ka, derivative=True)
        d_yn = spherical_yn(id_h, ka, derivative=True)

        h1 = jn + 1j * yn
        dh1 = d_jn + 1j * d_yn
        dh2 = d_jn - 1j * d_yn

        num = np.real(1j * h1 * dh2)
        denom = np.real((dh1 * dh2))

        # Check for overflow condition
        if abs(denom) > 1e200:
            sigma_sh[id_h] = 0.0
        else:
            sigma_sh[id_h] = num / denom

    return np.nan_to_num(sigma_sh, nan=0.0)


def impedance_Z(l,ka,rho,C):
    jn = spherical_jn(l, ka)
    yn = spherical_yn(l, ka)
    d_jn = spherical_jn(l, ka, derivative=True)
    d_yn = spherical_yn(l, ka, derivative=True)

    h1 = jn + 1j * yn
    dh1 = d_jn + 1j * d_yn
    return 1j*rho*C*h1/dh1

def spherical_node_mapping(coordinates: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Map Cartesian coordinates to spherical coordinates.

    Args:
        coordinates: Node coordinates (n_nodes × 3)
        radius: Sphere radius
        center: Sphere center coordinates [x, y, z]

    Returns:
        np.ndarray: Array with columns [x, y, z, r, theta, phi]
    """
    # Shift coordinates to sphere center
    shifted_coords = coordinates - center

    # Calculate spherical coordinates
    xy = shifted_coords[:,0]**2 + shifted_coords[:,1]**2
    r = np.sqrt(xy + shifted_coords[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), shifted_coords[:,2])
    phi = np.arctan2(shifted_coords[:,1], shifted_coords[:,0])

    # Combine all coordinates
    spherical = np.column_stack([
        shifted_coords,
        r[:, np.newaxis],
        theta[:, np.newaxis],
        phi[:, np.newaxis]
    ])

    return spherical

def best_fit_sphere(coordinates: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find best-fit sphere for given coordinates.

    Args:
        coordinates: Node coordinates (n_nodes × 3)

    Returns:
        Tuple containing:
        - np.ndarray: Sphere center coordinates [x, y, z]
        - float: Sphere radius
    """
    def objective(R: float, center: np.ndarray, coords: np.ndarray) -> float:
        d = np.linalg.norm(coords - center, axis=1)
        return np.sum(np.abs(d - R))

    # Initial guess for center
    center = coordinates.mean(axis=0)

    # Initial guess for radius
    init_radius = np.mean(np.linalg.norm(coordinates - center, axis=1))

    # Optimization bounds
    min_radius = 0
    max_radius = np.max(np.linalg.norm(coordinates - center, axis=1))

    # Optimize radius
    result = minimize(
        lambda R: objective(R, center, coordinates),
        init_radius,
        bounds=[(min_radius, max_radius)],
        method='L-BFGS-B',
        options={'ftol': 1e-9, 'gtol': 1e-9}
    )

    optimal_radius = result.x[0]

    return center, optimal_radius

def create_spherical_grid(
    n_lat: int,
    n_lon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spherical coordinate grid for harmonic analysis.

    Args:
        n_lat: Number of latitude points
        n_lon: Number of longitude points

    Returns:
        Tuple containing:
        - np.ndarray: 2D array of latitudes
        - np.ndarray: 2D array of longitudes
    """
    # Create grid points
    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(0, 360, n_lon, endpoint=False)

    # Create mesh grid
    grid_lons, grid_lats = np.meshgrid(lons, lats)

    return grid_lats, grid_lons

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

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def directivity_grid(n, extend=0, radius=1.0, center=(0.0, 0.0, 0.0), directivity_data=None):
    """
    Create a PyVista StructuredGrid representing a spherical surface.

    Parameters:
    - n (int): Number of latitude points (excluding south pole if extend=0).
    - extend (int): 0 to exclude south pole and 360° E, 1 to include them.
    - radius (float): Radius of the sphere (default: 1.0).
    - center (tuple): (x, y, z) coordinates of the sphere's center (default: (0, 0, 0)).
    - directivity_data (2D array, optional): Data to assign to grid points.

    Returns:
    - grid (pv.StructuredGrid): PyVista grid object with optional data.
    """
    # Validate extend parameter
    if extend not in [0, 1]:
        raise ValueError("extend must be 0 or 1.")

    # Set grid dimensions
    n_lat = n if extend == 0 else n + 1
    n_lon = 2 * n if extend == 0 else 2 * n + 1

    # Generate latitude grid (in degrees)
    if extend == 0:
        lat_min = -90.0 + 180.0 / n  # Stop before south pole
        lats = np.linspace(90.0, lat_min, n_lat)
    else:
        lats = np.linspace(90.0, -90.0, n_lat)  # Include south pole

    # Generate longitude grid (in degrees)
    if extend == 0:
        lon_max = 360.0 - 360.0 / (2 * n)  # Stop before 360°
        lons = np.linspace(0.0, lon_max, n_lon)
    else:
        lons = np.linspace(0.0, 360.0, n_lon)  # Include 360°

    # Create 2D coordinate grids and convert to radians
    lon_grid, lat_grid = np.meshgrid(np.radians(lons), np.radians(lats))

    # Convert spherical to Cartesian coordinates
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)

    # Create PyVista StructuredGrid
    grid = pv.StructuredGrid(x, y, z)

    # Assign directivity data if provided
    if directivity_data is not None:
        if directivity_data.shape != (n_lat, n_lon):
            raise ValueError(f"directivity_data must have shape ({n_lat}, {n_lon}), got {directivity_data.shape}")
        grid["directivity"] = directivity_data.ravel()

    # Scale the grid by radius
    grid.points *= radius

    # Translate the grid to the specified center
    grid.translate(center)

    return grid



# def lagrangian_mesh_relaxation(skin_mesh: dpf.MeshedRegion, itt: int, dt: float) -> dpf.MeshedRegion:
#     # Extract coordinates and connectivities
#     nodes = skin_mesh.nodes
#     elements = skin_mesh.elements
#     coor = nodes.coordinates_field.data
#     connectivities_field = elements.connectivities_field.data

#     # Convert to trimesh
#     mesh = dpf_to_trimesh(coor, connectivities_field)

#     # Compute Laplacian
#     L, M = robust_laplacian.mesh_laplacian(np.array(mesh.vertices), np.array(mesh.faces))

#     # Initialize sphere vertices
#     sphere_vertices = mesh.vertices

#     # Perform relaxation iterations
#     for it in range(itt):
#         sphere_vertices = sphere_vertices - dt * L.dot(sphere_vertices)
#         norm_sph_vert = np.sqrt(np.sum(sphere_vertices * sphere_vertices, 1))
#         sphere_vertices = sphere_vertices / np.tile(norm_sph_vert, (3, 1)).T

#     # Create new DPF mesh with relaxed vertices
#     dpf_sphere_mesh = skin_mesh.deep_copy()
#     dpf_sphere_mesh.nodes.coordinates_field.data = sphere_vertices

#     return dpf_sphere_mesh