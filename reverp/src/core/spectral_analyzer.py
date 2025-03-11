# reverp/core/spectral_analyzer.py
from typing import Dict, Any, Optional

import logging
import numpy as np
import pyshtools as pysh
from scipy.interpolate import griddata
from rich.progress import Progress
from ansys.dpf import core as dpf

from ..utils.spherical import (
    impedance_Z,
    spherical_node_mapping,
    create_spherical_grid,
    field_extrapolation
)

def plot_field_on_sphere(field, grid_lats, grid_lons, sphere_radius, sphere_center, sphere_coords):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata

    # Create figure with a single 3D subplot
    fig = plt.figure(figsize=(10, 8))  # Adjusted size for a single plot
    ax = fig.add_subplot(111, projection='3d')  # Single subplot

    # Convert spherical grid to Cartesian coordinates, shifted by sphere_center
    theta = np.radians(90 - grid_lats)
    phi = np.radians(grid_lons - 180)
    X = sphere_center[0] + sphere_radius * np.sin(theta) * np.cos(phi)
    Y = sphere_center[1] + sphere_radius * np.sin(theta) * np.sin(phi)
    Z = sphere_center[2] + sphere_radius * np.cos(theta)

    # Interpolate the imaginary part of mode onto the grid
    grid_mode = griddata(
        (90 - np.degrees(sphere_coords[:, 4]), 180 + np.degrees(sphere_coords[:, 5])),
        field,
        (grid_lats, grid_lons),
        method='nearest'
    )

    # Define normalization for consistent coloring (imaginary part can be positive/negative)
    vmax = np.max(np.abs(grid_mode))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    colors = plt.cm.coolwarm(norm(grid_mode))  # Diverging colormap for positive/negative values

    # Plot the spherical grid surface
    ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.8)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Field')

    # Add coordinate system
    arrow_length = sphere_radius * 2  # Scale arrows relative to sphere radius
    ax.quiver(
        sphere_center[0], sphere_center[1], sphere_center[2],
        arrow_length, 0, 0, color='r', label='X'
    )
    ax.quiver(
        sphere_center[0], sphere_center[1], sphere_center[2],
        0, arrow_length, 0, color='g', label='Y'
    )
    ax.quiver(
        sphere_center[0], sphere_center[1], sphere_center[2],
        0, 0, arrow_length, color='b', label='Z'
    )

    # Set labels and title
    ax.set_title('Field on Spherical Grid')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.axis("equal")
    # Set view angle
    ax.view_init(elev=20, azim=30)

    plt.tight_layout()
    return fig

class SpectralAnalyzer:

    def __init__(self, config: Optional[Dict] = None):
        """Initialize SpectralAnalyzer with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Get configuration parameters
        analysis_config = self.config.get("analysis", {})
        self.n_harmonics = analysis_config.get("spherical_harmonics")
        self.nlat = 2 * self.n_harmonics
        self.nlon = 2 * self.nlat

        # Get physics constants
        physics_config = self.config.get("physics", {})
        self.C = physics_config.get("speed_of_sound")
        self.RHO = physics_config.get("density")
        self.REFERENCE_POWER = physics_config.get("reference_power")

    def analyze(
        self,
        mesh_data: Dict[str, Any],
        sphere_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        try:
            self.logger.info("Performing spectral analysis")

            # Get frequencies from erp_calculation and initialize arrays
            frequencies = mesh_data["tfreq"].data
            n_frequencies = len(frequencies)

            spectral = np.zeros(n_frequencies)
            spectral_db = np.zeros(n_frequencies)

            # Create extractor for velocity fields
            op_extractor = dpf.operators.utility.extract_sub_fc()
            op_extractor.inputs.fields_container.connect(mesh_data.get("vel_disp_fc"))

            # Get node mappings
            node_ids = mesh_data["field_mesh_scoping"].ids
            node_id_to_index = {id: index for index, id in enumerate(node_ids)}

            # Get spherical coordinates
            sphere_coords = spherical_node_mapping(
                mesh_data["coor"],
                sphere_data["radius"],
                sphere_data["center"]
            )

            # Extract lat/lon data
            lats = 90 - np.degrees(sphere_coords[:, 4])  # Convert to degrees and adjust range
            lons = 180 + np.degrees(sphere_coords[:, 5])  # Convert to degrees and adjust range

            # Create projection grid
            grid_lats, grid_lons = create_spherical_grid(self.nlat, self.nlon)

            # Process each frequency
            # with Progress() as progress:
            # task = progress.add_task("[cyan]revERP: Processing frequencies...", total=n_frequencies)

            for id_f, freq in enumerate(frequencies):
                try:
                    # 1. Get velocity data for this frequency
                    op_extractor.inputs.label_space.connect({"time": id_f + 1})
                    current_Vn_fc = op_extractor.outputs.fields_container()

                    vn = (current_Vn_fc[0].data + current_Vn_fc[1].data * 1j)

                    k = 2*np.pi*freq/self.C
                    ka = k*sphere_data["radius"]

                    # Map velocities to node IDs
                    mode = [vn[node_id_to_index[id]] for id in node_ids]
                    if self.config.get('SDEM_projection').get('SDEM', False):
                        mode = np.array(mode)[mesh_data["v_used"]]

                    # 2. Project onto spherical grid
                    grid_values = griddata(
                        (lats, lons),
                        mode,
                        (grid_lats, grid_lons),
                        method='nearest'
                    )

                    # 3. Compute spherical harmonics
                    grid = pysh.SHGrid.from_array(grid_values)
                    clm = grid.expand(normalization='ortho')

                    # 4. Compute power from spectrum
                    clm2 = np.sum(np.sum(np.abs(clm.coeffs)**2,axis=1),axis=0)
                    for id_l in range(self.n_harmonics):
                        inv_wavelength = (k/(2*np.pi))
                        contribution = inv_wavelength * 1/(2*self.RHO*self.C)*clm2[id_l]*np.real(impedance_Z(id_l,ka,self.RHO,self.C))
                        if np.isnan(contribution):
                            contribution = 0
                        spectral[id_f] += contribution
                    spectral_db[id_f] = 10 * np.log10(spectral[id_f] / self.REFERENCE_POWER)

                    plm = np.zeros_like(clm.coeffs,dtype=np.complex128)
                    dlm = np.zeros_like(clm.coeffs,dtype=np.complex128)
                    for id_l in range(self.n_harmonics):
                        plm[:,id_l,:] = clm.coeffs[:,id_l,:]*impedance_Z(id_l,ka,self.RHO,self.C)
                        dlm[:,id_l,:] = plm[:,id_l,:]*(-1j**(id_l+1))/k
                    plm[np.isnan(plm)] = 0
                    dlm[np.isnan(dlm)] = 0

                    aaa = plm/plm
                    aaa[np.isnan(aaa)] = 0

                    bbb = np.array([  [[0,0],[1,0]],[ [0,0],[0,0]] ])
                    extrapolated_field = field_extrapolation(plm,sphere_data["radius"],0.4519,30,k,"XZ",plot=True, plot_type="abs", n_angles=360, vmin=0.39, vmax=0.00017)

                    synthesized_field = pysh.expand.MakeGridPointC(plm, -lats, lons, norm=4)
                    plot_field_on_sphere(np.real(synthesized_field), grid_lats, grid_lons, sphere_data["radius"], sphere_data["center"], sphere_coords)

                        # progress.update(task, advance=1)

                except Exception as e:
                    self.logger.error(f"Failed processing frequency {freq} Hz: {str(e)}")
                    raise

            return {
                "spectral": spectral,
                "spectral_db": spectral_db
            }

        except Exception as e:
            self.logger.error(f"Spectral analysis failed: {str(e)}")
            raise


# # %%





# # %%PRESSURE DISC
#                         min_radius = 1
#                         max_radius = 20
#                         n_radius = 50
#                         n_angles = 360
#                         radii = np.linspace(min_radius, max_radius, n_radius)
#                         angles = np.linspace(0, 360, n_angles)
#                         theta_grid, r_grid = np.meshgrid(angles, radii)
#                         # Convert to cartesian for visualization
#                         x = r_grid * np.cos(np.radians(theta_grid))
#                         z = r_grid * np.sin(np.radians(theta_grid))
#                         thetas = theta_grid.flatten()
#                         phis = np.full_like(thetas, 90.0)
#                         pressure_field = pysh.expand.MakeGridPointC(dlm, thetas, phis, norm=4)
#                         pressure_db = 20 * np.log10(np.abs(pressure_field) / 2E-5)  #Reference 20µPa
#                         pressure_disc = np.abs(pressure_field).reshape(r_grid.shape) #pressure_db.reshape(r_grid.shape)

#                         import matplotlib.pyplot as plt
#                         plt.figure(figsize=(10, 10))
#                         plt.pcolormesh(x, z, pressure_disc, cmap='turbo', shading='auto')
#                         cbar = plt.colorbar()
#                         cbar.set_label('Sound Pressure Level (dB)')
#                         plt.axis('equal')
#                         plt.title(f'Pressure Field on XZ Plane (f={freq:.1f} Hz)')
#                         plt.xlabel('X (m)')
#                         plt.ylabel('Z (m)')
#                         plt.tight_layout()
# # %% DIRECTIVITY
#                         # # 5. Compute directivity pattern
#                         # phis = np.linspace(0, 360, 360)
#                         # theta = -0
#                         # thetas = np.full_like(phis, theta)-90
#                         # # Calculate directivity at all points
#                         # D = pysh.expand.MakeGridPointC(dlm, thetas, phis, norm=4)
#                         # D = 10 * np.log10(D / 2E-5)
#                         # all_directivity.append(D)
# # %%

