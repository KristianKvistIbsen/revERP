# reverp/core/modal_analyzer.py
from typing import Dict, Any, Optional
import logging
import numpy as np
import pyshtools as pysh
from scipy.interpolate import griddata
from rich.progress import Progress
from ansys.dpf import core as dpf

from ..utils.spherical import (
    calculate_sigma_sh,
    spherical_node_mapping,
    create_spherical_grid
)

class ModalAnalyzer:
    """Handles modal analysis with frequency-dependent radiation efficiency calculation."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize ModalAnalyzer with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Get configuration parameters
        analysis_config = self.config.get("analysis", {})
        self.n_harmonics = analysis_config.get("spherical_harmonics")
        self.nlat = 2 * self.n_harmonics
        self.nlon = 2 * self.nlat

        # Get modal analysis specific parameters
        self.freq_resolution = analysis_config.get("modal_frequency_resolution")
        self.freq_range_factor = analysis_config.get("modal_frequency_range")

        # Get physics constants
        physics_config = self.config.get("physics", {})
        self.C = physics_config.get("speed_of_sound")
        self.REFERENCE_POWER = physics_config.get("reference_power")

    def analyze(
        self,
        mesh_data: Dict[str, Any],
        sphere_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform modal analysis and calculate radiation efficiencies across frequency range.
        """
        try:
            self.logger.info("Performing modal analysis")

            # Get eigenfrequencies from mesh data
            eigen_frequencies = mesh_data["tfreq"].data
            n_modes = len(eigen_frequencies)

            # Calculate frequency range for analysis
            max_freq = np.max(eigen_frequencies) * self.freq_range_factor
            min_freq = np.min(eigen_frequencies) / self.freq_range_factor
            freq_range = np.arange(min_freq, max_freq, self.freq_resolution)  # Start from 1Hz to avoid omega=0
            n_freq_points = len(freq_range)

            # Initialize arrays for results
            sigma_eff_modes = np.zeros((n_modes, n_freq_points))  # [n_modes × n_frequencies]
            # weights_array_modes = np.zeros((n_modes, n_freq_points, self.n_harmonics))  # [n_modes × n_frequencies × n_harmonics]

            # Create extractor for displacement fields
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
            lats = 90 - np.degrees(sphere_coords[:, 4])
            lons = 180 + np.degrees(sphere_coords[:, 5])

            # Create projection grid
            grid_lats, grid_lons = create_spherical_grid(self.nlat, self.nlon)

            # Process each eigenmode
            with Progress() as progress:
                mode_task = progress.add_task("[cyan]revERP: Processing eigenmodes...", total=n_modes)

                for mode_idx in range(n_modes):
                    try:
                        # Get displacement field for this mode
                        op_extractor.inputs.label_space.connect({"time": mode_idx + 1})
                        current_Dn_fc = op_extractor.outputs.fields_container()
                        if len(current_Dn_fc) == 1:
                            dn = current_Dn_fc[0].data
                        elif len(current_Dn_fc) == 2:
                            dn = current_Dn_fc[0].data + 1j * current_Dn_fc[1].data

                        mode = [dn[node_id_to_index[id]] for id in node_ids]
                        if self.config.get('SDEM_projection').get('SDEM', False):
                            mode = np.array(mode)[mesh_data["v_used"]]

                        # Project onto spherical grid
                        grid_values = griddata(
                            (lats, lons),
                            mode,
                            (grid_lats, grid_lons),
                            method='nearest'
                        )

                        # Compute spherical harmonics
                        grid = pysh.SHGrid.from_array(grid_values)
                        clm = grid.expand(normalization='ortho')

                        # 5. Compute weights
                        norms = np.linalg.norm(clm.coeffs, axis=0)
                        norms_sum = np.array([
                            np.linalg.norm(norms[l, :], axis=0)
                            for l in range(self.n_harmonics)
                        ])

                        weights = norms_sum / np.sum(norms_sum)

                        # Analyze radiation efficiency across frequency range
                        for freq_idx, freq in enumerate(freq_range):
                            sigma_sh = calculate_sigma_sh(
                                clm,
                                sphere_data["radius"],
                                freq,
                                self.C
                            )

                            # Store results
                            sigma_eff_modes[mode_idx, freq_idx] = np.sum(sigma_sh * weights)

                        progress.update(mode_task, advance=1)

                    except Exception as e:
                        self.logger.error(f"Failed processing mode {mode_idx + 1}: {str(e)}")
                        raise

            # Calculate radiation efficiency at natural frequencies
            sigma_eff_natural = np.zeros(n_modes)
            weights_natural = np.zeros((n_modes, self.n_harmonics))
            for mode_idx in range(n_modes):
                natural_freq = eigen_frequencies[mode_idx]
                freq_idx = np.abs(freq_range - natural_freq).argmin()
                sigma_eff_natural[mode_idx] = sigma_eff_modes[mode_idx, freq_idx]


            return {
                "eigen_frequencies": eigen_frequencies,
                "frequency_range": freq_range,
                "sigma_effective_modes": sigma_eff_modes,  # Full frequency range for each mode
                "sigma_effective_natural": sigma_eff_natural,  # At natural frequencies only
                "weights_natural": weights_natural,  # At natural frequencies only
                # "grid": {
                #     "lats": grid_lats,
                #     "lons": grid_lons
                # }
            }

        except Exception as e:
            self.logger.error(f"Modal analysis failed: {str(e)}")
            raise