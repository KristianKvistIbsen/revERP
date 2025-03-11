# reverp/core/harmonic_analyzer.py
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

class HarmonicAnalyzer:
    """Handles spherical harmonic analysis."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize HarmonicAnalyzer with configuration."""
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
        self.REFERENCE_POWER = physics_config.get("reference_power")

    def analyze(
        self,
        mesh_data: Dict[str, Any],
        erp_data: Dict[str, Any],
        sphere_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform spherical harmonic analysis for each frequency.
        """
        try:
            self.logger.info("Performing harmonic analysis")

            frequencies = mesh_data["tfreq"].data
            n_frequencies = len(frequencies)

            # INITIALIZE ARRAYS
            sigma_eff = np.zeros(n_frequencies)
            weights_array = np.zeros((n_frequencies, self.n_harmonics))
            revERP = np.zeros(n_frequencies)
            revERP_db = np.zeros(n_frequencies)
            # clm_array = np.zeros((n_frequencies, 2, self.n_harmonics, self.n_harmonics), dtype=np.complex128)

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
            lons = 180 + np.degrees(sphere_coords[:, 5])

            # Create projection grid
            grid_lats, grid_lons = create_spherical_grid(self.nlat, self.nlon)

            # Process each frequency
            with Progress() as progress:
                task = progress.add_task("[cyan]revERP: Processing frequencies...", total=n_frequencies)

                for id_f, freq in enumerate(frequencies):
                    try:
                        # 1. Get velocity data for this frequency
                        op_extractor.inputs.label_space.connect({"time": id_f + 1})
                        current_Vn_fc = op_extractor.outputs.fields_container()

                        vn = (current_Vn_fc[0].data + current_Vn_fc[1].data * 1j)

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
                        # clm_array[id_f] = clm.coeffs

                        # 4. Calculate radiation efficiencies
                        sigma_sh = calculate_sigma_sh(
                            clm,
                            sphere_data["radius"],
                            freq,
                            self.C
                        )

                        # 5. Compute weights
                        norms = np.linalg.norm(clm.coeffs, axis=0)
                        norms_sum = np.array([
                            np.linalg.norm(norms[l, :], axis=0)
                            for l in range(self.n_harmonics)
                        ])

                        weights = norms_sum / np.sum(norms_sum)

                        # Store results
                        weights_array[id_f] = weights
                        sigma_eff[id_f] = sum(sigma_sh * weights)

                        # Calculate revERP
                        revERP[id_f] = erp_data["erp"][id_f] * sigma_eff[id_f]
                        revERP_db[id_f] = 10 * np.log10(revERP[id_f] / self.REFERENCE_POWER)

                        # Update progress
                        progress.update(task, advance=1)

                    except Exception as e:
                        self.logger.error(f"Failed processing frequency {freq} Hz: {str(e)}")
                        raise

            return {
                "sigma_effective": sigma_eff,
                # "clm": clm_array,
                "weights": weights_array,
                "revERP": revERP,
                "revERP_db": revERP_db,
                # "grid": {
                    # "lats": grid_lats,
                    # "lons": grid_lons
                # }
            }

        except Exception as e:
            self.logger.error(f"Harmonic analysis failed: {str(e)}")
            raise


