# reverp/core/erp_calculator.py
from typing import Dict, Any, Optional
import logging
import numpy as np
from ansys.dpf import core as dpf
from ..lib.hansen_dpf_functions import compute_erp

class ERPCalculator:
    """Handles ERP calculations."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize ERPCalculator with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Get constants from config or use defaults
        self.RHO = self.config.get("physics", {}).get("density")
        self.C = self.config.get("physics", {}).get("speed_of_sound")
        self.REFERENCE_POWER = self.config.get("physics", {}).get("reference_power")

    def compute_erp(
        self,
        model: dpf.Model,
        skin_mesh: Any,  # This is actually a dpf.MeshedRegion
        mesh_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            self.logger.info("Computing ERP")
            print("\n\nComputing ERP")
            tfreq = model.metadata.time_freq_support.time_frequencies
            frequencies = tfreq.data
            if self.config.get("analysis",{}).get("analysis_type") == "harmonic":
                erp_fc = compute_erp(
                    model,
                    skin_mesh,
                    tfreq,
                    mesh_data["skin_mesh_scoping"],
                    self.RHO,
                    self.C,
                    self.REFERENCE_POWER
                )
                erp = erp_fc[0].data
            else:
                erp = np.zeros_like(frequencies)

            erp_db = 10 * np.log10(erp / self.REFERENCE_POWER)

            return {
                "erp": erp,
                "erp_db": erp_db,
                # "frequencies": frequencies,
                # "erp_fc": erp_fc
            }

        except Exception as e:
            self.logger.error(f"ERP calculation failed: {str(e)}")
            raise