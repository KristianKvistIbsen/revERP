# reverp/core/result_handler.py
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from dataclasses import dataclass
import csv
from pathlib import Path
from datetime import datetime

@dataclass
class RevERPResult:
    """Data class for storing RevERP analysis results."""
    analysis_type: str
    frequencies: np.ndarray
    model_info: Dict[str, Any]
    erp: Optional[np.ndarray] = None
    erp_db: Optional[np.ndarray] = None
    sigma_effective: Optional[np.ndarray] = None
    revERP: Optional[np.ndarray] = None
    revERP_db: Optional[np.ndarray] = None
    spectral: Optional[np.ndarray] = None
    spectral_db: Optional[np.ndarray] = None
    directivity: Optional[np.ndarray] = None
    grid_lats: Optional[np.ndarray] = None
    grid_lons: Optional[np.ndarray] = None
    weights_array: Optional[np.ndarray] = None
    clm: Optional[np.ndarray] = None
    skin_mesh: Optional[Any] = None,
    projected_mesh: Optional[Any] = None
    field_mesh: Optional[Any] = None

    eigen_frequencies: Optional[np.ndarray] = None
    frequency_range: Optional[np.ndarray] = None
    sigma_effective_modes: Optional[np.ndarray] = None
    sigma_effective_natural: Optional[np.ndarray] = None
    weights_natural: Optional[np.ndarray] = None

class ResultHandler:
    """Handles processing and storage of analysis results."""

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

    def dump_results_to_csv(self, results: List[RevERPResult], folder_path: str) -> None:
        """
        Dump all results to CSV files in the specified folder.

        Args:
            results: List of RevERPResult objects
            folder_path: Path to the output folder
        """
        try:
            # Create folder if it doesn't exist
            folder = Path(folder_path)
            folder.mkdir(parents=True, exist_ok=True)

            # Get timestamp for unique file names
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            self.logger.info(f"Dumping results to folder: {folder}")

            # Process each result
            for result in results:
                # Create filename based on named selection and timestamp
                ns_name = result.model_info.get("Named Selection")
                filename = f"revERP_{ns_name}_{timestamp}.csv"
                file_path = folder / filename

                self.logger.info(f"Writing result for {ns_name} to {file_path}")

                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)

                    # Write metadata header
                    writer.writerow(["RevERP Analysis Results"])
                    writer.writerow([f"Generated at: {datetime.now()}"])
                    writer.writerow([f"Analysis Type: {result.analysis_type}"])
                    writer.writerow([])

                    # Write model info
                    writer.writerow(["Model Information"])
                    for key, value in result.model_info.items():
                        writer.writerow([key, str(value)])
                    writer.writerow([])

                    if result.analysis_type == "harmonic":
                        self._write_harmonic_data(writer, result)
                    elif result.analysis_type == "spectral":
                        self._write_spectral_data(writer, result)
                    elif result.analysis_type == "modal":
                        self._write_modal_data(writer, result)
                    else:
                        raise ValueError(f"Unknown analysis type: {result.analysis_type}")

                self.logger.info(f"Successfully wrote results to {file_path}")

            # Create a summary file
            summary_path = folder / f"summary_{timestamp}.csv"
            self._write_summary(results, summary_path)

        except Exception as e:
            self.logger.error(f"Failed to dump results to CSV: {str(e)}")
            raise

    def _write_harmonic_data(self, writer: csv.writer, result: RevERPResult) -> None:
        """Write harmonic analysis specific data to CSV."""
        # Write main data header
        writer.writerow([
            "Frequency (Hz)",
            "ERP (W)",
            "ERP (dB)",
            "Sigma Effective",
            "revERP (W)",
            "revERP (dB)"
        ])

        # Write main results
        for i in range(len(result.frequencies)):
            writer.writerow([
                f"{result.frequencies[i]:.2f}",
                f"{result.erp[i]:.6e}",
                f"{result.erp_db[i]:.2f}",
                f"{result.sigma_effective[i]:.6f}",
                f"{result.revERP[i]:.6e}",
                f"{result.revERP_db[i]:.2f}"
            ])

        # Write separator
        writer.writerow([])

        # Write weights array if available
        if result.weights_array is not None:
            writer.writerow(["Spherical Harmonic Weights"])
            writer.writerow([f"Harmonic {i}" for i in range(result.weights_array.shape[1])])
            for i in range(len(result.frequencies)):
                row = [f"{weight:.6f}" for weight in result.weights_array[i]]
                writer.writerow(row)

    def _write_spectral_data(self, writer: csv.writer, result: RevERPResult) -> None:
        """Write spectral analysis specific data to CSV."""
        # Write main data header
        writer.writerow([
            "Frequency (Hz)",
            "spectral (W)",
            "spectral_db (dB)"
        ])

        # Write main results
        for i in range(len(result.frequencies)):
            writer.writerow([
                f"{result.frequencies[i]:.2f}",
                f"{result.spectral[i]:.6e}",
                f"{result.spectral_db[i]:.2f}"
            ])

        # Write separator
        writer.writerow([])

    def _write_modal_data(self, writer: csv.writer, result: RevERPResult) -> None:
        """Write modal analysis specific data to CSV."""
        # Write eigenfrequencies and their radiation efficiencies
        writer.writerow(["Modal Results"])
        writer.writerow([
            "Mode Number",
            "Eigenfrequency (Hz)",
            "Radiation Efficiency at Natural Frequency",
        ])

        for i in range(len(result.eigen_frequencies)):
            writer.writerow([
                i + 1,
                f"{result.eigen_frequencies[i]:.2f}",
                f"{result.sigma_effective_natural[i]:.6f}",
            ])

        writer.writerow([])

        # Write frequency-dependent radiation efficiencies for each mode
        writer.writerow(["Frequency-Dependent Radiation Efficiencies"])
        header = ["Frequency (Hz)"] + [f"Mode {i+1}" for i in range(result.sigma_effective_modes.shape[0])]
        writer.writerow(header)

        for i in range(len(result.frequency_range)):
            row = [f"{result.frequency_range[i]:.2f}"]
            row.extend([f"{result.sigma_effective_modes[j,i]:.6f}"
                       for j in range(result.sigma_effective_modes.shape[0])])
            writer.writerow(row)

        writer.writerow([])

        # Write weights if available
        if result.weights_natural is not None:
            writer.writerow(["Modal Weights at Natural Frequencies"])
            writer.writerow([f"Harmonic {i}" for i in range(result.weights_natural.shape[1])])
            for i in range(result.weights_natural.shape[0]):
                row = [f"{weight:.6f}" for weight in result.weights_natural[i]]
                writer.writerow(row)

    def _write_summary(self, results: List[RevERPResult], file_path: Path) -> None:
        """Write a summary file containing key information from all results."""
        try:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)

                writer.writerow(["RevERP Analysis Summary"])
                writer.writerow([f"Generated at: {datetime.now()}"])
                writer.writerow([])

                for result in results:
                    ns_name = result.model_info.get("Named Selection", "unnamed")
                    writer.writerow([f"Results for: {ns_name}"])
                    writer.writerow(["Metric", "Value"])

                    # Write analysis type specific summary
                    if result.analysis_type == "harmonic":
                        self._write_harmonic_summary(writer, result)
                    elif result.analysis_type == "spectral":
                        self._write_spectral_summary(writer, result)
                    elif result.analysis_type == "modal":
                        self._write_modal_summary(writer, result)
                    writer.writerow([])

        except Exception as e:
            self.logger.error(f"Failed to write summary file: {str(e)}")
            raise

    def _write_harmonic_summary(self, writer: csv.writer, result: RevERPResult) -> None:
        """Write summary statistics for harmonic analysis."""
        mean_erp_db = np.mean(result.erp_db)
        max_erp_db = np.max(result.erp_db)
        mean_rev_erp_db = np.mean(result.revERP_db)
        max_rev_erp_db = np.max(result.revERP_db)
        mean_sigma = np.mean(result.sigma_effective)

        writer.writerow(["Mean ERP (dB)", f"{mean_erp_db:.2f}"])
        writer.writerow(["Max ERP (dB)", f"{max_erp_db:.2f}"])
        writer.writerow(["Mean revERP (dB)", f"{mean_rev_erp_db:.2f}"])
        writer.writerow(["Max revERP (dB)", f"{max_rev_erp_db:.2f}"])
        writer.writerow(["Mean Sigma Effective", f"{mean_sigma:.6f}"])

    def _write_spectral_summary(self, writer: csv.writer, result: RevERPResult) -> None:
        """Write summary statistics for spectral analysis."""


    def _write_modal_summary(self, writer: csv.writer, result: RevERPResult) -> None:
        """Write summary statistics for modal analysis."""
        n_modes = len(result.eigen_frequencies)
        mean_eigen_freq = np.mean(result.eigen_frequencies)
        max_eigen_freq = np.max(result.eigen_frequencies)
        mean_sigma_natural = np.mean(result.sigma_effective_natural)
        max_sigma_natural = np.max(result.sigma_effective_natural)

        writer.writerow(["Number of Modes", str(n_modes)])
        writer.writerow(["Mean Eigenfrequency (Hz)", f"{mean_eigen_freq:.2f}"])
        writer.writerow(["Max Eigenfrequency (Hz)", f"{max_eigen_freq:.2f}"])
        writer.writerow(["Mean Natural Radiation Efficiency", f"{mean_sigma_natural:.6f}"])
        writer.writerow(["Max Natural Radiation Efficiency", f"{max_sigma_natural:.6f}"])

    def process_harmonic_results(
        self,
        named_selection: str,
        mesh_data: Dict[str, Any],
        erp_results: Dict[str, Any],
        harmonic_results: Dict[str, Any],
        sphere_data: Dict[str, Any]
    ) -> RevERPResult:
        """Process results from harmonic analysis."""
        try:
            self.logger.info(f"Processing harmonic results for {named_selection}")

            # Compile model information
            model_info = self._create_model_info(
                named_selection, mesh_data, sphere_data, "harmonic"
            )

            return RevERPResult(
                analysis_type="harmonic",
                frequencies=mesh_data["tfreq"].data,
                erp=erp_results["erp"],
                erp_db=erp_results["erp_db"],
                sigma_effective=harmonic_results["sigma_effective"],
                revERP=harmonic_results["revERP"],
                revERP_db=harmonic_results["revERP_db"],
                weights_array=harmonic_results["weights"],
                model_info=model_info,
                skin_mesh=mesh_data["skin_mesh"],
                projected_mesh=mesh_data["spherical_skin_mesh"],
                field_mesh=mesh_data["field_mesh"]
            )

        except Exception as e:
            self.logger.error(f"Failed to process harmonic results: {str(e)}")
            raise

    def process_spectral_results(
        self,
        named_selection: str,
        mesh_data: Dict[str, Any],
        spectral_results: Dict[str, Any],
        sphere_data: Dict[str, Any]
    ) -> RevERPResult:
        """Process results from harmonic analysis."""
        try:
            self.logger.info(f"Processing harmonic results for {named_selection}")

            # Compile model information
            model_info = self._create_model_info(
                named_selection, mesh_data, sphere_data, "harmonic"
            )

            return RevERPResult(
                analysis_type="spectral",
                frequencies=mesh_data["tfreq"].data,
                spectral=spectral_results["spectral"],
                spectral_db=spectral_results["spectral_db"],
                model_info=model_info,
                skin_mesh=mesh_data["skin_mesh"],
                projected_mesh=mesh_data["spherical_skin_mesh"],
                field_mesh=mesh_data["field_mesh"]
            )

        except Exception as e:
            self.logger.error(f"Failed to process spectral results: {str(e)}")
            raise

    def process_modal_results(
        self,
        named_selection: str,
        mesh_data: Dict[str, Any],
        modal_results: Dict[str, Any],
        sphere_data: Dict[str, Any]
    ) -> RevERPResult:
        """Process results from modal analysis."""
        try:
            self.logger.info(f"Processing modal results for {named_selection}")

            # Compile model information
            model_info = self._create_model_info(
                named_selection, mesh_data, sphere_data, "modal"
            )

            return RevERPResult(
                analysis_type="modal",
                frequencies=mesh_data["tfreq"].data,
                eigen_frequencies=modal_results["eigen_frequencies"],
                frequency_range=modal_results["frequency_range"],
                sigma_effective_modes=modal_results["sigma_effective_modes"],
                sigma_effective_natural=modal_results["sigma_effective_natural"],
                weights_natural=modal_results["weights_natural"],
                model_info=model_info,
                skin_mesh=mesh_data["skin_mesh"],
                projected_mesh=mesh_data["spherical_skin_mesh"],
                field_mesh=mesh_data["field_mesh"]
            )

        except Exception as e:
            self.logger.error(f"Failed to process modal results: {str(e)}")
            raise

    def _create_model_info(
        self,
        named_selection: str,
        mesh_data: Dict[str, Any],
        sphere_data: Dict[str, Any],
        analysis_type: str
    ) -> Dict[str, Any]:
        """Create model information dictionary."""
        model_info = {
            "Named Selection": named_selection,
            "Analysis Type": analysis_type,
            "Number of Nodes": mesh_data["n_nodes"],
            "Sphere Radius": sphere_data["radius"],
            "Sphere Center": sphere_data["center"],
            "Spherical Harmonics": self.config["analysis"]["spherical_harmonics"]
        }

        if analysis_type == "modal":
            model_info.update({
                "Frequency Resolution": self.config["analysis"]["modal_frequency_resolution"],
                "Frequency Range Factor": self.config["analysis"]["modal_frequency_range"]
            })

        return model_info

    def combine_results(self, results: List[RevERPResult]) -> RevERPResult:
        """Combine results from multiple named selections."""
        try:
            self.logger.info("Combining results from multiple selections")

            if not results:
                raise ValueError("No results to combine")

            analysis_type = results[0].analysis_type
            if not all(r.analysis_type == analysis_type for r in results):
                raise ValueError("Cannot combine results from different analysis types")

            if analysis_type == "harmonic":
                return self._combine_harmonic_results(results)
            elif analysis_type == "spectral":
                return self._combine_spectral_results(results)
            elif analysis_type == "modal":
                return self._combine_modal_results(results)
            else:
                raise ValueError("Analysis type not recognized")

        except Exception as e:
            self.logger.error(f"Failed to combine results: {str(e)}")
            raise

    # def _combine_harmonic_results(self, results: List[RevERPResult]) -> RevERPResult:
    #     """Combine harmonic analysis results."""
    #     frequencies = results[0].frequencies
    #     total_erp = np.sum([r.erp for r in results], axis=0)
    #     total_revERP = np.sum([r.revERP for r in results], axis=0)

    #     # Calculate combined dB values
    #     total_erp_db = 10 * np.log10(total_erp / self.config["physics"]["reference_power"])
    #     total_revERP_db = 10 * np.log10(total_revERP / self.config["physics"]["reference_power"])

    #     return RevERPResult(
    #         analysis_type="harmonic",
    #         frequencies=frequencies,
    #         erp=total_erp,
    #         erp_db=total_erp_db,
    #         sigma_effective=np.mean([r.sigma_effective for r in results], axis=0),
    #         revERP=total_revERP,
    #         revERP_db=total_revERP_db,
    #         weights_array=np.mean([r.weights_array for r in results], axis=0),
    #         model_info=self._create_combined_info(results)
    #     )

    # def _combine_modal_results(self, results: List[RevERPResult]) -> RevERPResult:
    #     """Combine modal analysis results."""
    #     # For modal analysis, we'll combine the frequency ranges and efficiencies
    #     frequency_range = results[0].frequency_range
    #     return RevERPResult(
    #         analysis_type="modal",
    #         frequencies=frequency_range,
    #         frequency_range=frequency_range,
    #         eigen_frequencies=np.concatenate([r.eigen_frequencies for r in results]),
    #         sigma_effective_modes=np.vstack([r.sigma_effective_modes for r in results]),
    #         sigma_effective_natural=np.concatenate([r.sigma_effective_natural for r in results]),
    #         weights_natural=np.vstack([r.weights_natural for r in results]),
    #         model_info=self._create_combined_info(results)
    #     )

    # def _create_combined_info(self, results: List[RevERPResult]) -> Dict[str, Any]:
    #     """Create model info for combined results."""
    #     return {
    #         "Named Selection": "Combined Result - weights_array and sigma_effective are averaged values",
    #         "Analysis Type": results[0].analysis_type,
    #         "Number of Named Selections combined": len(results),
    #         "Component Names": [r.model_info["Named Selection"] for r in results]
    #     }

