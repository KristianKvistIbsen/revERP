# reverp/visualization/plotters.py
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from ..core.result_handler import RevERPResult

class ResultPlotter:
    """Handles result visualization for both harmonic and modal analysis."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize ResultPlotter with configuration."""
        self.console = Console()
        self.config = config or {}

        # Get visualization settings
        viz_config = self.config.get('visualization', {})
        self.figure_size = viz_config.get('figure_size', [12, 8])
        self.dpi = viz_config.get('dpi', 100)
        self.colormap = viz_config.get('colormap', 'viridis')

        # Configure matplotlib
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['figure.dpi'] = self.dpi

    def plot_summary(self, results: List[RevERPResult]):
        """Create summary plots based on analysis type."""
        if not results:
            return

        if results[0].analysis_type == "modal":
            self.plot_modal_summary(results)
        else:
            self._plot_harmonic_summary(results)

    def _plot_harmonic_summary(self, results: List[RevERPResult]):
        """Create summary plots for harmonic analysis results."""
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        fig.suptitle('Harmonic Analysis Results', fontsize=16)

        for result in results:
            # 1. ERP and Corrected ERP
            axs[0, 0].semilogx(
                result.frequencies,
                result.erp_db,
                '--',
                label=f'ERP ({result.model_info["Named Selection"]})'
            )
            axs[0, 0].semilogx(
                result.frequencies,
                result.revERP_db,
                label=f'revERP ({result.model_info["Named Selection"]})'
            )
            axs[0, 0].set_xlabel('Frequency (Hz)')
            axs[0, 0].set_ylabel('ERP (dB)')
            axs[0, 0].set_title('ERP Comparison')
            axs[0, 0].legend()
            axs[0, 0].grid(True)

            # 2. Weights vs Frequency
            if result.weights_array is not None:
                harmonic_index = range(0, result.weights_array.shape[1])
                color_range = plt.cm.get_cmap(self.colormap)(
                    np.linspace(0, 1, len(result.frequencies))
                )
                for id_f in range(len(result.frequencies)):
                    axs[0, 1].plot(
                        harmonic_index,
                        result.weights_array[id_f, :],
                        color=color_range[id_f]
                    )
                axs[0, 1].set_xlabel('Harmonic Index')
                axs[0, 1].set_ylabel('Harmonic Weight')
                axs[0, 1].set_title('Weights vs Frequency')
                axs[0, 1].grid(True)

            # 3. Radiation Efficiencies
            axs[1, 0].semilogx(
                result.frequencies,
                result.sigma_effective,
                label=f'({result.model_info["Named Selection"]})'
            )
            axs[1, 0].set_xlabel('Frequency (Hz)')
            axs[1, 0].set_ylabel('Radiation Efficiency')
            axs[1, 0].set_title('Radiation Efficiencies vs Frequency')
            axs[1, 0].legend()
            axs[1, 0].grid(True)

        # 4. Model Information Table
        axs[1, 1].axis('off')
        table_data = [[key, str(value)] for key, value in results[0].model_info.items()]
        table = axs[1, 1].table(cellText=table_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        axs[1, 1].set_title('Model Information')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_modal_summary(self, results: List[RevERPResult]):
        """Create summary plots for modal analysis results."""
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        fig.suptitle('Modal Analysis Results', fontsize=16)

        for result in results:
            if result.analysis_type != "modal":
                continue

            # 1. Radiation Efficiency vs Frequency for each mode
            for mode_idx in range(result.sigma_effective_modes.shape[0]):
                axs[0, 0].semilogx(
                    result.frequency_range,
                    result.sigma_effective_modes[mode_idx, :],
                    label=f'Mode {mode_idx + 1}'
                )
            axs[0, 0].set_xlabel('Frequency (Hz)')
            axs[0, 0].set_ylabel('Radiation Efficiency')
            axs[0, 0].set_title('Mode Radiation Efficiencies')
            axs[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs[0, 0].grid(True)

            # 2. Natural Frequencies and their Radiation Efficiencies
            axs[0, 1].scatter(
                result.eigen_frequencies,
                result.sigma_effective_natural,
                label=result.model_info["Named Selection"]
            )
            axs[0, 1].set_xlabel('Natural Frequency (Hz)')
            axs[0, 1].set_ylabel('Radiation Efficiency')
            axs[0, 1].set_title('Natural Frequencies Radiation Efficiency')
            axs[0, 1].grid(True)

            # 3. Modal Weights Distribution
            if result.weights_natural is not None:
                harmonic_index = range(result.weights_natural.shape[1])
                for mode_idx in range(result.weights_natural.shape[0]):
                    axs[1, 0].plot(
                        harmonic_index,
                        result.weights_natural[mode_idx, :],
                        label=f'Mode {mode_idx + 1}'
                    )
                axs[1, 0].set_xlabel('Harmonic Index')
                axs[1, 0].set_ylabel('Weight')
                axs[1, 0].set_title('Modal Weights Distribution')
                axs[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axs[1, 0].grid(True)

            # 4. Model Information Table
            axs[1, 1].axis('off')
            table_data = [[key, str(value)] for key, value in result.model_info.items()]
            table = axs[1, 1].table(cellText=table_data, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            axs[1, 1].set_title('Model Information')

        plt.tight_layout()
        plt.show()

    def plot_mode_comparison(self, results: List[RevERPResult]):
        """Create mode comparison plots."""
        for result in results:
            if result.analysis_type != "modal":
                continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'Mode Analysis: {result.model_info["Named Selection"]}', fontsize=16)

            # Plot 1: Natural Frequencies Distribution
            ax1.bar(
                range(1, len(result.eigen_frequencies) + 1),
                result.eigen_frequencies,
                alpha=0.7
            )
            ax1.set_xlabel('Mode Number')
            ax1.set_ylabel('Natural Frequency (Hz)')
            ax1.set_title('Natural Frequencies Distribution')
            ax1.grid(True)

            # Plot 2: Radiation Efficiency at Natural Frequencies
            ax2.bar(
                range(1, len(result.sigma_effective_natural) + 1),
                result.sigma_effective_natural,
                alpha=0.7
            )
            ax2.set_xlabel('Mode Number')
            ax2.set_ylabel('Radiation Efficiency')
            ax2.set_title('Modal Radiation Efficiency')
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

    def plot_frequency_sweep(self, results: List[RevERPResult]):
        """Create detailed frequency sweep visualization."""
        for result in results:
            if result.analysis_type != "modal":
                continue

            plt.figure(figsize=(12, 6))

            # Plot radiation efficiency curves for all modes
            color_range = plt.cm.get_cmap(self.colormap)(
                np.linspace(0, 1, result.sigma_effective_modes.shape[0])
            )
            for mode_idx in range(result.sigma_effective_modes.shape[0]):
                plt.semilogx(
                    result.frequency_range,
                    result.sigma_effective_modes[mode_idx, :],
                    alpha=0.5,
                    color=color_range[mode_idx],
                    label=f'Mode {mode_idx + 1}'
                )

            # Add markers for natural frequencies
            plt.scatter(
                result.eigen_frequencies,
                result.sigma_effective_natural,
                color='red',
                s=100,
                zorder=5,
                label='Natural Frequencies'
            )

            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Radiation Efficiency')
            plt.title(f'Frequency Sweep Analysis: {result.model_info["Named Selection"]}')
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

    def plot_erp_comparison(self, results: List[RevERPResult]):
        """Create ERP comparison plot for harmonic analysis."""
        if not results or results[0].analysis_type != "harmonic":
            return

        plt.figure(figsize=(10, 6))

        for result in results:
            plt.semilogx(
                result.frequencies,
                result.erp_db,
                '--',
                label=f'ERP ({result.model_info["Named Selection"]})'
            )
            plt.semilogx(
                result.frequencies,
                result.revERP_db,
                label=f'revERP ({result.model_info["Named Selection"]})'
            )

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ERP (dB)')
        plt.title('ERP and revERP Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_all_plots(self, results: List[RevERPResult], plot_dir: str = 'plots'):
        """Save all generated plots to the specified directory."""
        # This functionality is now handled in the main class's _generate_plots method
        pass

    def plot_sdem_meshes(self, results: List['RevERPResult']):
        import pyvista as pv
        for result in results:
            if not result.projected_mesh:
                continue
            p = pv.Plotter(shape=(1, 4))
            p.subplot(0, 0)
            p.add_mesh(result.skin_mesh.grid, color='red')
            p.add_scalar_bar('Skin Mesh', vertical=True)
            p.subplot(0, 1)
            p.add_mesh(result.field_mesh.grid, color='green')
            p.add_scalar_bar('Field Mesh', vertical=True)
            p.subplot(0, 2)
            p.add_mesh(result.projected_mesh.grid, color='blue')
            p.add_scalar_bar('Projected Mesh', vertical=True)
            p.subplot(0, 3)
            p.add_mesh(result.skin_mesh.grid, color='red')
            p.add_mesh(result.field_mesh.grid, color='green')
            p.add_mesh(result.projected_mesh.grid, color='blue')
            p.add_scalar_bar('Combined Meshes', vertical=True)
            p.show()