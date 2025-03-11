# reverp/main.py
from typing import List, Dict, Any
import logging
import argparse
import yaml
from pathlib import Path
from rich.console import Console
import matplotlib.pyplot as plt
from datetime import datetime
# from time import perf_counter

from .src.core.mesh_handler import MeshHandler
from .src.core.erp_calculator import ERPCalculator
from .src.core.harmonic_analyzer import HarmonicAnalyzer
from .src.core.modal_analyzer import ModalAnalyzer
from .src.core.result_handler import ResultHandler, RevERPResult
from .src.visualization.plotters import ResultPlotter
from .src.utils.validators import validate_config
from .src.utils.validators import validate_model
from .src.utils.validators import validate_named_selections
from .src.utils.logging import setup_logging
from .src.utils.timing import TimingTracker
from .src.core.spectral_analyzer import SpectralAnalyzer  # Add this import

DEFAULT_CONFIG_PATH = Path("C:/01_gitrepos/revERP2.0/reverp/misc/DEFAULT_CONFIG.yaml")

class RevERP:
    """Main class for RevERP analysis."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize RevERP analysis system."""
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        self.timer = TimingTracker()

        with self.timer.track("Initialization"):
            # Load configuration
            self.config = self._load_config(config_path)

            # Now set up logging
            setup_logging(self.config)

            # Initialize common components
            self.mesh_handler = MeshHandler(self.config)
            self.result_handler = ResultHandler(self.config)
            self.plotter = ResultPlotter(self.config)

            # Initialize analysis-specific components
            self.analysis_type = self.config["analysis"]["analysis_type"]
            if self.analysis_type == "harmonic":
                self._init_harmonic_analyzer()
            elif self.analysis_type == "modal":
                self._init_modal_analyzer()
            elif self.analysis_type == "spectral":  # Add this condition
                self._init_spectral_analyzer()
            else:
                raise ValueError(f"Unsupported analysis type: {self.analysis_type}")

            self.logger.info(f"RevERP initialized successfully: {self.analysis_type.title()} Analysis")

    def _cleanup_previous_runs(self) -> None:
        """Clean up files from previous runs if configured to do so."""
        if not self.config['analysis'].get('clear_before_run', False):
            return

        self.logger.info("Cleaning up files from previous runs")
        logging.shutdown()
        try:
            # Get paths from config
            results_dir = Path(self.config['analysis'].get('dump_dir', 'results'))
            plots_dir = Path(self.config['visualization'].get('plot_dir', 'plots'))
            log_dir = Path(self.config['analysis'].get('log_dir', 'logs'))

            # Function to safely clean a directory
            def clean_directory(path: Path, file_types: List[str]) -> None:
                if path.exists():
                    for file_type in file_types:
                        for file in path.glob(file_type):
                            try:
                                file.unlink()
                                self.logger.debug(f"Deleted: {file}")
                            except Exception as e:
                                self.logger.warning(f"Failed to delete {file}: {str(e)}")

            # Clean results directory (CSV files)
            clean_directory(results_dir, ["*.csv"])

            # Clean plots directory (common image formats)
            clean_directory(plots_dir, ["*.png", "*.jpg", "*.jpeg", "*.pdf"])

            # Clean log directory (log files)
            clean_directory(log_dir, ["*.log"])

        except Exception as e:
            self.logger.warning(f"Cleanup operation failed: {str(e)}")
            self.console.print(f"[yellow]Warning: Cleanup operation failed: {str(e)}[/yellow]")

    def _init_harmonic_analyzer(self):
        """Initialize components for harmonic analysis."""
        self.erp_calculator = ERPCalculator(self.config)
        self.analyzer = HarmonicAnalyzer(self.config)

    def _init_modal_analyzer(self):
        """Initialize components for modal analysis."""
        self.analyzer = ModalAnalyzer(self.config)

    def _init_spectral_analyzer(self):
        """Initialize components for spectral analysis."""
        self.analyzer = SpectralAnalyzer(self.config)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration."""
        try:
            # Load default config first
            config = _load_default_config()

            # Load user config
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)

            if user_config is None:
                raise ValueError("Empty configuration file")

            # Update with user values
            for section in config:
                if section in user_config:
                    if isinstance(config[section], dict):
                        config[section].update(user_config[section])

            # Validate the configuration
            validate_config(config)
            return config

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}")

    def run_analysis(
        self,
        model_path: str,
        named_selections: List[str],
        skip_plots: bool = False
    ) -> List[RevERPResult]:
        """Run the complete RevERP analysis pipeline."""
        try:
            self.console.print("[bold green]Starting RevERP analysis...[/bold green]")

            with self.timer.track("Overall Analysis"):
                # Cleanup previous runs
                self._cleanup_previous_runs()

                # Print a small header and the settings in the log
                self.logger.info("\n\n // ============================== RevERP Analysis Settings ==============================\n\n")
                for section, values in self.config.items():
                    self.logger.info(f" // [{section}]")
                    for key, val in values.items():
                        self.logger.info(f" //   {key} = {val}")
                self.logger.info("\n\n // ============================== Starting RevERP ==============================\n\n")


                # Initialize model and validate it and named selections
                with self.timer.track("Model Loading"):
                    model = self.mesh_handler.load_model(model_path)
                    valid_model = validate_model(model,self.analysis_type)
                    if not valid_model[0]:
                        error_msg = f"Invalid model setup: {valid_model[1]}"
                        self.logger.error(error_msg)
                        self.console.print(f"[bold red]{error_msg}[/bold red]")
                        raise ValueError(error_msg)


                    valid_ns = validate_named_selections(model, named_selections)
                    if not valid_ns:
                        raise ValueError("No valid named selections provided")



                # Process each named selection
                results = []
                for ns in valid_ns:
                    self.console.print(f"[yellow]Processing named selection: {ns}[/yellow]")
                    result = self._process_named_selection(model, ns)
                    results.append(result)
                    self.console.print(f"[green]Completed analysis for {ns}[/green]")

                # Post-process results
                self._handle_post_processing(results, skip_plots)

                # Print timing summary
                timing_summary = self.timer.get_summary()
                self.logger.info("\n\n" + timing_summary)
                self.console.print(f"\n\n[cyan]{timing_summary}[/cyan]\n\n")

            self.console.print("[bold green]Analysis completed successfully![/bold green]")
            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            self.console.print(f"[bold red]Analysis failed: {str(e)}[/bold red]")
            raise

    def _process_named_selection(self, model: Any, ns: str) -> RevERPResult:
        """Process a single named selection based on analysis type."""
        # Common preprocessing
        with self.timer.track(f"Preprocessing {ns} and loading data"):
            skin_mesh = self.mesh_handler.get_skin_mesh(model, ns)
            mesh_data = self.mesh_handler.process_mesh(model, skin_mesh)
            sphere_data = self.mesh_handler.fit_sphere(mesh_data["coor"])

        # Analysis type specific processing
        if self.analysis_type == "harmonic":
            with self.timer.track(f"ERP Calculation {ns}"):
                erp_results = self.erp_calculator.compute_erp(model, skin_mesh, mesh_data)

            with self.timer.track(f"Harmonic Analysis {ns}"):
                analyzer_results = self.analyzer.analyze(mesh_data, erp_results, sphere_data)

            with self.timer.track(f"Results Processing {ns}"):
                return self.result_handler.process_harmonic_results(
                    ns, mesh_data, erp_results, analyzer_results, sphere_data
                )
        elif self.analysis_type == "spectral":
            with self.timer.track(f"Spectral Analysis {ns}"):
                analyzer_results = self.analyzer.analyze(mesh_data, sphere_data)

            with self.timer.track(f"Results Processing {ns}"):
                return self.result_handler.process_spectral_results(
                    ns, mesh_data, analyzer_results, sphere_data
                )
        else:
            with self.timer.track(f"Modal Analysis {ns}"):
                analyzer_results = self.analyzer.analyze(mesh_data, sphere_data)

            with self.timer.track(f"Results Processing {ns}"):
                return self.result_handler.process_modal_results(
                    ns, mesh_data, analyzer_results, sphere_data
                )

    def _handle_post_processing(self, results: List[RevERPResult], skip_plots: bool):
        """Handle result post-processing and visualization."""
        # Combine results if multiple selections
        if len(results) > 1:
            combined_result = self.result_handler.combine_results(results)
            results.append(combined_result)

        # Dump results if configured
        if self.config['analysis'].get('dump_results'):
            dump_dir = self.config['analysis'].get('dump_dir', 'results')
            self.result_handler.dump_results_to_csv(results, dump_dir)

        # Generate visualizations if not skipped
        if not skip_plots:
            viz_config = self.config.get('visualization', {})
            self._generate_plots(results, viz_config)

    def _generate_plots(self, results: List[RevERPResult], viz_config: Dict):
        """Generate visualization plots based on configuration and analysis type."""
        if not results:
            return

        if (self.analysis_type == "harmonic"):
            # Generate harmonic analysis plots
            if viz_config.get('result_summary_plot'):
                self.plotter.plot_summary(results)
            if viz_config.get('erp_plot'):
                self.plotter.plot_erp_comparison(results)

        elif self.analysis_type == "modal":
            # Generate modal analysis plots
            if viz_config.get('modal_summary_plot'):
                self.plotter.plot_modal_summary(results)
            if viz_config.get('mode_comparison_plot'):
                self.plotter.plot_mode_comparison(results)
            if viz_config.get('frequency_sweep_plot'):
                self.plotter.plot_frequency_sweep(results)

        # elif self.analysis_type == "spectral":
        #     if viz_config.get('spectral_summary_plot'):
        #         self.plotter.plot_spectral_summary(results)
        #     if viz_config.get('spectral_comparison_plot'):
        #         self.plotter.plot_spectral_comparison(results)

        if viz_config.get('SDEM_plot'):
            self.plotter.plot_sdem_meshes(results)

        # Handle plot saving if configured
        if viz_config.get('save_plots'):
            plot_dir = Path(viz_config.get('plot_dir', 'plots'))
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Configure plot saving parameters
            plt.rcParams['figure.dpi'] = viz_config.get('dpi')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save all active figures
            for i, fig in enumerate(plt.get_fignums()):
                figure = plt.figure(fig)
                plot_type = f"plot_{i+1}"
                filename = f"{self.analysis_type}_{plot_type}_{timestamp}.png"
                figure.savefig(plot_dir / filename, bbox_inches='tight')

def _load_default_config() -> Dict[str, Any]:
    """Load default configuration from YAML file."""
    try:
        with open(DEFAULT_CONFIG_PATH, 'r') as f:
            default_config = yaml.safe_load(f)
            if default_config is None:
                raise ValueError("Empty default configuration file")
            return default_config
    except Exception as e:
        raise ValueError(f"Failed to load default configuration: {str(e)}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='RevERP - Radiation Efficiency Varying Equivalent Radiated Power Analysis'
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to the ANSYS result file (.rst)'
    )
    parser.add_argument(
        '--ns', '-n',
        required=True,
        help='Named selections to analyze (comma-separated)'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--type',
        choices=['harmonic', 'modal'],
        help='Analysis type (overrides config file setting)'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Skip cleaning up previous results and logs'
    )
    return parser.parse_args()

def main():
    """Entry point for the RevERP analysis."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Load config and override analysis type if specified
        config_path = Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Initialize RevERP
        reverp = RevERP(str(config_path))

        # Override cleanup setting if specified in command line
        if args.no_cleanup:
            reverp.config["analysis"]["clear_before_run"] = False

        # Override analysis type if specified in command line
        if args.type:
            reverp.config["analysis"]["analysis_type"] = args.type
            reverp.analysis_type = args.type
            if args.type == "harmonic":
                reverp._init_harmonic_analyzer()
            else:
                reverp._init_modal_analyzer()

        # Split named selections
        named_selections = [ns.strip() for ns in args.ns.split(',')]

        # Run analysis
        results = reverp.run_analysis(
            model_path=args.model,
            named_selections=named_selections,
            skip_plots=args.no_plots
        )
        logging.shutdown()  # Ensures the log file is freed
        return results

    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        logging.error("Analysis failed", exc_info=True)
        raise

if __name__ == "__main__":
    main()