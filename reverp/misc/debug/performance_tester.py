#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from reverp.main import RevERP

def run_performance_test(base_path: str = r"C:\Users\105849\Desktop\SCALAERP_harmonic",
                        # models: list = ['A', 'B', 'C', 'D', 'E'],
                        models: list = ['E', 'D', 'C', 'B', 'A'],
                        # models: list = ['D', 'E'],
                        n_runs: int = 5,
                        named_selections: list = ["FSI"]) -> pd.DataFrame:
    """
    Run performance tests on multiple models with repeated runs.

    Args:
        base_path: Base directory containing model folders
        models: List of model names/folders
        n_runs: Number of runs per model
        named_selections: List of named selections to analyze
    """
    console = Console()
    results = []

    # Create configuration
    config_path = "config.yaml"  # Assuming default config location

    # Initialize results storage
    all_times = {model: {
        'preprocessing': [],
        'erp_calculation': [],
        'harmonic_analysis': [],
        'results_processing': [],
        'total': []
    } for model in models}

    # Run analysis for each model
    for model in models:
        console.print(f"\n[bold cyan]Processing Model {model}[/bold cyan]")
        model_path = os.path.join(base_path, model, "file.rst")

        for run in range(n_runs):
            console.print(f"\n[yellow]Run {run + 1}/{n_runs}[/yellow]")

            # Initialize RevERP
            reverp = RevERP(config_path)

            # Run analysis
            try:
                _ = reverp.run_analysis(
                    model_path=model_path,
                    named_selections=named_selections,
                    skip_plots=True  # Skip plots for performance testing
                )

                # Extract timing data
                timing_data = reverp.timer.timings

                # Store relevant timings
                prefix = named_selections[0]  # Assuming single named selection for now
                all_times[model]['preprocessing'].append(timing_data.get(f'Preprocessing {prefix}', 0))
                all_times[model]['erp_calculation'].append(timing_data.get(f'ERP Calculation {prefix}', 0))
                all_times[model]['harmonic_analysis'].append(timing_data.get(f'Harmonic Analysis {prefix}', 0))
                all_times[model]['results_processing'].append(timing_data.get(f'Results Processing {prefix}', 0))
                all_times[model]['total'].append(timing_data.get('Overall Analysis', 0))

            except Exception as e:
                console.print(f"[bold red]Error processing model {model} (run {run + 1}): {str(e)}[/bold red]")
                continue

    # Calculate statistics
    stats_df = pd.DataFrame()
    for model in models:
        model_stats = {}
        for metric in all_times[model]:
            times = all_times[model][metric]
            if times:  # Check if we have any successful runs
                model_stats[f'{metric}_mean'] = np.mean(times)
                model_stats[f'{metric}_std'] = np.std(times)
            else:
                model_stats[f'{metric}_mean'] = np.nan
                model_stats[f'{metric}_std'] = np.nan
        stats_df = pd.concat([stats_df, pd.DataFrame([model_stats], index=[model])])

    # Display results
    console.print("\n[bold green]Performance Test Results:[/bold green]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model")
    table.add_column("Total Time (s)", justify="right")
    table.add_column("Preprocessing (s)", justify="right")
    table.add_column("ERP Calc (s)", justify="right")
    table.add_column("Harmonic (s)", justify="right")
    table.add_column("Processing (s)", justify="right")

    for model in models:
        table.add_row(
            model,
            f"{stats_df.loc[model, 'total_mean']:.2f} ± {stats_df.loc[model, 'total_std']:.2f}",
            f"{stats_df.loc[model, 'preprocessing_mean']:.2f} ± {stats_df.loc[model, 'preprocessing_std']:.2f}",
            f"{stats_df.loc[model, 'erp_calculation_mean']:.2f} ± {stats_df.loc[model, 'erp_calculation_std']:.2f}",
            f"{stats_df.loc[model, 'harmonic_analysis_mean']:.2f} ± {stats_df.loc[model, 'harmonic_analysis_std']:.2f}",
            f"{stats_df.loc[model, 'results_processing_mean']:.2f} ± {stats_df.loc[model, 'results_processing_std']:.2f}"
        )

    console.print(table)

    # Save results to CSV
    output_file = "performance_results.csv"
    stats_df.to_csv(output_file)
    console.print(f"\nResults saved to: {output_file}")

    return stats_df

if __name__ == "__main__":
    results = run_performance_test()