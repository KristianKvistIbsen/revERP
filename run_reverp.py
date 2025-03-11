# run_reverp.py
import sys
from pathlib import Path
import os

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

from reverp.main import RevERP

def main():
    # Get absolute path to config.yaml
    config_path = os.path.join(project_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Initialize RevERP
    model_path = r"N:\generic_acoustic_model_files\dp0\SYS\MECH\file.rst"
    named_selections = ["FSI"]

    print(f"Using config file: {config_path}")
    reverp = RevERP(config_path)

    results = reverp.run_analysis(
        model_path=model_path,
        named_selections=named_selections
    )

    return results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise