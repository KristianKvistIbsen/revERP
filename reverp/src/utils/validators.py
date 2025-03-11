# reverp/utils/validators.py
from typing import Dict, Any

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values.

    Raises:
        ValueError: If any configuration values are invalid
    """
    # Check required sections
    required_sections = ['physics', 'analysis']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate physics parameters
    physics = config['physics']
    if not isinstance(physics.get('density'), (float)) or physics['density'] <= 0:
        raise ValueError("Physics: density must be a positive number")
    if not isinstance(physics.get('speed_of_sound'), (float)) or physics['speed_of_sound'] <= 0:
        raise ValueError("Physics: speed_of_sound must be a positive number")
    if not isinstance(physics.get('reference_power'), (float)) or physics['reference_power'] <= 0:
        raise ValueError("Physics: reference_power must be a positive number")

    # Validate analysis parameters
    analysis = config['analysis']
    if not isinstance(analysis.get('spherical_harmonics'), int) or analysis['spherical_harmonics'] <= 0:
        raise ValueError("Analysis: spherical_harmonics must be a positive integer")
    if analysis.get('analysis_type') not in ['harmonic', 'spectral', 'modal']:
        raise ValueError("Analysis: analysis_type must be either 'harmonic' or 'modal'")
    if not isinstance(analysis.get('modal_frequency_resolution'), (int, float)) or analysis['modal_frequency_resolution'] <= 0:
        raise ValueError("Analysis: modal_frequency_resolution must be a positive number")
    if not isinstance(analysis.get('modal_frequency_range'), (int, float)) or analysis['modal_frequency_range'] <= 0:
        raise ValueError("Analysis: modal_frequency_range must be a positive number")
    if not isinstance(analysis.get('minimum_frequency'), (int, float)) or analysis['minimum_frequency'] < 0:
        raise ValueError("Analysis: minimum_frequency must be >= 0")
    if not isinstance(analysis.get('maximum_frequency'), (int, float)) or analysis['maximum_frequency'] < 0:
        raise ValueError("Analysis: maximum_frequency must be >= 0")
    if not isinstance(analysis.get('clear_before_run'), bool):
        raise ValueError("Analysis: clear_before_run must be a boolean")
    if not isinstance(analysis.get('dump_results'), bool):
        raise ValueError("Analysis: dump_results must be a boolean")
    if not isinstance(analysis.get('dump_dir'), str):
        raise ValueError("Analysis: dump_dir must be a string")

    sdem = config.get('SDEM_projection', {})
    if not isinstance(sdem.get('SDEM'), bool):
        raise ValueError("SDEM_projection: SDEM must be a boolean")
    if not isinstance(sdem.get('SDEM_itt'), int) or sdem['SDEM_itt'] <= 0:
        raise ValueError("SDEM_projection: SDEM_itt must be a positive integer")
    if not isinstance(sdem.get('SDEM_dt'), (int, float)) or sdem['SDEM_dt'] <= 0:
        raise ValueError("SDEM_projection: SDEM_dt must be a positive number")
    if not isinstance(sdem.get('SDEM_eps'), (int, float)) or sdem['SDEM_eps'] <= 0:
        raise ValueError("SDEM_projection: SDEM_eps must be a positive number")
    if sdem.get('shrink_wrap_stl') is not None and not isinstance(sdem['shrink_wrap_stl'], str):
        raise ValueError("SDEM_projection: shrink_wrap_stl must be a string or null")
    if not isinstance(sdem.get('shrink_wrap_map_filter_radius'), (int, float)) or sdem['shrink_wrap_map_filter_radius'] <= 0:
        raise ValueError("SDEM_projection: shrink_wrap_map_filter_radius must be a positive number")

    viz = config.get('visualization', {})
    if not isinstance(viz.get('save_plots'), bool):
        raise ValueError("Visualization: save_plots must be a boolean")
    if not isinstance(viz.get('plot_dir'), str):
        raise ValueError("Visualization: plot_dir must be a string")
    for key in ['result_summary_plot', 'erp_plot', 'modal_summary_plot', 'mode_comparison_plot', 'frequency_sweep_plot']:
        if key in viz and not isinstance(viz[key], bool):
            raise ValueError(f"Visualization: {key} must be a boolean")
    if not isinstance(viz.get('figure_size'), list) or len(viz['figure_size']) != 2:
        raise ValueError("Visualization: figure_size must be a list of length 2")
    if any(not isinstance(x, (int, float)) or x <= 0 for x in viz['figure_size']):
        raise ValueError("Visualization: all figure_size elements must be positive")
    if not isinstance(viz.get('dpi'), (int, float)) or viz['dpi'] <= 0:
        raise ValueError("Visualization: dpi must be a positive number")
    if not isinstance(viz.get('colormap'), str):
        raise ValueError("Visualization: colormap must be a string")

def validate_model(model,analysis_type) -> tuple[bool, str]:
    try:
        # Check unit system
        if model.metadata.result_info.unit_system != 'MKS: m, kg, N, s, V, A, degC':
            print("What the hell kind of a unit system is: f{model.metadata.result_info.unit_system}")
        else:
            print("MKS unit system detected")

        # Get analysis type from model
        rst_type = model.metadata.result_info.analysis_type.lower()

        # For harmonic analysis
        if analysis_type in ["harmonic", "spectral"]:
            if not ('msup' in rst_type or '___' in rst_type):
                return False, "Provided .rst file is not of type: Harmonic"

        # For modal analysis
        elif analysis_type == "modal":
            if not ('modal' in rst_type or '___' in rst_type):
                return False, "Provided .rst file is not of type: Modal"

        else:
            return False, f"Unsupported analysis type: {rst_type}"

        return True, ""

    except AttributeError as e:
        return False, f"Invalid model structure: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_named_selections(model: Any, named_selections: list[str]) -> list[str]:
    """
    Validate and filter named selections.

    Args:
        model: The model containing named selections
        named_selections: List of named selections to validate

    Returns:
        List of valid named selections
    """
    available_ns = [ns for ns in model.metadata.available_named_selections
                    if not ns.startswith('_')]

    valid_ns = [ns for ns in named_selections if ns in available_ns]

    if len(valid_ns) != len(named_selections):
        invalid_ns = set(named_selections) - set(valid_ns)
        print(f"Warning: Invalid named selections: {invalid_ns}")

    return valid_ns