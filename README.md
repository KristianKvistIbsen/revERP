# revERP: Radiation Efficiency Varying Equivalent Radiated Power
revERP is a python-based tool for early stage analysis of acoustic performance. The revERP method uses spherical harmonic decomposition [1] to provide improved Equivalent Radiated Power (ERP) calculations, and utilized area-preserving conformal mapping through Spherical Density Equalizing Mapping ported to Python from [2]. 

The SDEM and FLASH algorithms from [2] are originally provided in the MATLAB programming language under: 

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

This code has been ported to the python programming language and is in use in the current software, hence the Apache License Version 2.0 is still in force. The current author claims no novelty or rights to the SDEM or FLASH algorithms, and all novelty is attributed the original authors in [2].

The code is implemented using pyANSYS dpf-core [3], allowing for revERP to function directly with result files (file.rst) from ANSYS Mechanical [4]. A basic implementation of the method is presented in [5], showing excellent improvements over the classical ERP method. 

## Features

- Improved ERP calculation with spherical harmonic decomposition.
- Directly implemented for ANSYS Mechanical harmonic analysis (damped+undamped).
- Approximate radiation efficiency of mode shapes as a function of frequency (damped+undamped ANSYS modal analysis).
- Capable of analyzing several named selections simultaneously.
- Configurable physics parameters and analysis settings.
- Logging system.
- Result post processor.
- Quasi-isometric mapping approach for scalar field problems.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KristianKvistIbsen/revERP.git
cd revERP
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

RevERP uses a YAML configuration file (`config.yaml`) to manage various parameters:

```yaml
physics:
  density: 1.3  # kg/m^3
  speed_of_sound: 343.0  # m/s
  reference_power: 1.0e-12  # W

analysis:
  spherical_harmonics: 60  # Number of spherical harmonics

logging:
  level: "INFO"
  file: "reverp.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  ...
```

## Usage

### Basic Usage

```python
from reverp.main import RevERP

# Initialize RevERP with configuration
reverp = RevERP("config.yaml")

# Run analysis
results = reverp.run_analysis(
    model_path="path/to/your/model.rst",
    named_selections=["your_selection_name"],
    skip_plots=False
)
```

### Command Line Interface

```bash
python -m reverp.main --model path/to/model.rst --ns selection_name --config config.yaml
```

Options:
- `--model, -m`: Path to the ANSYS RST file
- `--ns, -n`: Named selections to analyze (comma-separated)
- `--config, -c`: Path to configuration file
- `--no-plots`: Skip generating plots

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or request features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Sources

[1] Mark A. Wieczorek and Matthias Meschede (2018). SHTools — Tools for working with spherical harmonics, Geochemistry, Geophysics, Geosystems, 19, 2574-2592, doi:10.1029/2018GC007529

[2] Lyu, Z., Lui, L., & Choi, G. (2024). Spherical Density-Equalizing Map for Genus-0 Closed Surfaces. SIAM Journal on Imaging Sciences, 17(4), 2110–2141.

[3] https://github.com/ansys/pydpf-core

[4] https://www.ansys.com/

[5] Kvist, K., Sorokin, S., & Larsen, J. (2025). Radiation efficiency varying equivalent radiated power. The Journal of the Acoustical Society of America, 157(1), 169-177.


## Authors

Kristian Kvist - Initial work - [KristianKvistIbsen](https://github.com/KristianKvistIbsen)

## Citation

If you use RevERP in your research, please cite:

```bibtex
@software{reverp2024,
  author = {Kvist, Kristian},
  title = {RevERP: Radiation Efficiency Varying Equivalent Radiated Power},
  year = {2025},
  url = {https://github.com/KristianKvistIbsen/revERP/}
}
```