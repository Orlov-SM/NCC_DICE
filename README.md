# NCC-DICE Project

Python/Pyomo implementation of an NCC-modified DICE climate-economy model, with routines to compute and plot reachability sets (RS) under different mitigation constraints.

## Repository Structure

- `ncc_rs.py`: core model builder (`makeNCCModel`) with vanilla and RS objective modes.
- `RSroutines.py`: solver routines for boundary and RS sampling (grid/parallel/ray variants).
- `plotRS.py`: main script to generate RS point clouds and save convex-hull CSV outputs.
- `plotter.py`: plotting utilities for saved RS CSV data.
- `diceParams/`: exogenous parameter data (including non-CO2 forcing).
- `plots_data/`: generated RS boundary CSV files.
- `plots_png/`, `paper_plots/`: generated figures.

## Requirements

Python dependencies are pinned in `requirements.txt`:

- `matplotlib==3.9.2`
- `numpy==2.1.2`
- `pandas==2.2.3`
- `pyomo==6.8.2`
- `scipy==1.14.1`

Additionally, you need an NLP solver executable:

- `ipopt` available in your system `PATH` (used via `SolverFactory("ipopt")`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install IPOPT separately for your OS and verify:

```bash
ipopt -v
```

## Usage

Generate RS CSV files (default setup in `plotRS.py`):

```bash
python plotRS.py
```

Plot results from CSV files:

```bash
python plotter.py
```

## Outputs

- RS hull CSV files are written to `plots_data/` (for example: `NCC_mu1.20_tdev1.csv`).
- Figures are written to `plots_png/` by plotting scripts.

## Notes

- The model reads `diceParams/nonCO2_forcing.csv` using a relative path; run scripts from the project root.
- Computational cost depends heavily on `resol1`, `mu_max_range`, and solver tolerances in `plotRS.py` and `RSroutines.py`.
- There is currently no automated test suite in the repository.
