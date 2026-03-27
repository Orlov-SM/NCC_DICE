# Windows Handoff

This file summarizes the recent changes added on branch `feature/scaled-area-loss`
so work can continue on another machine without relying on chat history.

## Recommended git workflow

Yes: keep these changes on a dedicated branch and push that branch to GitHub.

Suggested flow from the current Mac workspace:

```bash
git checkout feature/scaled-area-loss
git add scaling_utils.py geometry_utils.py compensation_analysis.py plotter.py requirements.txt .gitignore WINDOWS_HANDOFF.md
git commit -m "Add scaled-area compensation analysis for mu_max"
git push -u origin feature/scaled-area-loss
```

Notes:

- The working tree also contains other modified files such as `plotRS.py`,
  `.vscode/settings.json`, `scenario_config.py`, and some `plots_data/*.csv`.
- Those may be intentional local/project changes, but they are separate from the
  new compensation-analysis layer. Review them before including them in the same commit.
- Do not commit `.mplconfig/`, `.venv313`, or `.DS_Store`; they are now ignored.

## New files added

- `scaling_utils.py`
- `geometry_utils.py`
- `compensation_analysis.py`
- `WINDOWS_HANDOFF.md`

## Existing files changed

- `plotter.py`
- `requirements.txt`
- `.gitignore`

Other files in the worktree may also be modified, but they are not required for
the new scaled-area loss feature itself.

## What was implemented

### 1. Common scaling for comparisons

`scaling_utils.py` adds:

- `compute_global_scaling_bounds(...)`
- `normalize_points(...)`
- `denormalize_points(...)`

Important behavior:

- scaling is computed from **all borders in one comparison together**
- each scenario is **not** scaled independently
- both raw and scaled coordinates remain available

### 2. Geometry layer for reachable-set borders

`geometry_utils.py` adds:

- polygon construction from existing border point order
- polygon validity diagnostics
- area, intersection, and difference helpers
- relative area-loss computation

Important safety behavior:

- the code does **not** reorder border points
- the code does **not** silently replace a border with its convex hull
- if a border is invalid, the code raises an explicit error unless repair is
  deliberately enabled

### 3. Compensation analysis for `mu_max`

`compensation_analysis.py` adds the first complete analysis path for `mu_max`:

- load baseline and compensated borders for one `t_dev`
- scale both with one common scaling
- build polygons in scaled coordinates
- compute

  `Loss(R_base, R_comp) = area(R_base \\ R_comp) / area(R_base)`

- save publication/debug plots:
  - raw comparison
  - scaled comparison
  - shaded lost region
  - loss-vs-`mu_max` plot

### 4. Optional scaled plotting in the old plotter

`plotter.py` now has:

- `PLOT_IN_SCALED_COORDS = False`

When set to `True`, each output figure uses one common scaling across all series
plotted in that figure.

Default behavior remains the old raw-coordinate plotting.

## Validation already run

The following checks were run successfully in the project venv:

```bash
python -m py_compile plotter.py compensation_analysis.py scaling_utils.py geometry_utils.py
python compensation_analysis.py
python plotter.py
```

Observed debug output from `compensation_analysis.py` included:

- self-comparison loss approximately zero
- positive loss when comparing a larger set against a smaller one

## Output files produced by the new analysis

Saved under `plots_png/compensation/`:

- `raw_mu1.20_vs_1.40_tdev5.png`
- `scaled_mu1.20_vs_1.40_tdev5.png`
- `loss_region_mu1.20_vs_1.40_tdev5.png`
- `mu_max_relative_loss.png`

These output images are ignored by git because `plots_png/` is already ignored.

## Windows setup instructions

Use Python 3.13 if possible. Earlier work in this repo showed that Python 3.14
can fail on pinned scientific packages such as matplotlib.

From PowerShell on Windows:

```powershell
git clone https://github.com/Orlov-SM/NCC_DICE.git
cd NCC_DICE
git checkout feature/scaled-area-loss
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

You also need IPOPT available on `PATH`.

Quick check:

```powershell
ipopt -v
python -c "import shapely, numpy, scipy, pyomo; print('ok')"
```

## How to use the new functionality

### Run the new compensation analysis

```powershell
python compensation_analysis.py
```

This will:

- generate the raw comparison plot
- generate the scaled comparison plot
- generate the shaded lost-region plot
- compute and print polygon areas and loss values
- generate the `mu_max` relative-loss summary plot

### Use the legacy plotter in scaled mode

Open `plotter.py` and set:

```python
PLOT_IN_SCALED_COORDS = True
```

Then run:

```powershell
python plotter.py
```

### Main config points in `compensation_analysis.py`

- `BASELINE_MU_MAX`
- `COMPARE_MU_MAX_VALUES`
- `DEMO_COMPARED_MU_MAX`
- `DEMO_T_DEV`
- `REPAIR_INVALID_POLYGONS`

## Good first prompt for the next Codex agent

Use this on the Windows machine:

> Read `WINDOWS_HANDOFF.md`, inspect `scaling_utils.py`, `geometry_utils.py`,
> `compensation_analysis.py`, and `plotter.py`, then continue extending the
> compensation-analysis workflow from `mu_max` to the next scenario while keeping
> backward compatibility.
