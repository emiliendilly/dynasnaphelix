# PyElastica Multi-Wrap Rod Unwinding Simulation

## Quick Setup (Recommended)

This repository includes a one-click environment setup.

### Step 1 — Download files

Make sure you have in the same folder:

- `2_perv_snapping.py`
- `requirements.txt`
- `setup_pyelastica_env.command`

### Step 2 — Run setup

#### macOS (double click)

Just double-click:

```
setup_pyelastica_env.command
```

#### Terminal (macOS/Linux)

```bash
chmod +x setup_pyelastica_env.command
./setup_pyelastica_env.command
```

---

## What this script does

The launcher automatically:

1. Creates a virtual environment:
   ```
   .venv/
   ```

2. Activates it

3. Installs all dependencies from:
   ```
   requirements.txt
   ```

4. Leaves you inside an active terminal ready to run the simulation

---

## Running the simulation for perversion antiperversion snapping at fixed elongation : n experiment

Once the setup finishes, just run for n experiment at fixed elongation:

```bash
python3 2_perv_snapping_final.py   --lambda-bend 1.0   --gamma-twist .666  --elongation 0.7  --extra-unwind-turns 19 --output-dir outputs 
```

## Example of usage: fixed link, varying axial elongation `z`

To run a single simulation at one fixed target link, set `--link-points 1` and use the same value for `--link-min` and `--link-max`.

Example for fixed target link `Lk = -3.5`, then translating from `z = 0.5` to `z = 0.1`:

```bash
python link_stop_sweep.py \
  --lambda-bend 2.0 \
  --gamma-twist 1.0 \
  --link-min -12.5 \
  --link-max -12.5 \
  --link-points 1 \
  --z-init 0.5 \
  --z-final 0.1 \
  --output-dir outputs_fixed_link_Lk_minus_12p5
```
---



## Manual setup (alternative)

If you prefer manual control:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Notes

- The terminal stays open with the environment activated
- To exit later:
  ```
  deactivate
  ```
- The environment is fully isolated in `.venv`

---

## Simulation Summary

This project simulates:

- A multi-wrap elastic rod
- Stress-free initial configuration
- Controlled elongation
- Clamp-driven unwinding
- Link tracking and torque extraction

Outputs include:

```
outputs/
├── simulation_parameters.txt
├── link_history_all_steps.txt
├── unwinding_summary.txt
├── unwinding.gif
└── frames_txt/
```

---

## Dependencies

Installed automatically via:

```bash
pip install -r requirements.txt
```

Includes:

- numpy
- scipy
- matplotlib
- pyelastica
- numba
- pyvista
- vtk

---

## License

Add your license here.
