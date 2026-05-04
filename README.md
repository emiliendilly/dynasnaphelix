# PyElastica Multi-Wrap Rod Unwinding Simulation

Numerical experiment for simulating a stress-free multi-wrap elastic rod using PyElastica, including staged deformation, controlled elongation, and clamp-driven unwinding.

---

## Overview

This project simulates a Cosserat rod initialized as a multi-wrap circular coil, explicitly constructed and then set as the true stress-free configuration.

Stages:
1. Multi-wrap initialization
2. Stress-free assignment
3. Pull (±z)
4. Centering to z-axis
5. Endpoint alignment
6. Helix fitting
7. Target elongation
8. Clamp-driven unwinding

---

## Installation

pip install pyelastica numpy matplotlib pillow tqdm

Export environment:
python -m pip freeze > requirements.txt

---

## Usage

python multi_wrap_unwinding.py \
  --lambda-bend 2.0 \
  --gamma-twist 1.0 \
  --elongation 0.6 \
  --output-dir outputs

---

## Outputs

outputs/
├── simulation_parameters.txt
├── link_history_all_steps.txt
├── unwinding_summary.txt
├── unwinding.gif
└── frames_txt/

---

## Notes

- Explicit time stepping (PositionVerlet)
- Stable timestep selection
- Link tracking robust during unwinding
- Tw/Wr disabled by default

