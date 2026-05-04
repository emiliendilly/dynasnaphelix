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

certifi==2026.1.4
charset-normalizer==3.4.4
contourpy==1.3.0
cycler==0.12.1
fonttools==4.60.2
idna==3.11
importlib_resources==6.5.2
kiwisolver==1.4.7
llvmlite==0.40.1
matplotlib==3.9.4
numba==0.57.1
numpy==1.24.4
packaging==26.0
pandas==2.3.3
pillow==11.3.0
platformdirs==4.4.0
pooch==1.9.0
pyelastica==0.3.2
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytz==2025.2
pyvista==0.39.1
requests==2.32.5
scipy==1.13.1
scooby==0.11.0
six==1.17.0
tqdm==4.67.3
tzdata==2025.3
urllib3==2.6.3
vtk==9.5.2
zipp==3.23.0


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

