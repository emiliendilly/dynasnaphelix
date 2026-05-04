# PyElastica Multi-Wrap Rod Unwinding Simulation

Numerical experiment for simulating a stress-free multi-wrap elastic rod
using PyElastica, including staged deformation, controlled elongation,
and clamp-driven unwinding.

------------------------------------------------------------------------

## Overview

This project simulates a Cosserat rod initialized as a **multi-wrap
circular coil**, explicitly constructed and then set as the **true
stress-free configuration**.

The simulation proceeds through:

1.  Multi-wrap circular initialization\
2.  Stress-free rest state assignment\
3.  Pulling in ±z\
4.  Centering onto the z-axis\
5.  Endpoint frame alignment\
6.  Helix fitting\
7.  Target elongation\
8.  Clamp-driven unwinding

------------------------------------------------------------------------

## Key Features

-   Explicit multi-wrap geometry (no relaxation into shape)
-   Stress-free initialization (no pre-stress)
-   Anisotropic elasticity:
    -   `Lambda = B2 / B1`
    -   `Gamma = Bt / B1`
-   Helix axis fitting
-   Controlled elongation along fitted axis
-   Clamp-driven unwinding (rotation-controlled)
-   Robust link tracking:
    -   Iterative (stages 1--4)
    -   Direct (stage 5)
-   Full data export (per-frame + time history)
-   GIF visualization

------------------------------------------------------------------------

## Installation

``` bash
pip install pyelastica numpy matplotlib pillow tqdm
```

Export your environment:

``` bash
pip3 intall
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
```

------------------------------------------------------------------------

## Usage

``` bash
python multi_wrap_unwinding.py \
  --lambda-bend 2.0 \
  --gamma-twist 1.0 \
  --elongation 0.6 \
  --output-dir outputs \
  --extra-unwind-turns 15 \
  --sim-dt-per-frame 0.4
```

------------------------------------------------------------------------

## Parameters

### Required

  Argument          Description
  ----------------- --------------------------------------
  `--lambda-bend`   Bending anisotropy ratio ( B2 / B1 )
  `--gamma-twist`   Twist stiffness ratio ( Bt / B1 )
  `--elongation`    Target clamp separation / arc length

### Optional

  Argument                    Default           Description
  --------------------------- ----------------- -----------------------------------
  `--output-dir`              outputs           Output directory
  `--extra-unwind-turns`      15                Number of imposed unwinding turns
  `--sim-dt-per-frame`        0.4               Sampling interval
  `--torque-mode`             internal_couple   Torque quantity
  `--torque-offset-samples`   5                 Offset from rod ends

------------------------------------------------------------------------

## Numerical Experiment

### Geometry

-   Multi-turn circle: \[
    `\theta `{=tex}`\in [0, 2\pi \cdot n_{\text{turns}}]`{=tex}\]
-   Small z-perturbation avoids degeneracy.

### Default Parameters

  Parameter         Value
  ----------------- -----------
  Elements          120
  Turns             8
  Circle radius     0.0039 m
  Rod radius        0.001 m
  Density           500 kg/m³
  Young's modulus   3.0e4 Pa
  Poisson ratio     0.4

------------------------------------------------------------------------

## Stress-Free Initialization

The initial configuration is made stress-free:

    rest_lengths = current lengths
    rest_sigma   = current sigma
    rest_kappa   = current kappa

------------------------------------------------------------------------

## Material Model

Anisotropic bending aligned with natural curvature:

-   ( B1 ): along natural curvature direction\
-   ( B2 = `\lambda `{=tex}B1 ): orthogonal direction\
-   ( Bt = `\gamma `{=tex}B1 ): twist stiffness

------------------------------------------------------------------------

## Simulation Stages

### Stage 1 --- Pull

-   Ends pulled along ±z\
-   Free rotation

### Stage 2 --- Center

-   Ends moved to z-axis\
-   ( x = 0, y = 0 )

### Stage 3 --- Align

-   Left tangent → +z\
-   Right tangent → −z

### Stage 4 --- Target Elongation

Clamp separation set to:

    elongation × rod length

### Stage 5 --- Unwinding

-   Clamp positions fixed\
-   Directors rotated about z-axis

```{=html}
<!-- -->
```
    phi_extra = 2π × turns

Link update:

    Lk = Lk_start - phi_extra / (2π)

------------------------------------------------------------------------

## Outputs

    outputs/
    ├── simulation_parameters.txt
    ├── link_history_all_steps.txt
    ├── unwinding_summary.txt
    ├── unwinding.gif
    └── frames_txt/

------------------------------------------------------------------------

## Exported Data

### simulation_parameters.txt

Stores all parameters for reproducibility.

------------------------------------------------------------------------

### link_history_all_steps.txt

    time stage dpsi_left dpsi_right link_iterative

-   Tracks link per solver step\
-   During unwinding: direct update

------------------------------------------------------------------------

### unwinding_summary.txt

    time
    phi_extra
    left_torque
    right_torque (sign flipped)
    mean_I3/I2
    link
    Tw
    Wr
    Lk_closure

------------------------------------------------------------------------

### Per-frame Files

Each frame contains:

-   Geometry (position, arclength)
-   Strain (kappa, sigma)
-   Forces & moments
-   Lab/material representations
-   Kirchhoff invariants
-   Link values

------------------------------------------------------------------------

## Torque Analysis

Two modes:

    internal_couple
    internal_torques

Projected along fitted helix axis.

------------------------------------------------------------------------

## Kirchhoff Torque Estimate

\[ I_2 = \|\|n\|\|,`\quad `{=tex}I_3 = n
`\cdot `{=tex}m,`\quad `{=tex}`\tau `{=tex}= `\frac{I_3}{I_2}`{=tex} \]

Exported as:

    mean_I3_over_I2

------------------------------------------------------------------------

## Topology Tracking

### Iterative (Stages 1--4)

    Lk -= (dpsi_left + dpsi_right) / (2π)

### Direct (Stage 5)

    Lk = Lk_start - phi_extra / (2π)

------------------------------------------------------------------------

## Example Run

``` bash
python multi_wrap_unwinding.py \
  --lambda-bend 3.0 \
  --gamma-twist 1.5 \
  --elongation 0.65 \
  --extra-unwind-turns 12 \
  --output-dir outputs/run1
```

------------------------------------------------------------------------

## Analysis Example

``` python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("outputs/unwinding_summary.txt", skiprows=1)

phi = data[:,1]
torque = data[:,2]

plt.plot(phi, torque)
plt.xlabel("rotation")
plt.ylabel("torque")
plt.show()
```

------------------------------------------------------------------------

## Notes

-   Explicit time stepping (PositionVerlet)
-   Conservative timestep for stability
-   Live plotting enabled by default
-   GIF export enabled
-   Twist/Writhe disabled by default (NaN outputs)

------------------------------------------------------------------------

## Citation

PyElastica: https://github.com/GazzolaLab/PyElastica

``` bibtex
@article{zhang2019modeling,
  title={Modeling and simulation of complex dynamic musculoskeletal architectures},
  author={Zhang et al.},
  journal={Nature Communications},
  year={2019}
}
```

------------------------------------------------------------------------

## License

Add your license here (MIT, GPL, etc.)
