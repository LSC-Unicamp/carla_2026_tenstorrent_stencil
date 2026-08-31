# 3D Jacobi Stencil for Tenstorrent Blackhole

## Overview

This repository contains the reference implementation of the 3D Jacobi stencil kernel optimized for the Tenstorrent Blackhole architecture. This code accompanies the paper **"Shift Matrices and Single-Batch Halos: Optimizing 3D Stencils on Tenstorrent Blackhole Architecture"**, presented at CARLA 2026 (Córdoba, Argentina).

Our implementation leverages specific hardware features to accelerate 3D stencil computations, demonstrating significant speedups over existing approaches for the supported configurations. The codebase is provided to support the reproducibility of the paper's results.

*Note: As this is an active research repository, some internal code comments may remain in Portuguese.*

## Prerequisites

To build and run this project, ensure your system meets the following requirements:

* **Hardware:** Tenstorrent Blackhole p150a
* **Software:**
* [Tenstorrent TT-Metalium](https://github.com/tenstorrent/tt-metal) (version `v0.66.0`)
* CMake
* Python 3


* **Environment:** The `TT_METAL` environment variable must be properly set to your SDK installation path.

## Building

Clone the repository and build the project using CMake:

```bash
git clone https://github.com/LSC-Unicamp/carla_2026_tenstorrent_stencil
cd carla_2026_tenstorrent_stencil
cmake -S . -B build -G "Unix Makefiles"
cmake --build build -j

```

## Reproducing the Experiments

All benchmarks and experiments discussed in the paper are orchestrated through the `jacobi_experiments.py` script.

To view the available execution parameters:

```bash
python3 jacobi_experiments.py -h

```

**Note on parameters:** Some command-line arguments are automatically overridden during test execution to guarantee  compatibility with the specific configurations evaluated in the paper. Additionally, the `run_script.sh` utility is called internally by the Python script and may need to be adjusted depending on your specific environment setup. It is not intended to be executed manually.

## Contact

For questions regarding the code, hardware setup, or the paper, please feel free to open an issue in this repository or contact Giancarlo Bonvenuto directly via email (g248527@dac.unicamp.br).

## Citation

*To be updated*
