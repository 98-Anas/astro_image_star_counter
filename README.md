# Parallel Astronomical Star Counter
<img width="970" height="546" alt="JG8zWjbRfptn2HzRy2cJsV-970-80 jpg" src="https://github.com/user-attachments/assets/275c9fc5-c99c-46e4-a30b-d1fe054f138d" />


## Project Story
This project tells the story of how a large astronomical image can be processed in parallel using MPI. It begins with a proof-of-concept script and evolves through optimization and advanced scaling. The base dataset is a very large grayscale TIFF image, and the task is to detect and count bright star-like points using OpenCV.

The narrative is captured in [`parallel_star_counter_docs.md`](parallel_star_counter_docs.md), and the code evolves through three phases in the [`phases/`](phases/) folder:

- [Phase 1 — Foundation](phases/phase1_foundation.md)
- [Phase 2 — Optimization](phases/phase2_optimization.md)
- [Phase 3 — Advanced Pipeline](phases/phase3_advanced.md)

## Why this project exists
Astronomical images are huge: the target image is 12,788 rows by 40,000 columns, which is more than 511 million pixels. Sequential image processing is slow and memory-intensive. This project demonstrates how to divide that work across multiple MPI processes and then combine results efficiently.

The core problem is well suited to parallel computing because:
- each image region can be processed independently
- the workload is uniform across pixels
- the algorithm is deterministic and reproducible

## Project layout
### Documentation
- [`parallel_star_counter_docs.md`](parallel_star_counter_docs.md) — complete project explanation and code walkthrough
- [`prerequisites_installation.md`](prerequisites_installation.md) — Conda-based install guide, Raspberry Pi cluster comparison, and setup notes
- [`phases/phase1_foundation.md`](phases/phase1_foundation.md) — original architectural story and first implementation
- [`phases/phase2_optimization.md`](phases/phase2_optimization.md) — optimized Python 3 MPI implementation
- [`phases/phase3_advanced.md`](phases/phase3_advanced.md) — advanced dynamic pipeline design
 - [`cluster_preparation.md`](cluster_preparation.md) — detailed Raspberry Pi cluster preparation guide (RPi4 / RPi5)

### Code
- [`code_optimization/`](code_optimization/) — Python implementations of the MPI star counter
  - [`code_optimization/original_star_counter.py`](code_optimization/original_star_counter.py)
  - [`code_optimization/optimized_star_counter.py`](code_optimization/optimized_star_counter.py)
  - [`code_optimization/advanced_star_counter.py`](code_optimization/advanced_star_counter.py)

### Installation utilities
- [`installation_scripts/install_arm.sh`](installation_scripts/install_arm.sh) — ARM/Raspberry Pi setup script
- [`installation_scripts/install_amd.sh`](installation_scripts/install_amd.sh) — AMD/x86_64 setup script

### Cluster preparation
- [`cluster_preparation.md`](cluster_preparation.md) — step-by-step instructions for preparing a minimum 2-node RPi cluster, networking, NFS, Conda, OpenMPI, and orchestration choices (k3s/Slurm)

### Image resources
- [`astro_images/`](astro_images/) — sample images and astronomy illustrations used for testing and documentation

### Phase results (visuals)
Each phase includes a `*_results/` folder with example outputs and visualizations. Open the folders below to inspect result images and diagnostics.

- [`phases/phase1_results/`](phases/phase1_results/) — initial run outputs and screenshots
- [`phases/phase2_results/`](phases/phase2_results/) — optimized-run outputs (empty if not generated)
- [`phases/phase3_results/`](phases/phase3_results/) — advanced-run outputs (empty if not generated)

## How the project evolves
1. **Phase 1 — Foundation**: the original MPI implementation that reads the image on rank 0, distributes row chunks, thresholds locally, and reduces star counts.
2. **Phase 2 — Optimization**: fixes Python 3 compatibility, corrects buffer allocation, adds explicit MPI counts/displacements, and cleans up the algorithm.
3. **Phase 3 — Advanced Pipeline**: removes hard-coded dimensions, supports command-line image paths, adds halo overlap, uses memory-mapped files, and reports load balance.

## Running the project
### Recommended setup
Follow [`prerequisites_installation.md`](prerequisites_installation.md) to create a Conda environment and install MPI.

### Run the optimized version
```bash
mpirun -n 4 python code_optimization/optimized_star_counter.py
```

### Run the advanced version
```bash
mpirun -n 4 python code_optimization/advanced_star_counter.py path/to/heic1502a.tif --halo 30 --threshold-constant 0
```

### Run the original version
```bash
mpirun -n 4 python code_optimization/original_star_counter.py
```

## Platform-specific installers
Use the provided installer scripts depending on your hardware:
- [`installation_scripts/install_arm.sh`](installation_scripts/install_arm.sh) — Raspberry Pi and other ARM systems
- [`installation_scripts/install_amd.sh`](installation_scripts/install_amd.sh) — AMD/x86_64 systems

## How to explore the story
Read the main documentation first:
- [`parallel_star_counter_docs.md`](parallel_star_counter_docs.md)
Then follow the phase progression:
- [`phases/phase1_foundation.md`](phases/phase1_foundation.md)
- [`phases/phase2_optimization.md`](phases/phase2_optimization.md)
- [`phases/phase3_advanced.md`](phases/phase3_advanced.md)

## Key takeaways
- Parallel image processing can reduce runtime dramatically for large astronomical images.
- Correct MPI usage requires careful buffer allocation, explicit counts, and consistent data layout.
- The advanced pipeline is the most robust for variable image sizes and multi-node execution.

## Next steps
Future extensions could include:
- support for FITS or other astronomical data formats
- GPU acceleration for thresholding
- more sophisticated star segmentation beyond binary thresholding
- integration with job schedulers for larger multi-node clusters

## Social / Posts
I shared the project and experimentation notes for phase(1) on LinkedIn — you can view and share your feedback there:

- [LinkedIn post](https://www.linkedin.com/posts/anas-ayman98_highperformancecomputing-raspberrypi-supercomputers-share-7017960789101715456-6Eeq/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC_flRYBhpHmU-k6fTsQWlIKmsnG6GSidPM)

- [Project Source (All rights reserved for the phase1 code author)](https://github.com/swanandM/Galaxy-Star-Count-Mpi4py.git)
