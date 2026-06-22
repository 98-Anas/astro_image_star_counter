# Phase 3 — Advanced Pipeline

## Phase goal
This phase upgrades the MPI star counter into a fully dynamic pipeline that can handle variable image dimensions, command-line configuration, and improved chunk accuracy.

## Advanced features introduced
- Command-line image path and runtime options
- Automatic threshold block size tuning based on image size
- Halo overlap for chunk boundaries to avoid edge artifacts
- Memory-mapped image loading to reduce RAM spikes
- Point-to-point chunk distribution for irregular halo-extended blocks
- Per-process compute time reporting
- Load balance summary at the end of execution

## Implemented in Phase 3
- `code_optimization/advanced_star_counter.py`

This script is the most production-ready version in the project and is designed for larger images and multi-node MPI execution.

## Key technical additions
- `argparse` to replace hard-coded values with runtime options
- `compute_distribution(...)` supports more processes than image rows and marks idle ranks
- `compute_overlap_bounds(...)` adds halo rows while keeping core counts non-overlapping
- `compute_block_size(...)` chooses an adaptive threshold window based on image resolution
- `load_image_as_memmap(...)` uses a temporary disk-backed memmap to limit heap memory
- `comm.Send` / `comm.Recv` for halo-extended chunks that do not fit cleanly into `Scatterv`
- `comm.allreduce` and `comm.gather` for global summary reporting

## Why Phase 3 matters
This phase shifts the project from a fixed tutorial example to a flexible computational pipeline. It addresses real-world needs:
- different input image sizes
- cluster-friendly execution
- accurate boundary handling
- performance diagnostic output

## Usage example
```bash
mpirun -n 4 python code_optimization/advanced_star_counter.py path/to/heic1502a.tif --halo 30 --threshold-constant 0
```

## Project progression
Phase 3 is the final evolution in this project. It proves that the same star-counting problem can be solved with increasing robustness:
- Phase 1: concept and initial MPI design
- Phase 2: correctness and optimization
- Phase 3: flexibility, accuracy, and runtime diagnostics
