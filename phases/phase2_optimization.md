# Phase 2 — Optimization

## Phase goal
This phase refines the initial MPI pipeline by fixing correctness issues and applying Python 3 best practices. The result is a reliable parallel star counter that can be executed on a single host or a simple MPI cluster.

## What is improved in Phase 2
- Correct row distribution using integer division with `//`
- Valid 2D local receive buffer allocation with `np.zeros`
- Explicit `Scatterv` send counts and displacements
- Removal of unnecessary broadcasts
- Validation of image load success
- Clear binary threshold star counting semantics
- Function-based code structure with reusable helpers

## Implemented in Phase 2
- `code_optimization/optimized_star_counter.py`

This file contains a deployable MPI script that is stable and suitable for educational and production use.

## Key technical changes
- `compute_row_distribution(total_rows, num_processes)` splits rows evenly and handles remainders.
- `np.ascontiguousarray` ensures MPI receives C-order buffers.
- `comm.Scatterv([a, send_counts, send_displacements, MPI.UNSIGNED_CHAR], [local_chunk, MPI.UNSIGNED_CHAR], root=0)` is used correctly.
- `np.count_nonzero(local_thresh == 255)` counts stars precisely after thresholding.

## Why Phase 2 matters
The improvements in this phase make the program:
- portable across Python 3 environments
- more predictable in MPI behavior
- easier to maintain and modify

## Bridge to Phase 3
Phase 3 builds on this stable base and introduces dynamic input handling, memory mapping, overlap-aware chunk boundaries, and advanced load reporting.
