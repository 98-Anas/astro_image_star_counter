# Phase 1 — Foundation

## Project intent
This phase introduces the core problem: counting stars in a large astronomical image using parallel processing. The goal is to make the original MPI-based algorithm clear and identify its early implementation issues.

## Why this problem
Astronomical images are extremely large and contain many point sources. A single Hubble-style TIFF image can contain hundreds of millions of pixels. Sequential image processing on such data is slow, so distributing the work across multiple processes is a practical first step.

## What is implemented in Phase 1
- Load a grayscale astronomical image on the root MPI process
- Divide the image into horizontal row chunks across MPI ranks
- Scatter row chunks to worker processes
- Apply adaptive thresholding to each chunk with OpenCV
- Count bright pixels as stars locally
- Reduce local counts back to the root process

## Core concepts
- `mpi4py` and `MPI.COMM_WORLD`
- `rank` and `size` for process identity and team size
- `cv2.imread(..., cv2.IMREAD_GRAYSCALE)` for grayscale read
- `comm.Scatterv` for variable-sized row distribution
- `cv2.adaptiveThreshold` for local star detection
- `comm.Reduce(..., op=MPI.SUM)` for result aggregation

## Original implementation
The original source preserves the project’s first working approach. It is captured in:

- `code_optimization/original_star_counter.py`

This version demonstrates the initial MPI design and the shape of the parallel dataflow.

## Known limitations
Phase 1 intentionally leaves several issues unresolved so the project can evolve clearly:
- Python 2-era syntax and division semantics
- Incorrect array initialization for the scatter buffer
- Redundant broadcasts of shape and empty buffers
- Missing `Scatterv` counts and displacements
- Absence of image load validation
- Ambiguous star-count threshold logic

## Next step
Phase 2 improves the implementation by fixing these problems, cleaning the code, and making the MPI usage robust and Python 3 compatible.
