import sys
import os
import math
import argparse
import numpy as np
import cv2
from mpi4py import MPI


# ---------------------------------------------------------------------------
# Distribution Utilities
# ---------------------------------------------------------------------------

def compute_distribution(total_rows, num_processes):
    """
    Compute per-process row counts and starting row indices.

    Handles the case where num_processes > total_rows by assigning zero rows
    to idle processes rather than crashing.

    Parameters
    ----------
    total_rows    : int  Total number of rows in the image.
    num_processes : int  Total number of MPI processes available.

    Returns
    -------
    row_counts    : list[int]  Number of rows assigned to each process.
    displacements : list[int]  Starting row index for each process.
    active_procs  : int        Number of processes that receive at least one row.
    """
    active_procs = min(total_rows, num_processes)
    base = total_rows // active_procs
    remainder = total_rows % active_procs

    row_counts = []
    for i in range(num_processes):
        if i < remainder:
            row_counts.append(base + 1)
        elif i < active_procs:
            row_counts.append(base)
        else:
            row_counts.append(0)  # Idle process

    displacements = [0]
    for i in range(1, num_processes):
        displacements.append(displacements[-1] + row_counts[i - 1])

    return row_counts, displacements, active_procs


def compute_overlap_bounds(rank, total_rows, row_counts, displacements, halo):
    """
    Compute the slice of rows each process should request including halo rows,
    and the interior slice to use for counting after thresholding.

    Parameters
    ----------
    rank         : int  This process's rank.
    total_rows   : int  Total image rows.
    row_counts   : list[int]
    displacements: list[int]
    halo         : int  Number of overlap rows on each side of the chunk.

    Returns
    -------
    fetch_start  : int  First row to fetch from the full image (clamped to 0).
    fetch_end    : int  One-past-last row to fetch (clamped to total_rows).
    core_start   : int  First row inside fetch_start that belongs to this process.
    core_end     : int  One-past-last core row.
    """
    core_start_abs = displacements[rank]
    core_end_abs = core_start_abs + row_counts[rank]

    fetch_start = max(0, core_start_abs - halo)
    fetch_end = min(total_rows, core_end_abs + halo)

    core_start = core_start_abs - fetch_start  # relative index inside fetched block
    core_end = core_end_abs - fetch_start

    return fetch_start, fetch_end, core_start, core_end


# ---------------------------------------------------------------------------
# Threshold Tuning
# ---------------------------------------------------------------------------

def compute_block_size(total_rows, total_cols):
    """
    Compute an adaptive threshold block size scaled to the image resolution.

    The block size is set to approximately 0.15% of the smaller image dimension,
    then rounded up to the nearest odd integer. This keeps the local neighborhood
    large enough to capture sky background gradients while small enough to isolate
    individual stellar point sources.

    Minimum block size is enforced at 11 (OpenCV minimum for adaptiveThreshold).
    """
    smaller_dim = min(total_rows, total_cols)
    raw = math.ceil(smaller_dim * 0.0015)
    if raw < 11:
        raw = 11
    if raw % 2 == 0:
        raw += 1
    return raw


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def read_image_shape(image_path):
    """
    Read only the image dimensions without loading pixel data into RAM.
    Uses OpenCV to decode the header, then discards the pixel buffer.
    Returns (total_rows, total_cols).
    """
    probe = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if probe is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    return probe.shape


def load_image_as_memmap(image_path, total_rows, total_cols, tmp_dir=None):
    """
    Load a grayscale image into a memory-mapped NumPy array on disk.

    This avoids holding the full image in heap RAM. The OS pages in only
    the data that is actually accessed during the scatter operation.

    Returns a read-only uint8 memmap array of shape (total_rows, total_cols).
    """
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    if tmp_dir is None:
        tmp_dir = os.path.abspath(os.path.dirname(__file__))

    mmap_path = os.path.join(tmp_dir, "_astro_mmap.dat")
    mmap_array = np.memmap(mmap_path, dtype=np.uint8, mode="w+",
                           shape=(total_rows, total_cols))
    mmap_array[:] = raw
    mmap_array.flush()
    return np.memmap(mmap_path, dtype=np.uint8, mode="r",
                     shape=(total_rows, total_cols))


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(
        description="Parallel astronomical star counter using MPI."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input astronomical image (TIFF, PNG, FITS-exported, etc.)"
    )
    parser.add_argument(
        "--halo", type=int, default=30,
        help="Number of overlap rows on each chunk boundary (default: 30)."
    )
    parser.add_argument(
        "--threshold-constant", type=int, default=0,
        help="Constant subtracted from local mean in adaptive threshold (default: 0)."
    )
    args = parser.parse_args()
    image_path = args.image_path
    halo = args.halo
    C_constant = args.threshold_constant

    shape_buf = np.zeros(2, dtype=np.int64)

    if rank == 0:
        t_io_start = MPI.Wtime()
        total_rows, total_cols = read_image_shape(image_path)
        t_io_end = MPI.Wtime()
        print(f"[Rank 0] Image dimensions: {total_rows} rows x {total_cols} cols")
        print(f"[Rank 0] Image header read time: {t_io_end - t_io_start:.4f} sec")
        shape_buf[0] = total_rows
        shape_buf[1] = total_cols

    comm.Bcast(shape_buf, root=0)
    total_rows = int(shape_buf[0])
    total_cols = int(shape_buf[1])

    row_counts, displacements, active_procs = compute_distribution(total_rows, size)

    if active_procs < size and rank == 0:
        print(
            f"[Warning] {size} processes launched but image has only {total_rows} rows. "
            f"{size - active_procs} process(es) will be idle."
        )

    my_row_count = row_counts[rank]
    is_idle = (my_row_count == 0)

    if not is_idle:
        fetch_start, fetch_end, core_start, core_end = compute_overlap_bounds(
            rank, total_rows, row_counts, displacements, halo
        )
        fetch_rows = fetch_end - fetch_start
    else:
        fetch_rows = 0
        core_start = 0
        core_end = 0

    block_size = compute_block_size(total_rows, total_cols)
    if rank == 0:
        print(f"[Rank 0] Computed adaptive threshold block size: {block_size}")

    w_start = MPI.Wtime()

    full_image = None
    if rank == 0:
        t_load_start = MPI.Wtime()
        full_image = load_image_as_memmap(image_path, total_rows, total_cols)
        t_load_end = MPI.Wtime()
        print(f"[Rank 0] Memory-mapped image load time: {t_load_end - t_load_start:.4f} sec")

        for dest in range(1, size):
            if row_counts[dest] == 0:
                continue
            d_fetch_start, d_fetch_end, _, _ = compute_overlap_bounds(
                dest, total_rows, row_counts, displacements, halo
            )
            chunk = np.ascontiguousarray(
                full_image[d_fetch_start:d_fetch_end, :], dtype=np.uint8
            )
            comm.Send(chunk, dest=dest, tag=dest)

        if not is_idle:
            local_chunk = np.ascontiguousarray(
                full_image[fetch_start:fetch_end, :], dtype=np.uint8
            )
        else:
            local_chunk = None

    else:
        if not is_idle:
            local_chunk = np.empty((fetch_rows, total_cols), dtype=np.uint8)
            comm.Recv(local_chunk, source=0, tag=rank)
        else:
            local_chunk = None

    t_compute_start = MPI.Wtime()

    local_star_count = np.int64(0)

    if not is_idle:
        local_thresh = cv2.adaptiveThreshold(
            local_chunk,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C_constant
        )

        core_thresh = local_thresh[core_start:core_end, :]
        local_star_count = np.int64(np.count_nonzero(core_thresh == 255))

    t_compute_end = MPI.Wtime()
    compute_time = t_compute_end - t_compute_start

    print(f"[Rank {rank}] Local star count: {local_star_count} "
          f"| Compute time: {compute_time:.4f} sec")

    total_star_count = comm.allreduce(local_star_count, op=MPI.SUM)

    all_compute_times = comm.gather(compute_time, root=0)

    if rank == 0:
        w_end = MPI.Wtime()
        total_wall_time = w_end - w_start

        min_t = min(all_compute_times)
        max_t = max(all_compute_times)
        avg_t = sum(all_compute_times) / len(all_compute_times)
        imbalance_pct = ((max_t - min_t) / max_t * 100) if max_t > 0 else 0.0

        print()
        print("=" * 60)
        print(f"  Image           : {image_path}")
        print(f"  Dimensions      : {total_rows} x {total_cols}")
        print(f"  MPI processes   : {size} ({active_procs} active)")
        print(f"  Block size used : {block_size}")
        print(f"  Halo rows       : {halo}")
        print(f"  Total stars     : {total_star_count}")
        print(f"  Wall time       : {total_wall_time:.4f} sec")
        print(f"  Compute times   : min={min_t:.4f}s  max={max_t:.4f}s  avg={avg_t:.4f}s")
        print(f"  Load imbalance  : {imbalance_pct:.1f}%")
        print("=" * 60)


if __name__ == "__main__":
    main()
