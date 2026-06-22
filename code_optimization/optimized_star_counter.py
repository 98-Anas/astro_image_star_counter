# Optimized MPI star counter code from documentation
# This version uses Python 3, correct Scatterv usage, and robust validation.

import numpy as np
import cv2
from mpi4py import MPI


def compute_row_distribution(total_rows, num_processes):
    """
    Compute how many rows each process receives and where
    in the source array each process's chunk starts.

    Returns:
        row_counts   : list of row counts per process
        displacements: list of starting row indices per process
    """
    base = total_rows // num_processes
    remainder = total_rows % num_processes

    row_counts = []
    for i in range(num_processes):
        row_counts.append(base + 1 if i < remainder else base)

    displacements = [0]
    for i in range(1, num_processes):
        displacements.append(displacements[-1] + row_counts[i - 1])

    return row_counts, displacements


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    TOTAL_ROWS = 12788
    TOTAL_COLS = 40000
    IMAGE_PATH = "heic1502a.tif"
    THRESHOLD_BLOCK_SIZE = 59  # Must be odd; controls local neighborhood size
    THRESHOLD_CONSTANT = 0     # Subtracted from local mean; 0 means no offset

    # Step 1: Root process reads the image and benchmarks I/O time
    a = None
    if rank == 0:
        t1 = MPI.Wtime()
        img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        t2 = MPI.Wtime()

        if img is None:
            raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}")

        print(f"Time taken to read image: {t2 - t1:.4f} sec")
        a = np.ascontiguousarray(img, dtype=np.uint8)  # Ensure C-contiguous layout

    # Step 2: Compute the row distribution for all processes
    row_counts, displacements = compute_row_distribution(TOTAL_ROWS, size)

    # Step 3: Each process determines its own local row count
    local_row_count = row_counts[rank]

    # Step 4: Allocate the local receive buffer
    local_chunk = np.zeros((local_row_count, TOTAL_COLS), dtype=np.uint8)

    # Step 5: Build Scatterv arguments (element counts and byte offsets)
    send_counts = [r * TOTAL_COLS for r in row_counts]
    send_displacements = [d * TOTAL_COLS for d in displacements]

    w1 = MPI.Wtime()

    # Step 6: Distribute image chunks from root to all processes
    comm.Scatterv(
        [a, send_counts, send_displacements, MPI.UNSIGNED_CHAR],
        [local_chunk, MPI.UNSIGNED_CHAR],
        root=0
    )

    # Step 7: Each process applies adaptive thresholding to its local chunk
    local_thresh = cv2.adaptiveThreshold(
        local_chunk,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=THRESHOLD_BLOCK_SIZE,
        C=THRESHOLD_CONSTANT
    )

    # Step 8: Count white pixels (== 255) as detected stars
    local_star_count = np.count_nonzero(local_thresh == 255)
    print(f"Rank {rank}: local star count = {local_star_count}")

    # Step 9: Reduce all local counts to the root process via summation
    total_star_count = np.array(0, dtype=np.int64)
    local_count_array = np.array(local_star_count, dtype=np.int64)
    comm.Reduce(local_count_array, total_star_count, op=MPI.SUM, root=0)

    # Step 10: Root process reports the final result and total elapsed time
    if rank == 0:
        w2 = MPI.Wtime()
        print(f"Total detected stars: {total_star_count}")
        print(f"Total parallel processing time: {w2 - w1:.4f} sec")


if __name__ == "__main__":
    main()
