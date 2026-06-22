# Prerequisites and Installation Guide (Conda-based)

## 1. Overview
This project uses Python, OpenCV, NumPy, and MPI to perform parallel star counting on large astronomical images. The recommended setup is a Conda environment to isolate dependencies and ensure reproducible execution across systems.

This guide includes:
- Conda environment creation and dependency installation
- MPI runtime installation
- Verification commands
- Run examples
- Comparison between 2 Raspberry Pi 4 nodes and 2 Raspberry Pi 5 nodes as a small cluster

---

## 2. System Requirements
### Recommended minimum hardware
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 8 GB minimum, 16 GB recommended for large TIFF images
- Disk: 10 GB free
- Network: Ethernet or reliable Wi-Fi for MPI across nodes

### Recommended OS
- Linux (Ubuntu 20.04+ recommended)
- macOS is supported for development, but cluster testing is easier on Linux
- Windows is supported through WSL2, but the Raspberry Pi comparison is Linux-based

---

## 3. Install Conda
If you do not already have Conda installed, install Miniconda or Anaconda.

### Install Miniconda
1. Download Miniconda for your platform from https://docs.conda.io/en/latest/miniconda.html
2. Install it following the platform instructions

### Verify Conda installation
```bash
conda --version
```

---

## 4. Create the Conda Environment
From the project root directory (`d:\astro_image_star_counter`), create and activate a Conda environment.

```bash
conda create -n astro-mpi python=3.11 -y
conda activate astro-mpi
```

> Use Python 3.11 or 3.10 for best compatibility with `mpi4py` and `opencv-python`.

---

## 5. Install Python Dependencies
Install the required Python packages inside the Conda environment.

```bash
conda install numpy -y
pip install opencv-python mpi4py
```

### Why use pip for OpenCV and mpi4py?
- `opencv-python` is usually easier to install via `pip` because it provides prebuilt wheels.
- `mpi4py` requires an MPI implementation at runtime, but the pip package installs the Python wrapper.

---

## 6. Install MPI Implementation
MPI is required to launch the program across multiple processes or hosts.

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y libopenmpi-dev openmpi-bin
```

### macOS (Homebrew)
```bash
brew install open-mpi
```

### Raspberry Pi OS / Ubuntu on Pi
```bash
sudo apt update
sudo apt install -y libopenmpi-dev openmpi-bin
```

---

## 7. Verify the Environment
### Check Python packages
```bash
python -c "import numpy, cv2, mpi4py; print('ok', numpy.__version__, cv2.__version__, mpi4py.__version__)"
```

### Check MPI
```bash
mpirun --version
```

If both commands print successfully, the environment is ready.

---

## 8. Run the Project
From the Conda environment, run one of the Python scripts.

### Example using the optimized script
```bash
mpirun -n 4 python code_optimization/optimized_star_counter.py
```

### Example using the advanced script with an image path
```bash
mpirun -n 4 python code_optimization/advanced_star_counter.py path/to/heic1502a.tif --halo 30 --threshold-constant 0
```

### Running across multiple nodes
If you have two machines connected by network, create a host file like `hosts.txt`:
```text
pi4-node1 slots=4
pi4-node2 slots=4
```
Then run:
```bash
mpirun --hostfile hosts.txt -n 8 python code_optimization/advanced_star_counter.py path/to/heic1502a.tif
```

---

## 9. Conda Activation Reminder
Every new terminal session must activate the environment first:
```bash
conda activate astro-mpi
```

To remove the environment later:
```bash
conda deactivate
conda remove -n astro-mpi --all -y
```

---

## 10. Raspberry Pi Cluster Comparison: 2x Pi 4 vs 2x Pi 5
This section compares running the project on a small cluster of two Raspberry Pi 4 boards versus two Raspberry Pi 5 boards.

### A. CPU and memory characteristics
| Feature | Raspberry Pi 4 (per board) | Raspberry Pi 5 (per board) |
|---|---|---|
| CPU | Broadcom BCM2711 quad-core Cortex-A72 @ 1.5 GHz | Broadcom BCM2712 quad-core Cortex-A76 @ 2.4 GHz |
| RAM | 4 GB or 8 GB LPDDR4 | 4 GB or 8 GB LPDDR5 |
| Memory bandwidth | ~34 GB/s | ~68 GB/s |
| GPU | VideoCore VI | VideoCore VII |
| Storage | microSD or USB attached storage | microSD, USB4, NVMe (via adapter) |

### B. Performance expectations
- Pi 5 is significantly faster per core: the Cortex-A76 architecture is newer and provides higher IPC and clock.
- Memory bandwidth on Pi 5 is roughly double, which matters for image processing because large TIFF buffers and thresholding are memory-bound operations.
- If the program is I/O-heavy and reads large images from storage, Pi 5 will also benefit from faster NVMe/USB4 storage options.

### C. Parallel workload behavior
For MPI image processing, the workload is mostly:
1. reading image data
2. distributing data across processes
3. performing per-chunk thresholding
4. collecting reduction results

**Pi 4 cluster**
- Good for small experiments and low-cost clusters
- Expect limited performance from CPU-bound thresholding and memory-bound image operations
- With 2 boards and 4 processes each, effective compute capacity is similar to an 8-core desktop at lower per-core speed
- Suitable for smaller images and development

**Pi 5 cluster**
- Better for real work: higher clock, newer architecture, and faster memory
- More efficient per-process throughput for OpenCV operations
- Less likely to become bottlenecked by memory read/write
- Better for larger images and sustained MPI workloads

### D. Network and cluster considerations
- Both Pi 4 and Pi 5 should use gigabit Ethernet for MPI communication.
- Use a dedicated switch or direct Ethernet link for best stability.
- For cluster execution, make sure all nodes have consistent Conda environments and MPI versions.
- Pi 5 supports faster networking adapters via USB4, but standard built-in Gigabit Ethernet is sufficient for this workload.

### E. Practical recommendation
- For learning and prototyping: `2x Raspberry Pi 4` is a good low-cost setup.
- For better performance and future-proofing: `2x Raspberry Pi 5` is the superior choice.
- If you need to run larger TIFF images or more concurrent MPI processes, choose Pi 5.

### F. Simple professional explanation
A cluster with two Raspberry Pi 5 boards will generally be faster and more efficient than two Raspberry Pi 4 boards because the Pi 5 has a newer CPU design and a much faster memory subsystem. When processing large astronomical images, the program spends a lot of time reading pixel buffers and applying threshold filters, so the extra CPU speed and memory bandwidth of the Pi 5 translate directly into shorter runtime. The Pi 4 cluster is still usable, but it is best viewed as a budget entry-level option rather than the high-performance choice.

---

## 11. Notes on Raspberry Pi usage
- Use `conda` on Raspberry Pi OS only if your board is 64-bit and `conda` supports the architecture. If standard Conda is not available, use `mambaforge` or `miniforge` for ARM64.
- On Pi 4 and Pi 5, use `pip install opencv-python` only if compatible wheels are available. Otherwise, consider `pip install opencv-python-headless` or build OpenCV from source.
- Ensure the `mpi4py` package is built against the same OpenMPI library installed on each node.

---

## 12. Summary
- Create a Conda environment: `conda create -n astro-mpi python=3.11 -y`
- Install dependencies: `conda install numpy -y && pip install opencv-python mpi4py`
- Install MPI runtime on Linux: `sudo apt install libopenmpi-dev openmpi-bin`
- Run with MPI: `mpirun -n 4 python code_optimization/advanced_star_counter.py path/to/image.tif`
- Raspberry Pi 5 offers better performance than Raspberry Pi 4 for this project, especially for large images and memory-heavy processing.
