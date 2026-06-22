# Cluster Preparation Guide — Raspberry Pi (RPi4 / RPi5)

This document explains how to prepare a minimum 2-node Raspberry Pi cluster for MPI-based workloads (star counting pipeline), covering hardware selection, OS and software installation, networking, storage, and testing. It includes practical commands, configuration examples, and additional sources (including the requested playlist).

Playlist source:
- Primary: https://youtube.com/playlist?list=PLw5C-otg1WCfIbjfKddgQou54Wm8ILfMw&si=Q6H9R583fPvw1MQb

Other recommended sources:
- Raspberry Pi Documentation: https://www.raspberrypi.org/documentation/
- Raspberry Pi OS (64-bit) releases and images: https://www.raspberrypi.com/software/operating-systems/
- OpenMPI: https://www.open-mpi.org/
- MPI for Python (`mpi4py`): https://mpi4py.readthedocs.io/
- Miniforge / Mambaforge: https://github.com/conda-forge/miniforge
- k3s (lightweight Kubernetes): https://k3s.io/
- Slurm Workload Manager: https://slurm.schedmd.com/
- Apptainer (Singularity successor): https://apptainer.org/

---

## Summary recommendations (short)
- Use Raspberry Pi OS 64-bit (or Ubuntu Server 64-bit) for both RPi4 and RPi5.
- Prefer Raspberry Pi 5 if budget allows: higher clock, Cortex-A76, double memory bandwidth, NVMe/USB4 options.
- Use wired Gigabit Ethernet and a small switch for networked MPI runs.
- Use Miniforge/Mambaforge for Conda environments on ARM.

---

## 1. Hardware selection and considerations

RPi4 vs RPi5 differences (practical impact):
- CPU: RPi4 has Cortex-A72 @ 1.5GHz; RPi5 has Cortex-A76 @ 2.4GHz. RPi5 offers substantially higher IPC and clock for compute-bound tasks.
- Memory: both available in 4/8 GB variants; RPi5 uses LPDDR5 with much higher bandwidth (important for memory-bound image ops).
- Storage: RPi5 supports faster USB/PCIe NVMe adapters; RPi4 relies more on microSD or USB storage.
- Thermal & power: RPi5 draws more power under load and benefits from active cooling (fan + heatsink); ensure adequate PSU (5V/8A or recommended per board with headroom for multiple boards).
- Networking: built-in Gigabit Ethernet on both; use wired connection for low-latency MPI.

If hardware can change, design the cluster steps to be architecture-agnostic (detect `uname -m`) and install appropriate builds (ARM64 vs x86_64).

Minimum parts per node:
- Raspberry Pi 4 or 5 board (64-bit recommended)
- microSD (Class 10) or NVMe/USB storage (fast recommended)
- USB-C power supply (match board requirements)
- Heatsink + fan
- Ethernet cable

Optional for improved performance:
- USB3 NVMe adapter (or native NVMe on RPi5 via adapter)
- Gigabit switch with PoE if you want central power

---

## 2. OS and base image

Recommended OS images (64-bit):
- Raspberry Pi OS (64-bit) or Ubuntu Server (64-bit for ARM)

Flash image to microSD or NVMe and enable SSH headless:

```bash
# Example (on your workstation)
# Replace IMAGE.img and /dev/sdX accordingly
sudo dd if=2026-xx-raspios-arm64.img of=/dev/sdX bs=4M status=progress conv=fsync
sync
# Optionally create an empty file named 'ssh' in the boot partition to enable SSH
# and wpa_supplicant.conf for Wi-Fi if needed
```

Boot each Pi, log in, and perform base updates:

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

Ensure you are running a 64-bit kernel/userland (check `uname -m` and `getconf LONG_BIT`).

---

## 3. Networking and hostnames

- Use static IPs or DHCP reservations so the head node can contact workers predictably.
- Set meaningful hostnames (`pi-master`, `pi-node1`, `pi-node2`, ...).

Example hostname change and hosts file update (on each node):

```bash
sudo hostnamectl set-hostname pi-node2
# Edit /etc/hosts or configure DHCP reservation
sudo vim /etc/hosts
# add lines like
# 192.168.1.10 pi-master
# 192.168.1.11 pi-node1
# 192.168.1.12 pi-node2
```

Time synchronization (important for logs): install `chrony` or `ntp`:

```bash
sudo apt install -y chrony
sudo systemctl enable --now chrony
```

---

## 4. SSH key setup and passwordless SSH for MPI

MPI often launches processes via SSH; passwordless SSH from the head node to workers simplifies runs.

On `pi-master` (head node):

```bash
# generate keys if needed
ssh-keygen -t ed25519 -C "pi-cluster" -f ~/.ssh/id_ed25519 -N ""
# copy key to each worker
ssh-copy-id -i ~/.ssh/id_ed25519.pub pi@pi-node1
ssh-copy-id -i ~/.ssh/id_ed25519.pub pi@pi-node2
```

Test a passwordless SSH session:

```bash
ssh pi@pi-node1 echo ok
```

If you use firewall/ufw, open SSH ports between nodes (or disable ufw on trusted private network).

---

## 5. Shared storage (optional but useful)

For convenience you can export an NFS share from the master and mount it on workers so that your dataset and code are accessible via a common path.

On `pi-master`:

```bash
sudo apt install -y nfs-kernel-server
sudo mkdir -p /srv/astro_shared
sudo chown pi:pi /srv/astro_shared
# Export (add to /etc/exports):
# /srv/astro_shared 192.168.1.0/24(rw,sync,no_subtree_check)
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
```

On worker nodes:

```bash
sudo apt install -y nfs-common
sudo mkdir -p /mnt/astro_shared
sudo mount pi-master:/srv/astro_shared /mnt/astro_shared
# add to /etc/fstab for persistence
```

Alternatively use rsync for synchronizing code/data between nodes.

---

## 6. Install Conda (Miniforge / Mambaforge) and Python environment

Use Miniforge/Mambaforge on ARM nodes for consistent package availability.

On each node (or on master and then share environment with NFS if desired):

```bash
# Example - install Miniforge
cd /tmp
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda create -n astro-mpi python=3.11 -y
conda activate astro-mpi
conda install -y numpy
pip install opencv-python mpi4py
```

Notes:
- If `opencv-python` binary wheels are not available for your ARM platform, install `opencv-python-headless` or build OpenCV from source (longer).
- `mpi4py` requires an MPI library at runtime (OpenMPI installed in next step).

---

## 7. Install OpenMPI and system packages

On every node:

```bash
sudo apt install -y libopenmpi-dev openmpi-bin build-essential
# verify
mpirun --version
```

You can build OpenMPI from source for optimized performance, but the distro package is usually fine for small clusters.

---

## 8. Create an MPI hostfile and basic run test

On `pi-master` create a `hosts` file listing the nodes and slot counts (cores):

`~/hosts`:
```
pi-master slots=4
pi-node1  slots=4
pi-node2  slots=4
```

Simple test using `mpirun` and `hostname` to ensure processes are launched across nodes:

```bash
mpirun --hostfile ~/hosts -np 6 hostname
```

You should see hostnames from `pi-master`, `pi-node1`, and `pi-node2` according to slot assignment.

Run a small Python MPI test (ensure `mpi4py` installed in the active Conda env):

`hello_mpi.py`:

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Hello from rank {rank}/{size} on host")
```

Run it:

```bash
mpirun --hostfile ~/hosts -np 6 python hello_mpi.py
```

---

## 9. Prepare and run the star-counter scripts

Place your scripts (from this repo `code_optimization/`) on the shared path or replicate them on each node. Example run (advanced script, with image path accessible on all nodes via NFS or local copy):

```bash
mpirun --hostfile ~/hosts -np 4 python /mnt/astro_shared/code_optimization/advanced_star_counter.py /mnt/astro_shared/heic1502a.tif --halo 30
```

Notes:
- For best performance ensure the image is locally accessible or use fast NFS/SSD; network I/O can be the bottleneck.
- Adjust `-np` and `slots` to match physical cores available.

---

## 10. Monitoring and diagnostics

- Use `htop` on nodes for CPU/memory monitoring.
- Use `iperf3` for network throughput testing between nodes:

```bash
# on server
iperf3 -s
# on client
iperf3 -c pi-master
```

- Use `dstat` or `vmstat` to inspect I/O and CPU behavior during runs.

---

## 11. Containerization and portability (Docker / Singularity / Apptainer)

For reproducible environments consider containers. On multi-node HPC, `Singularity` (now Apptainer) is preferred because it runs containers unprivileged and integrates well with MPI.

- Docker: great for development but requires `docker` daemon and root privileges; not ideal on shared HPC nodes.
- Apptainer/Singularity: designed for HPC, supports binding host MPI libraries into containers and running MPI jobs securely.

Quick notes for Apptainer on ARM:

```bash
# install prerequisites and build apptainer (or use distro packages if available)
# then build an image from a Dockerfile or Singularity recipe
apptainer build astro.sif docker://python:3.11
apptainer exec astro.sif python -c 'import mpi4py; print(mpi4py.__version__)'
```

When running MPI inside containers, ensure the container uses the system MPI libraries or a compatible ABI to allow host `mpirun` to launch processes.

---

## 12. DevOps vs HPC: Kubernetes (k3s) & Docker vs Slurm & Singularity

This project can be executed using container orchestration (DevOps style) or traditional HPC schedulers. Below is a comparison and when to use each.

### Kubernetes (k3s / k8s) — DevOps style
- Purpose: container orchestration, long-running services, microservices, batch jobs with additional tooling (Kubernetes Jobs/CronJobs).
- Pros:
  - Strong container lifecycle management and scheduling
  - Works well with Docker/OCI images
  - Easy to run peripheral services (monitoring, dashboards, storage)
- Cons:
  - Not optimized for tightly-coupled MPI workloads by default (networking and pod placement must be tuned)
  - Requires learning K8s concepts and potentially more overhead on tiny clusters
- How to run MPI on K8s:
  - Use MPI Operator (e.g., Kubeflow MPI Operator) or launch MPI in hostNetwork mode
  - Ensure low-latency CNI and correct placement of pods across nodes
- Lightweight option for RPi: `k3s` (lightweight Kubernetes) — but expect manual tuning to support MPI.

### Slurm — HPC style
- Purpose: batch scheduler designed for parallel HPC jobs (MPI, OpenMP, job arrays)
- Pros:
  - Native MPI integration, efficient job start-up, accounting, and queuing
  - Lightweight for HPC clusters and well-suited for MPI
- Cons:
  - Less focused on containerized microservices (though Slurm can launch containers)
  - Requires Munge authentication and Slurm daemons across nodes
- When to choose Slurm:
  - If you primarily run MPI and batch HPC jobs, Slurm is the better choice.

### Docker vs Singularity/Apptainer — container runtimes
- Docker: broad ecosystem, but needs daemon/root; less common on HPC nodes.
- Singularity/Apptainer: designed for HPC, runs unprivileged, integrates with job schedulers and MPI.

Recommendation matrix:
- Small teaching / development cluster: `k3s` + Docker may be acceptable and easier to experiment with containers.
- Production MPI workloads (tight coupling, large datasets): `Slurm` + `Apptainer` (or direct system environment) is recommended.

---

## 13. Slurm quick setup notes (small cluster)

On each node:

```bash
sudo apt install -y munge libmunge-dev slurm-wlm slurm-wlm-basic-plugins
# configure /etc/munge/munge.key (copy from master) and /etc/slurm-llnl/slurm.conf
# basic slurm.conf example (adapt ControlMachine and NodeName/Partition)
```

Slurm setup is detailed; refer to Slurm documentation for full configuration.

---

## 14. Kubernetes (k3s) quick notes for RPi

k3s supports ARM; install on each node (master and agents):

```bash
# install k3s on master
curl -sfL https://get.k3s.io | sh -
# install agent on worker using token from master
curl -sfL https://get.k3s.io | K3S_URL=https://<MASTER_IP>:6443 K3S_TOKEN=<TOKEN> sh -
```

MPI on Kubernetes requires MPI Operator or hostNetworking and proper CNI setup. Use caution — Kubernetes scheduling decisions can cause MPI processes to be placed inefficiently without topology awareness.

---

## 15. Troubleshooting and common pitfalls
- Missing `mpirun` errors: ensure OpenMPI is installed on all nodes and PATH is correct.
- Permission issues with NFS: check UID/GID mappings and `no_root_squash` if necessary (careful with security).
- `mpi4py` import errors: ensure `mpi4py` built/installed against the same OpenMPI library present at runtime.
- Performance: verify networking (iperf3) and disk I/O; often the bottleneck is storage or network, not CPU.

---

## 16. Example checklist before first run
- [ ] All nodes booting and reachable via SSH
- [ ] Passwordless SSH from `pi-master` to workers
- [ ] `mpirun --version` works on all nodes
- [ ] `python -c "import mpi4py"` works in the target Conda env on all nodes
- [ ] Hosts file prepared (`~/hosts`) with correct slot counts
- [ ] Dataset accessible on all nodes (NFS or local copies)

---

## 17. Useful commands summary

```bash
# Update and prepare
sudo apt update && sudo apt upgrade -y
sudo apt install -y libopenmpi-dev openmpi-bin build-essential nfs-kernel-server nfs-common chrony iperf3

# Install Miniforge (ARM example)
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda create -n astro-mpi python=3.11 -y
conda activate astro-mpi
conda install -y numpy
pip install opencv-python mpi4py

# Test MPI
mpirun --hostfile ~/hosts -np 6 python hello_mpi.py

# Network test
iperf3 -s  # on server
iperf3 -c pi-master  # on client
```

---

## 18. References and further reading
- Playlist (requested): https://youtube.com/playlist?list=PLw5C-otg1WCfIbjfKddgQou54Wm8ILfMw&si=Q6H9R583fPvw1MQb
- Raspberry Pi Documentation: https://www.raspberrypi.org/documentation/
- OpenMPI: https://www.open-mpi.org/
- mpi4py docs: https://mpi4py.readthedocs.io/
- Miniforge: https://github.com/conda-forge/miniforge
- k3s: https://k3s.io/
- Slurm: https://slurm.schedmd.com/
- Apptainer (Singularity): https://apptainer.org/

---

