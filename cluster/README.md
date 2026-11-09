# Benchmark on the DFKI cluster

Runs the benchmark from #83 as a Slurm job array on the cluster: one array task per
model × dataset × boxes combination, each on one GPU. The results land in the same
`benchmark/<run>/` layout as on the workstation, so one table can be built from both.

Needs: an account on `lnode01` and about 60 GB of free space in your scratch folder (`~/SCRATCH`, a link in your home folder
on the cluster).

## 1. Image

The Singularity image is converted from the public multi-arch image on GHCR
(`ghcr.io/adriandavidauer/asd-for-hri:main`), which already contains the AV models and their
weights. Converting needs no root and no definition file, so it can be done on `lnode01` directly:

```bash
mkdir -p ~/SCRATCH/images && cd ~/SCRATCH/images
singularity pull --force asd4hri.sif docker://ghcr.io/adriandavidauer/asd-for-hri:main
```

Pull again after a new image is published on `main`. For a fixed version use the
`sha-<commit>` tag instead of `main`.

## 2. Data

One folder holds the datasets and receives the results. It is mounted as `/app/data` in the
container, so it needs this layout (the same as the workstation's data root):

```text
~/SCRATCH/asd4hri/
├── ava/        AVA val videos + annotations/
├── videos/     UniTalk videos (val/)
├── csv/        UniTalk annotation CSVs
├── wasd/       WASD clips
└── easycom/    EasyCom videos + csv/
```

Copy it from the PC that holds the data with `cluster/sync_data.sh`. It copies only the files the
benchmark reads, about 58 GB: no WASD source videos, no extracted UniTalk frames, no training CSVs.
It can be stopped and started again.

```bash
CLUSTER_USER=<user> cluster/sync_data.sh unitalk        # 0.7 GB, enough for the first test job
CLUSTER_USER=<user> cluster/sync_data.sh                 # all: unitalk wasd easycom ava
```

`SRC` (default `/Data/data`) and `DEST` (default `SCRATCH/asd4hri`, relative to your home folder on the cluster) can be changed the same
way.

The cluster storage is not backed up and is shared: delete the datasets when the benchmark is done.

## 3. Submit

On `lnode01`, in a checkout of this repository:

```bash
bash cluster/benchmark.slr --list            # the index of every run
mkdir -p logs                                # Slurm writes logs/slurm_<job>_<task>.log
sbatch --array=0 cluster/benchmark.slr       # one test task first
sbatch --array=1-39%8 cluster/benchmark.slr  # then the rest, at most 8 at once
```

Partition, account and time limit are set in the `#SBATCH` lines of `cluster/benchmark.slr`.
Paths and the run tag can be changed when submitting, for example
`IMAGE=... DATA=... TAG=volta sbatch --array=0-39 cluster/benchmark.slr`. A finished run is
skipped, so failed tasks can be resubmitted with the same command.

Watch the jobs with `squeue -u $USER`; cancel them with `scancel <jobid>`.

## 4. Collect

Every task also rewrites `benchmark/benchmark.csv`, and parallel tasks can overwrite each other's
version. So when all tasks are done, build the table once more:

```bash
singularity exec --containall --bind $(readlink -f ~/SCRATCH/asd4hri):/app/data \
    $(readlink -f ~/SCRATCH/images/asd4hri.sif) \
    bash -c "cd /app && python -m src.experiments.run_benchmark --data_dir /app/data --collect_only"
```

To merge the cluster runs with the workstation runs, copy `benchmark/<run>/stats` from one side to
the other and run the same `--collect_only` command there. The tag (`cluster`, `gpu`, `pi`) in the
run name keeps the runs apart. The accuracy of a run does not depend on the machine, but FPS and
the time per stage do.
