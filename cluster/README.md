# Benchmark on the DFKI cluster

Runs the benchmark from #83 as a Slurm job array on the cluster: one array task per
model × dataset × boxes combination, each on one GPU. The results land in the same
`benchmark/<run>/` layout as on the workstation, so one table can be built from both.

Needs: an account on `lnode01` and about 110 GB of free space under `/mnt/beegfs/data`.

## 1. Image

The Singularity image is converted from the public multi-arch image on GHCR
(`ghcr.io/adriandavidauer/asd-for-hri:main`), which already contains the AV models and their
weights. Converting needs no root and no definition file, so it can be done on `lnode01` directly:

```bash
mkdir -p /mnt/beegfs/public/images/asd4hri && cd /mnt/beegfs/public/images/asd4hri
singularity pull asd4hri.sif docker://ghcr.io/adriandavidauer/asd-for-hri:main
```

Pull again after a new image is published on `main`. For a fixed version use the
`sha-<commit>` tag instead of `main`.

## 2. Data

One folder holds the datasets and receives the results. It is mounted as `/app/data` in the
container, so it needs this layout (the same as the workstation's data root):

```text
/mnt/beegfs/data/asd4hri/
├── ava/        AVA val videos + annotations/
├── videos/     UniTalk videos (val/)
├── csv/        UniTalk annotation CSVs
├── wasd/       WASD clips
└── easycom/    EasyCom videos + csv/
```

Copy it from the workstation, for example:

```bash
rsync -a --info=progress2 /Data/data/{ava,videos,csv,wasd,easycom} \
    <user>@lnode01:/mnt/beegfs/data/asd4hri/
```

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
singularity exec --containall --bind /mnt/beegfs/data/asd4hri:/app/data \
    /mnt/beegfs/public/images/asd4hri/asd4hri.sif \
    bash -c "cd /app && python -m src.experiments.run_benchmark --data_dir /app/data --collect_only"
```

To merge the cluster runs with the workstation runs, copy `benchmark/<run>/stats` from one side to
the other and run the same `--collect_only` command there. The tag (`cluster`, `gpu`, `pi`) in the
run name keeps the runs apart. The accuracy of a run does not depend on the machine, but FPS and
the time per stage do.
