# Results

The interactive results page shows every benchmarked model and setting of every release:
accuracy against speed (frames per second) for the full pipeline and for ground-truth boxes, a
version slider, the progress of each configuration, the time per pipeline stage and a table of
all runs.

**[Open the results page](https://adriandavidauer.github.io/ASD-for-HRI/results/)**

The page is also attached to every release as `results.html`.

## Which runs a release has

All models run on every major release. In between, a release reruns only the configurations
its changes affect, e.g. our own models after retraining them, or one protocol after a pipeline
fix. The baselines from the literature don't have to be rerun every time.

On the page, a configuration (dataset, model, boxes, variant) without a run in the selected
version shows its latest earlier run, marked with `*`; the hover text and the table name the
version the result comes from. Runs are never carried into a new major version, so a model
that was dropped disappears there. A carried run keeps the Pi timing of its own version.

## Where the data comes from

The results only live in the releases; nothing is committed to the repository. Each `v*`
release carries:

| Asset | Content |
|---|---|
| `benchmark.csv` | required: the table `run_benchmark --tag <version>` writes (`benchmark/benchmark_<version>.csv`, renamed) for the runs with that version's image, e.g. on the cluster |
| `benchmark_pi.csv` | optional: the same table of the Raspberry Pi runs; it gives the Pi fps |

A point gets a Pi fps only from a Pi run of the same version, dataset, architecture, settings
and boxes. Without one it has no Pi value: it is left out of the Pi-timing views, and the page
says how many runs have no Pi timing.

When a release is published, the workflow `.github/workflows/results_chart.yml` downloads these
assets from all `v*` releases (oldest first), builds the page with
`python -m src.experiments.make_results_chart` and publishes it. If the CSVs are uploaded
after the release was published, run the workflow by hand (*Actions → Results chart → Run
workflow*).

To build the page locally from your own CSVs:

```bash
python -m src.experiments.make_results_chart \
    --version v0.0.0=benchmark.csv --pi v0.0.0=benchmark_pi.csv -o results.html
```
