# Feature-Optimized Adaptive Detection of Known and Unknown Network Attacks in Smart Home IoT

Code accompanying the manuscript *"Feature-Optimized Adaptive Detection of Known and
Unknown Network Attacks in Smart Home IoT"* (under review, IEEE Internet of Things
Journal). The scripts reproduce the supervised intrusion-detection experiments
reported in the paper: Fisher-score feature optimization, per-model evaluation of six
supervised detection models (BLR, KNN, SVM, DT, RF, XGBoost) under both
knowledge-based and zero-day scenarios, and the temporal attack-initiation analysis.

## Repository contents

| File | Purpose |
|------|---------|
| `01_prepare_blr_dataset.py` | Builds the per-class 80/20 training / evaluation CSVs from the per-MPI benign and malicious feature files. |
| `ids_fisher.py` | Main experiments **with** Fisher-score feature selection (the paper's reported results). Sweeps the feature count and evaluates all six models. |
| `ids.py` | Same experiments **without** feature selection (full-feature baseline). |
| `ml_algorithms_multiclass.py` | Multiclass variant (predicts attack type rather than benign/malicious). |
| `on_time_script.py` | Temporal analysis: how close to attack initiation each interval is detected. |

Each experiment script has a `CONFIG` block at the top — edit the input/output paths
there, then run the script. No command-line arguments are needed.

## Requirements

Python 3.9+ and:

```
pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn skfeature-chappers
```

`skfeature-chappers` provides `skfeature.function.similarity_based.fisher_score`,
used only by `ids_fisher.py`.

## Pipeline order

1. **`01_prepare_blr_dataset.py`** — reads `Benign_.csv` and `Malicious_.csv` (one row
   per 1-second Monitoring Period Interval) and writes `training.csv` and
   `evaluation.csv` with the label column `Class` (0 = benign, 1 = malicious) in the
   first column.
2. **`ids_fisher.py`** — reads a training CSV and an evaluation CSV and, for each
   feature count in `FEATURE_COUNTS`, writes a `fisher_<k>/` folder with metrics,
   confusion-matrix images, and per-interval prediction outputs. Use `ids.py` for the
   full-feature baseline and `ml_algorithms_multiclass.py` for the multiclass variant.
3. **`on_time_script.py`** — reads the per-interval prediction CSVs produced above
   (each must contain the ground-truth `Original` column) and writes `ON_TIME.txt`.

For a quick end-to-end smoke test, set `FEATURE_COUNTS = [10]` in `ids_fisher.py` to
do a single fast pass before running the full 10–80 sweep.

## Reproducibility

All models and cross-validation folds use a fixed `random_state`, so repeated runs on
the same inputs reproduce the reported results. This is intentional: it ensures that
differences between models reflect the models themselves rather than variation in the
random partitioning of the data.

## Data availability

The per-MPI feature CSVs are derived from the public smart-home IoT network-traffic
dataset referenced in the paper. The raw packet captures and the
packet-to-per-second feature-extraction step are not included in this repository. To
run the scripts, provide per-MPI feature CSVs with the label in the first column named
`Class`.

## Citation

If you use this code, please cite the paper:

> C. Ioannou, C. Sergiou, C. Charilaou, and V. Vassiliou, "Feature-Optimized Adaptive
> Detection of Known and Unknown Network Attacks in Smart Home IoT," *IEEE Internet of
> Things Journal* (under review).

## License

Released under the MIT License. See [LICENSE](LICENSE).
