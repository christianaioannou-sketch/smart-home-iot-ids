"""
01_prepare_blr_dataset.py
-------------------------
Builds the training / evaluation CSVs for the BLR (and general ML) experiments.

What it does (unchanged from the original 'BLR analysis.py'):
  1. Reads a benign feature CSV and a malicious feature CSV (one row per 1-s MPI).
  2. Adds a label column: 0 = benign, 1 = malicious.
  3. Aligns the two files to a common set of columns (missing cols filled with 0).
  4. Splits each class 80/20 into train/eval, then concatenates.
  5. Cleans infinities / negative zeros and writes training.csv and evaluation.csv.

HOW TO RUN
  - Put Benign_.csv and Malicious_.csv in the folder set by DATA_DIR below.
  - Set OUTPUT_DIR to where you want the results written.
  - python 01_prepare_blr_dataset.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# CONFIG — edit these two paths for your machine, then run.
# Use a relative folder (".") or an absolute one. Forward slashes work on
# Windows too, e.g. Path("C:/Users/you/data").
# ----------------------------------------------------------------------
DATA_DIR = Path(".")            # where Benign_.csv and Malicious_.csv live
OUTPUT_DIR = Path("./output")   # where training.csv / evaluation.csv go
BENIGN_FILE = "Benign_.csv"
MALICIOUS_FILE = "Malicious_.csv"
LABEL_COL = "Class"             # must match the experiment scripts, which read
                                # df["Class"]; the label is the first column.
TEST_SIZE = 0.20
RANDOM_STATE = 0               

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load and drop the leading index column each file carries.
# ----------------------------------------------------------------------
df_benign = pd.read_csv(DATA_DIR / BENIGN_FILE)
df_malicious = pd.read_csv(DATA_DIR / MALICIOUS_FILE)
df_benign = df_benign.drop(columns=df_benign.columns[0])
df_malicious = df_malicious.drop(columns=df_malicious.columns[0])

# ----------------------------------------------------------------------
# 2. Label: benign = 0, malicious = 1 (inserted as the first column).
# ----------------------------------------------------------------------
df_benign.insert(0, LABEL_COL, 0)
df_malicious.insert(0, LABEL_COL, 1)

# ----------------------------------------------------------------------
# 3. Per-class 80/20 split.
# ----------------------------------------------------------------------
train_benign, test_benign = train_test_split(
    df_benign, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_malicious, test_malicious = train_test_split(
    df_malicious, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ----------------------------------------------------------------------
# 4. Align columns across the two classes (fill any missing with 0),
#    so concatenation keeps a consistent feature set.
# ----------------------------------------------------------------------
def align_columns(a, b):
    all_cols = a.columns.union(b.columns)
    return (a.reindex(columns=all_cols, fill_value=0),
            b.reindex(columns=all_cols, fill_value=0))

train_benign, train_malicious = align_columns(train_benign, train_malicious)
test_benign, test_malicious = align_columns(test_benign, test_malicious)

results_training = pd.concat([train_benign, train_malicious])
results_evaluation = pd.concat([test_benign, test_malicious])

# ----------------------------------------------------------------------
# 5. Clean non-finite values and write out.
# ----------------------------------------------------------------------
results_training = results_training.replace([np.inf, -np.inf, -0.0], 0)
results_evaluation = results_evaluation.replace([np.inf, -np.inf, -0.0], 0)

results_training.to_csv(OUTPUT_DIR / "training.csv", index=False)
results_evaluation.to_csv(OUTPUT_DIR / "evaluation.csv", index=False)
print(f"Wrote {OUTPUT_DIR/'training.csv'}  ({len(results_training)} rows)")
print(f"Wrote {OUTPUT_DIR/'evaluation.csv'} ({len(results_evaluation)} rows)")
